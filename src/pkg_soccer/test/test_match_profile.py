import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from pkg_soccer.control import Config
from pkg_soccer.rl_control import Actor
from pkg_soccer.teammate_prediction import VelocityTeammateController

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'pkg_soccer'


class MatchProfileTest(unittest.TestCase):
    def test_shipped_pair_is_the_evaluated_checkpoint(self):
        expected = {
            'attack': '1dc4f550a1d9bb8a41c9b7b41d91497ac942de654abfb4523802b13841c17d06',
            'assist': '4a233f145390299e7d28b56d6cacf19ae294be868d3c48080dbafb26dc0b490e',
        }
        for role, digest in expected.items():
            path = PACKAGE / ('policy_2v2_' + role + '_actor.npz')
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            actor = Actor(path)
            self.assertEqual(actor.role, role)
            self.assertEqual(actor.obs_dim, 41)
            self.assertEqual(actor.peer_order, 'opponent_first')
            self.assertEqual([w.shape[0] for w, _ in actor.layers], [256] * 6 + [3])
            np.testing.assert_allclose(actor.scale, [.45, .20, .70])
            np.testing.assert_allclose(actor.contact_offset, [.10, 0.])
            self.assertTrue(np.isfinite(actor(np.zeros(41))).all())

    def controller(self, role):
        return VelocityTeammateController(
            Config(), policy='shared_2v2', speed_limit=.45, speed_gain=1., team_role=role,
            attack_model_path=str(PACKAGE / 'policy_2v2_attack_actor.npz'),
            assist_model_path=str(PACKAGE / 'policy_2v2_assist_actor.npz'))

    def test_role_selection_and_constant_velocity_prediction(self):
        for role in ('attack', 'assist'):
            c = self.controller(role)
            self.assertIs(c.actor, c.attack_actor if role == 'attack' else c.assist_actor)
            np.testing.assert_allclose(c.effective_scale, [.45, .20, .70])
            for step in range(7):
                t = 1. + .05 * step
                c.update_peer('teammate', .2*t + (.008 if step % 2 else -.008),
                              0., 0., t, 'sim', t)
            raw = c._observed['teammate']
            now = t + .5
            c._predict_missing(now)
            self.assertIn('teammate', c._predicted)
            self.assertAlmostEqual(c.teammate.x, raw.x + c.tracks['teammate'].velocity[0] * .5)
            first = c.teammate
            c._predict_missing(now)
            self.assertEqual(c.teammate, first)
            c._predict_missing(t + 1.1)
            self.assertIsNone(c.teammate)

    def launch(self, arguments, motion='true', name='robot_test', topics=None, team_role=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'install').mkdir()
            (root / 'install/setup.bash').write_text('test -z "${COLCON_TRACE}" || true\n')
            executable = root / 'ros2'
            executable.write_text(
                '#!/usr/bin/env python3\nimport os,sys,json\n'
                'if sys.argv[1:] == ["topic", "list"]:\n'
                ' print(os.environ.get("MOCK_MOTION_TOPICS", ""))\n'
                'else:\n print("CAPTURED="+json.dumps(sys.argv[1:]))\n')
            executable.chmod(0o755)
            env = os.environ.copy()
            for key in ('DOG_NAME', 'TEAM_A_1', 'TEAM_A_2', 'ATTACK_GOAL', 'COLCON_TRACE', 'TEAM_ROLE'):
                env.pop(key, None)
            env.update(A_WS=str(root), PATH=str(root) + os.pathsep + env['PATH'],
                       PYTHONPATH=str(ROOT), ENABLE_MOTION=motion,
                       RL_MODEL_PATH='/obsolete/model.npz')
            if name is not None:
                env['DOG_NAME'] = name
            if team_role is not None:
                env['TEAM_ROLE'] = team_role
            if topics is not None:
                env['MOCK_MOTION_TOPICS'] = topics
            return subprocess.run(['bash', str(ROOT / 'start_match.sh')] + arguments,
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True, timeout=20)

    def arguments(self, result):
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        line = next(line for line in result.stdout.splitlines() if line.startswith('CAPTURED='))
        return json.loads(line.split('=', 1)[1])

    def test_launcher_roles_opponents_goal_and_exact_model_paths(self):
        for role, own, mate, initial in (
                ('A1', 'xll_dog', 'Lizj_dog', 'attack'),
                ('A2', 'Lizj_dog', 'xll_dog', 'assist')):
            args = self.arguments(self.launch([role, '/vrpn/rival_one/pose', '/vrpn/rival_two/pose', 'goal_left']))
            for argument in ('dog_pose_topic:=/vrpn/' + own + '/pose',
                             'teammate_pose_topic:=/vrpn/' + mate + '/pose',
                             'team_role:=' + initial, 'attack_goal:=goal_left',
                             'opponent_pose_topic:=/vrpn/rival_one/pose',
                             'opponent2_pose_topic:=/vrpn/rival_two/pose',
                             'rl_teammate_prediction_model:=regression_velocity',
                             'rl_auto_team_phase:=true', 'rl_continuous_play:=true', 'rl_fast_start:=true',
                             'rl_speed_limit:=' + ('0.60' if role == 'A1' else '0.45'),
                             'rl_speed_gain:=0.0', 'rl_tracking_fallback:=true',
                             'enable_motion:=true', 'dog_name:=robot_test'):
                self.assertIn(argument, args)
            for model_role in ('attack', 'assist'):
                self.assertIn('rl_' + model_role + '_model_path:=' +
                              str(PACKAGE / ('policy_2v2_' + model_role + '_actor.npz')), args)
            self.assertFalse(any(value.endswith(':=') for value in args))
            self.assertNotIn('rl_model_path:=/obsolete/model.npz', args)

    def test_fixed_defense_and_attack_roles(self):
        argv = ['A2', '/vrpn/rival_one/pose', '/vrpn/rival_two/pose']
        for role in ('defense', 'attack', 'assist'):
            args = self.arguments(self.launch(argv, team_role=role))
            self.assertIn('team_role:=' + role, args)
            self.assertIn('rl_auto_role_switch:=false', args)
            self.assertIn('rl_auto_team_phase:=false', args)
        self.assertIn('rl_auto_role_switch:=true', self.arguments(self.launch(argv)))
        self.assertEqual(self.launch(argv, team_role='invalid').returncode, 2)

    def test_motion_namespace_must_be_unique_and_dry_run_can_omit_it(self):
        argv = ['A1', '/vrpn/rival_one/pose', '/vrpn/rival_two/pose']
        args = self.arguments(self.launch(argv, name=None, topics='/actual_robot/motion_status'))
        self.assertIn('dog_name:=actual_robot', args)
        ambiguous = self.launch(argv, name=None, topics='/one/motion_status\n/two/motion_status')
        self.assertEqual(ambiguous.returncode, 2)
        self.assertNotIn('CAPTURED=', ambiguous.stdout)
        args = self.arguments(self.launch(argv, name=None, motion='false'))
        self.assertIn('enable_motion:=false', args)
        self.assertFalse(any(value.startswith('dog_name:=') for value in args))

    def test_invalid_or_duplicate_topics_never_start_node(self):
        for args in (['A1'], ['A1', '$/vrpn/a/pose', '/vrpn/b/pose'],
                     ['A1', '/vrpn/xll_dog/pose', '/vrpn/b/pose']):
            result = self.launch(args)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn('CAPTURED=', result.stdout)


if __name__ == '__main__':
    unittest.main()
