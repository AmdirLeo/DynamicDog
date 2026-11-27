import math
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np

from pkg_soccer.control import Config
from pkg_soccer.teammate_prediction import VelocityTeammateController


class BlockingDefenseTest(unittest.TestCase):
    def controller(self, angle=0., dog=(-.4, 0.), mate=(-2., 1.),
                   role='defense', auto_phase=False, ball=(.15, 0.), continuous=False, fallback=False, speed=.45, gain=1., kickoff=0.):
        co, si = math.cos(angle), math.sin(angle)
        def rotate(x, y):
            return co*x-si*y, si*x+co*y
        gx, gy = rotate(4.585, 0.)
        cfg = Config(goal_x=gx, goal_y=gy, goal_yaw=angle, field_yaw=angle,
                     field_min_x=-5., field_max_x=5., field_min_y=-2.75,
                     field_max_y=2.75)
        package = Path(__file__).resolve().parents[1] / 'pkg_soccer'
        c = VelocityTeammateController(
            cfg, policy='shared_2v2', team_role=role, auto_team_phase=auto_phase, speed_limit=speed, speed_gain=gain,
            allow_tracking_fallback=fallback, kickoff_attack_seconds=kickoff,
            continuous_play=continuous,
            attack_model_path=str(package / 'policy_2v2_attack_actor.npz'),
            assist_model_path=str(package / 'policy_2v2_assist_actor.npz'))
        for t in (1., 1.05, 1.1):
            for name, point in (('dog', dog), ('ball', ball),
                                ('opponent', (0., 0.)), ('opponent2', (2., 1.)),
                                ('teammate', mate)):
                x, y = rotate(*point)
                c.update(name, x, y, angle, t, 'world', t)
        return c

    def test_contact_continues_without_ppo_or_role_switch(self):
        c = self.controller()
        c.actor = Mock(side_effect=AssertionError('Defense must bypass PPO'))
        velocity = c.command(1.1)
        self.assertEqual(c.state, 'DEFENSE_BLOCK')
        self.assertEqual(c.team_role, 'defense')
        self.assertFalse(c.auto_role_switch)
        self.assertGreater(velocity[0], .05)
        self.assertLessEqual(np.linalg.norm(velocity[:2]), .18)
        c.actor.assert_not_called()

    def test_field_rotation_and_swapping_goals_preserve_body_command(self):
        expected = self.controller().command(1.1)
        for angle in (math.pi/2, -math.pi/2, math.pi):
            np.testing.assert_allclose(self.controller(angle).command(1.1), expected, atol=1e-6)

    def test_close_teammate_is_not_pushed(self):
        c = self.controller(mate=(-.2, 0.))
        velocity = c.command(1.1)
        self.assertLessEqual(velocity[0], 1e-7)

    def test_wrong_side_routes_around_opponent_and_respects_limits(self):
        c = self.controller(dog=(.5, 0.))
        velocity = c.command(1.1)
        self.assertLess(velocity[0], 0.)
        self.assertGreater(abs(velocity[1]), .01)
        self.assertLessEqual(abs(velocity[1]), .200001)
        self.assertLessEqual(np.linalg.norm(velocity[:2]), .450001)
        self.assertLessEqual(abs(velocity[2]), .700001)

    def test_mark_hysteresis_and_possession_change(self):
        c = self.controller()
        c.command(1.1)
        self.assertEqual(c.blocking_defense.target, 'opponent')
        c.opponent2 = c.opponent2._replace(x=.25, y=0.)
        c.blocking_defense.command(c)
        self.assertEqual(c.blocking_defense.target, 'opponent')
        c.ball = c.ball._replace(x=2., y=1.)
        c.opponent2 = c.opponent2._replace(x=2., y=1.)
        c.blocking_defense.command(c)
        self.assertEqual(c.blocking_defense.target, 'opponent2')

    def test_stale_tracks_and_paused_play_still_stop(self):
        c = self.controller()
        self.assertEqual(c.command(3.), c.ZERO)
        self.assertEqual(c.state, 'WAIT_POSE')
        c = self.controller()
        c.pause_play(1.1, 'goal')
        self.assertEqual(c.command(1.1), c.ZERO)


if __name__ == '__main__':
    unittest.main()
