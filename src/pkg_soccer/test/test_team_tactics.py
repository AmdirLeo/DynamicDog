import math
import unittest
from unittest.mock import Mock

import numpy as np

import test_blocking_defense as fixtures


class TeamTacticsTest(unittest.TestCase):
    def controller(self, **kwargs):
        kwargs.setdefault('role', 'attack')
        return fixtures.BlockingDefenseTest().controller(auto_phase=True, **kwargs)

    def test_both_defend_different_opponents_and_keep_attack_roles(self):
        for angle in (0., math.pi/2, -math.pi/2, math.pi):
            attack = self.controller(angle=angle, ball=(-.3, 0.))
            assist = self.controller(angle=angle, ball=(-.3, 0.), role='assist',
                                     dog=(-2., 1.), mate=(-.4, 0.))
            for c in (attack, assist):
                c.actor = Mock(side_effect=AssertionError('Defense must bypass PPO'))
                self.assertTrue(np.isfinite(c.command(1.1)).all())
                self.assertEqual(c.state, 'DEFENSE_BLOCK')
                self.assertTrue(c.defending)
                c.actor.assert_not_called()
            self.assertEqual(attack.blocking_defense.target, 'opponent')
            self.assertEqual(assist.blocking_defense.target, 'opponent2')
            self.assertEqual(attack.team_role, 'attack')
            self.assertEqual(assist.team_role, 'assist')

    def test_attack_half_runs_complementary_ppo_networks(self):
        for role in ('attack', 'assist'):
            c = self.controller(role=role, ball=(.3, 0.))
            self.assertIs(c.actor, c.attack_actor if role == 'attack' else c.assist_actor)
            c.actor = Mock(return_value=np.zeros(3))
            c.command(1.1)
            self.assertFalse(c.defending)
            self.assertEqual(c.state, 'RL')
            c.actor.assert_called_once()

    def test_midfield_hysteresis_and_resume_attack_in_same_controller(self):
        c = self.controller(ball=(-.3, 0.))
        c.actor = Mock(return_value=np.zeros(3))
        c.command(1.1)
        self.assertTrue(c.defending)
        for t, bx in ((1.15, -.1), (1.2, .1), (1.25, .3),
                      (1.3, .1), (1.35, -.1), (1.4, -.3)):
            for name in ('dog', 'ball', 'teammate', 'opponent', 'opponent2'):
                pose = getattr(c, name)
                c.update(name, bx if name == 'ball' else pose.x, pose.y,
                         pose.yaw, t, 'world', t)
            c.command(t)
            self.assertEqual(c.defending, t < 1.25 or t >= 1.4)
        self.assertEqual(c.actor.call_count, 3)
        self.assertEqual(c.team_role, 'attack')

    def test_invalid_observations_do_not_change_phase_or_roles(self):
        c = self.controller(ball=(-.3, 0.))
        c.command(1.1)
        c.ball = c.ball._replace(x=2.)
        self.assertEqual(c.command(3.), c.ZERO)
        self.assertEqual(c.team_tactics.phase, 'defense')
        self.assertEqual(c.team_role, 'attack')

    def test_midfield_kickoff_starts_in_attack_and_marks_survive_role_swaps(self):
        c = self.controller(ball=(0., 0.))
        c.command(1.1)
        self.assertFalse(c.defending)
        c.team_role = 'assist'
        self.assertEqual(c.team_tactics.mark, 'opponent')

    def test_fixed_defense_cannot_be_combined_with_auto_phases(self):
        with self.assertRaises(ValueError):
            self.controller(role='defense')


if __name__ == '__main__':
    unittest.main()
