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
            attack = self.controller(angle=angle, ball=(-1.3, 0.))
            assist = self.controller(angle=angle, ball=(-1.3, 0.), role='assist',
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

    def test_one_meter_midfield_resumes_attack_in_same_controller(self):
        c = self.controller(ball=(-1.3, 0.))
        c.actor = Mock(return_value=np.zeros(3))
        c.command(1.1)
        self.assertTrue(c.defending)
        for t, bx in ((1.15, -1.1), (1.2, -1.), (1.25, -.8),
                      (1.3, -.6), (1.35, -.9), (1.4, -1.2)):
            for name in ('dog', 'ball', 'teammate', 'opponent', 'opponent2'):
                pose = getattr(c, name)
                c.update(name, bx if name == 'ball' else pose.x, pose.y,
                         pose.yaw, t, 'world', t)
            c.command(t)
            self.assertEqual(c.defending, t < 1.2 or t >= 1.4)
        self.assertEqual(c.actor.call_count, 4)
        self.assertEqual(c.team_role, 'attack')

    def test_entire_midfield_band_attacks_from_defense_in_either_direction(self):
        for angle in (0., math.pi/2, -math.pi/2, math.pi):
            for role in ('attack', 'assist'):
                for bx in (-1., -.5, 0., .5, 1., 2.):
                    c = self.controller(angle=angle, role=role, ball=(bx, 0.))
                    c.team_tactics.phase = 'defense'
                    c.command(1.1)
                    self.assertFalse(c.defending)
                    self.assertEqual(c.state, 'RL')

    def test_invalid_observations_do_not_change_phase_or_roles(self):
        c = self.controller(ball=(-1.3, 0.))
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

    def refresh(self, c, now):
        for t in (now-.1, now-.05, now):
            for name in c.tracks:
                pose = c._observed[name]
                c.update(name, pose.x, pose.y, pose.yaw, t, pose.frame, t)

    def test_kickoff_forces_complementary_attack_then_returns_to_dual_defense(self):
        for role in ('attack', 'assist'):
            c = self.controller(role=role, kickoff=5., ball=(-1.3, 0.))
            c.actor = Mock(return_value=np.zeros(3))
            c.command(1.1)
            self.assertFalse(c.defending)
            self.assertEqual(c.team_role, role)
            self.assertIn('开场强制进攻', c.reason)
            self.refresh(c, 5.9)
            c.command(5.9)
            self.assertFalse(c.defending)
            self.assertEqual(c.team_role, role)
            self.refresh(c, 6.2)
            c.command(6.2)
            self.assertTrue(c.defending)
            self.assertEqual(c.state, 'DEFENSE_BLOCK')

    def test_kickoff_does_not_expire_waiting_for_initial_data_or_repeat_on_rearm(self):
        c = self.controller(kickoff=5., ball=(-1.3, 0.))
        c.command(10.)
        self.assertEqual(c.state, 'WAIT_POSE')
        self.assertIsNone(c.team_tactics.kickoff_started_at)
        self.refresh(c, 20.)
        c.command(20.)
        self.assertFalse(c.defending)
        self.assertEqual(c.team_tactics.kickoff_started_at, 20.)
        c.rearm(30., 'restart')
        self.refresh(c, 30.)
        c.command(30.)
        self.assertTrue(c.defending)
        self.assertEqual(c.team_tactics.kickoff_started_at, 20.)

    def test_kickoff_blocks_early_role_swap_even_with_bad_attacker_position(self):
        c = self.controller(kickoff=5., continuous=True, ball=(0., 0.),
                            dog=(2., 0.), mate=(-.7, 0.))
        c.command(1.1)
        self.refresh(c, 5.9)
        c.command(5.9)
        self.assertEqual(c.team_role, 'attack')
        self.refresh(c, 6.2)
        c.command(6.2)
        self.assertEqual(c.team_role, 'assist')

    def test_kickoff_rejects_invalid_durations(self):
        for duration in (-1., float('nan'), float('inf'), 61.):
            with self.assertRaises(ValueError):
                self.controller(kickoff=duration)


if __name__ == '__main__':
    unittest.main()
