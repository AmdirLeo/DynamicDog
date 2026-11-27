import math
import unittest
from unittest.mock import Mock

import numpy as np

import test_blocking_defense as fixtures


class TrackingFallbackTest(unittest.TestCase):
    def controller(self, **kwargs):
        kwargs.setdefault('role', 'attack')
        c = fixtures.BlockingDefenseTest().controller(
            continuous=True, fallback=True, **kwargs)
        c.actor = Mock(return_value=np.array([.8, 0., .5]))
        return c

    def remove(self, c, name):
        c._observed[name] = None
        setattr(c, name, None)
        c.tracks[name].invalidate('missing')

    def test_missing_peers_never_hold_ball_based_control(self):
        for role in ('attack', 'assist'):
            for missing in (('teammate',), ('opponent',), ('opponent2',),
                            ('teammate', 'opponent', 'opponent2')):
                c = self.controller(role=role)
                for name in missing:
                    self.remove(c, name)
                command = c.command(1.1)
                self.assertEqual(c.state, 'TRACKING_FALLBACK')
                self.assertGreater(np.linalg.norm(command[:2]), 0.)
                self.assertLessEqual(np.linalg.norm(command[:2]), .200001)
                c.actor.assert_not_called()

    def test_velocity_not_ready_uses_rules_then_returns_to_ppo(self):
        c = self.controller()
        for track in c.tracks.values():
            track.invalidate('warmup')
        self.assertGreater(np.linalg.norm(c.command(1.1)[:2]), 0.)
        self.assertEqual(c.state, 'TRACKING_FALLBACK')
        c.actor.assert_not_called()
        for t in (1.15, 1.20, 1.25):
            for name in c.tracks:
                p = c._observed[name]
                c.update(name, p.x, p.y, p.yaw, t, p.frame, t)
        c.command(1.25)
        self.assertEqual(c.state, 'RL')
        c.actor.assert_called_once()

    def test_actual_peer_expiry_does_not_stop_when_dog_ball_keep_arriving(self):
        c = self.controller()
        for t in np.arange(1.15, 2.16, .05):
            for name in ('dog', 'ball'):
                p = c._observed[name]
                c.update(name, p.x, p.y, p.yaw, float(t), p.frame, float(t))
        c.command(float(t)+1e-6)
        self.assertEqual(c.state, 'TRACKING_FALLBACK')
        self.assertIsNone(c.teammate)
        c.actor.assert_not_called()

    def test_missing_defensive_mark_uses_ball_lane_otherwise_keeps_blocking(self):
        for marked_available in (False, True):
            c = self.controller(auto_phase=True, ball=(-1.3, 0.))
            self.remove(c, 'teammate')
            self.remove(c, 'opponent2')
            if not marked_available:
                self.remove(c, 'opponent')
            else:
                c.tracks['opponent'].invalidate('velocity_warmup')
            command = c.command(1.1)
            self.assertTrue(np.isfinite(command).all())
            self.assertEqual(c.state, 'TRACKING_FALLBACK')
            self.assertTrue(c.defending)
            self.assertLessEqual(np.linalg.norm(command[:2]), .200001)
            c.actor.assert_not_called()

    def test_core_loss_mismatch_or_fault_still_stops(self):
        for missing in ('dog', 'ball'):
            c = self.controller()
            self.remove(c, missing)
            self.assertEqual(c.command(1.1), c.ZERO)
            self.assertIn(missing, c.reason)
        c = self.controller()
        c._observed['ball'] = c.ball._replace(frame='different_world')
        self.assertEqual(c.command(1.1), c.ZERO)
        self.assertIn('坐标系不一致', c.reason)
        c = self.controller()
        c.stop('hardware_fault')
        self.assertEqual(c.command(1.1), c.ZERO)

    def test_mismatched_peer_frame_is_ignored_and_field_rotation_is_consistent(self):
        commands = []
        for angle in (0., math.pi/2, math.pi):
            c = self.controller(angle=angle)
            c._observed['opponent'] = c.opponent._replace(frame='other')
            commands.append(c.command(1.1))
            self.assertEqual(c.state, 'TRACKING_FALLBACK')
        for command in commands[1:]:
            np.testing.assert_allclose(command, commands[0], atol=1e-6)

    def test_a1_gain_reaches_higher_translation_without_faster_turning(self):
        c = self.controller(speed=.60, gain=0.)
        velocity = c.command(1.1)
        self.assertAlmostEqual(velocity[0], .48, places=5)
        self.assertAlmostEqual(velocity[2], .35, places=5)
        self.assertEqual(c.state, 'RL')
        self.remove(c, 'opponent')
        self.assertLessEqual(np.linalg.norm(c.command(1.1)[:2]), .200001)


if __name__ == '__main__':
    unittest.main()
