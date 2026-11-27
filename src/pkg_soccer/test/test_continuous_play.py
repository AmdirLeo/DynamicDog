import unittest
from unittest.mock import Mock

import numpy as np

from pkg_soccer.stand_guard import StandGuard
import test_blocking_defense as fixtures


class ContinuousPlayTest(unittest.TestCase):
    def controller(self, **kwargs):
        c = fixtures.BlockingDefenseTest().controller(
            role='attack', continuous=True, **kwargs)
        c.actor = Mock(return_value=np.array([.5, 0., 0.]))
        return c

    def assert_running(self, c):
        command = c.command(1.1)
        self.assertGreater(command[0], 0.)
        self.assertEqual(c.state, 'RL')
        self.assertFalse(c.finished)
        self.assertFalse(c.paused)

    def test_goal_is_counted_without_waiting_or_recounting(self):
        c = self.controller()
        c.scored = True
        self.assert_running(c)
        self.assertEqual(c.goals, 1)
        self.assert_running(c)
        self.assertEqual(c.goals, 1)

    def test_current_and_predicted_boundaries_no_longer_lock(self):
        for dog, ball in (((5.1, 0.), (.15, 0.)),
                          ((-.4, 0.), (5.2, 0.)),
                          ((-.4, 0.), (4.8, 0.)),
                          ((-.4, 0.), (-4.8, 0.)),
                          ((4.95, 0.), (4.5, 0.))):
            self.assert_running(self.controller(dog=dog, ball=ball))

    def test_referee_pause_and_timeout_do_not_stop(self):
        c = self.controller()
        c.pause_play(1., '等待裁判')
        c._resume_ready = Mock(side_effect=AssertionError('Must not wait for restart'))
        self.assert_running(c)
        c.started_at = -10000.
        self.assert_running(c)
        c = self.controller(ball=(0., 2.5))
        c.continue_deadzone = False
        c.referee_wait = Mock(side_effect=AssertionError('Must not wait in deadzone'))
        self.assert_running(c)

    def test_turn_progress_does_not_lock_continuous_mode(self):
        c = self.controller()
        c.actor = Mock(return_value=np.array([.5, 0., .5]))
        c.turn_started = c.turn_checkpoint = -20.
        c.turn_checkpoint_yaw = c.dog.yaw
        c.turn_sign = 1.
        c.turn_origin = (c.dog.x, c.dog.y)
        c.turn_response_until = 0.
        self.assert_running(c)

    def test_tracking_loss_and_explicit_stop_remain_effective(self):
        c = self.controller()
        self.assertEqual(c.command(3.), c.ZERO)
        self.assertEqual(c.state, 'WAIT_POSE')
        c.actor.assert_not_called()
        c = self.controller()
        c.stop('用户退出或硬件故障')
        self.assertEqual(c.command(1.1), c.ZERO)
        self.assertTrue(c.finished)
        c.actor.assert_not_called()

    def guard(self, recover=True):
        g = StandGuard(305, recover_status_timeout=recover)
        for t in (1., 1.05, 1.1):
            g.update(305, 0, 100, True, t)
        self.assertEqual(g.command(1.1), 'WALK')
        return g

    def test_status_timeout_recovers_in_timer_and_callback_orders(self):
        for timer_first in (False, True):
            g = self.guard()
            if timer_first:
                self.assertEqual(g.command(3.), 'WAIT')
                self.assertEqual(g.state, 'WAIT_POSTURE')
            g.update(305, 0, 100, True, 3.1)
            self.assertEqual(g.command(3.1), 'WALK')

    def test_fault_or_changed_posture_never_auto_recovers(self):
        for motion, healthy, switch in ((305, False, 0), (101, True, 0), (305, True, 14)):
            g = self.guard()
            g.update(motion, switch, 100, healthy, 3.)
            self.assertEqual(g.command(3.), 'WAIT')
            self.assertEqual(g.state, 'POSTURE_FAILED')
            g.update(305, 0, 100, True, 3.05)
            self.assertEqual(g.command(3.05), 'WAIT')
        g = self.guard(recover=False)
        self.assertEqual(g.command(3.), 'WAIT')
        self.assertEqual(g.state, 'POSTURE_FAILED')


if __name__ == '__main__':
    unittest.main()
