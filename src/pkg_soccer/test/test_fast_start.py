from pathlib import Path
import unittest

from pkg_soccer.control import Config
from pkg_soccer.measurement import Track
from pkg_soccer.stand_guard import StandGuard
from pkg_soccer.teammate_prediction import VelocityTeammateController


class FastStartTest(unittest.TestCase):
    def test_one_healthy_standing_frame_starts_but_resting_requires_service(self):
        for motion in (111, 112, 305):
            g = StandGuard(305, fast_start=True)
            self.assertEqual(g.command(1.), 'WAIT')
            g.update(motion, 0, 100, True, 1.)
            self.assertEqual(g.command(1.), 'WALK')
        g = StandGuard(305, fast_start=True)
        g.update(101, 0, 100, True, 1.)
        self.assertEqual(g.command(1.), 'REQUEST')
        g.requested(1.)
        g.update(111, 0, 100, True, 1.1)
        self.assertEqual(g.command(1.1), 'WAIT')
        g.result(True, 111, 0)
        self.assertEqual(g.command(1.1), 'WALK')

    def test_incomplete_or_faulted_standing_cannot_start(self):
        for motion, progress, healthy, switch in ((111, 50, True, 0),
                                                  (112, 100, False, 0),
                                                  (112, 100, True, 14)):
            g = StandGuard(305, fast_start=True)
            g.update(motion, switch, progress, healthy, 1.)
            self.assertEqual(g.command(1.), 'WAIT')

    def test_bootstrap_velocity_is_replaced_by_measurement(self):
        t = Track(3., fast_start=True)
        t.update(1., 1., 0., 0., 0., 'world')
        self.assertTrue(t.valid(1.))
        self.assertEqual(t.velocity, (0., 0., 0.))
        self.assertFalse(t.valid(1.3))
        t.update(1.05, 1.05, .025, 0., 0., 'world')
        t.update(1.1, 1.1, .05, 0., 0., 'world')
        self.assertEqual(t.quality, 'ok')
        self.assertAlmostEqual(t.velocity[0], .5)

    def test_bootstrap_does_not_mask_a_gap_jump_or_invalid_sample(self):
        for now, x in ((1.3, 0.), (1.05, 10.), (1.05, float('nan'))):
            t = Track(3., fast_start=True)
            t.update(1., 1., 0., 0., 0., 'world')
            t.update(now, now, x, 0., 0., 'world')
            self.assertFalse(t.valid(now))
        t = Track(3.)
        t.update(1., 1., 0., 0., 0., 'world')
        self.assertFalse(t.valid(1.))

    def test_controller_runs_after_first_complete_pose_set(self):
        package = Path(__file__).resolve().parents[1] / 'pkg_soccer'
        c = VelocityTeammateController(
            Config(), policy='shared_2v2', speed_limit=.45,
            continuous_play=True, fast_start=True,
            attack_model_path=str(package / 'policy_2v2_attack_actor.npz'),
            assist_model_path=str(package / 'policy_2v2_assist_actor.npz'))
        points = [('dog', -.5, 0.), ('ball', 0., 0.), ('teammate', -1., 1.),
                  ('opponent', 1., 1.), ('opponent2', 1., -1.)]
        for name, x, y in points[:-1]:
            c.update(name, x, y, 0., 1., 'world', 1.)
        self.assertEqual(c.command(1.), c.ZERO)
        self.assertEqual(c.state, 'WAIT_POSE')
        name, x, y = points[-1]
        c.update(name, x, y, 0., 1., 'world', 1.)
        c.command(1.)
        self.assertEqual(c.state, 'RL')
        self.assertEqual(c.command(1.3), c.ZERO)
        self.assertEqual(c.state, 'WAIT_POSE')


if __name__ == '__main__':
    unittest.main()
