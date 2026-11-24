"""Regression checks for restarts, rewards and role-policy deployment.

Run from src/pkg_soccer: python -m unittest train.test_training_contract
"""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pkg_soccer.control import Config
from pkg_soccer.rl_control import RLController
from .team_env import EnvConfig, TeamEnv


class TrainingContract(unittest.TestCase):
    def test_collision_penalty_only_for_involved_teams(self):
        env = TeamEnv(EnvConfig(shaping_ball=0, shaping_stage=0, reward_step=0,
                                reward_proximity=0, reward_collision=-.22))
        # The opponent's internal collision must not penalize the observing team.
        for positions, expected in (
                ([(-3, 1), (-3, -1), (2, 0), (2.3, 0)], [0, 0, -.22, -.22]),
                ([(-2, 0), (-1.7, 0), (3, 1), (3, -1)], [-.22, -.22, 0, 0]),
                ([(-.1, 0), (-3, -1), (.1, 0), (3, 1)], [-.22]*4),
                ([(-3, 1), (-3, -1), (3, 1), (3, -1)], [0]*4)):
            env.pos[:] = positions
            env.colliding = env._dog_collisions()
            np.testing.assert_allclose(env._reward([], []), expected, atol=1e-6)

    def test_restarts_are_local_and_playable(self):
        env = TeamEnv(seed=219)
        for origin in ((0, 2.8), (0, -2.8), (5.1, 1.1), (-5.1, -1.1),
                       (5.2, 2.9), (-5.2, -2.9)):
            for _ in range(100):
                env.ball[:] = origin
                env.pending = ('dead', None)
                env.reset()
                self.assertEqual(env._events(env.ball.copy()), [])
                self.assertLess(np.linalg.norm(env.ball - origin), 2.2)
                _, _, done, info = env.step(np.zeros((4, 3)))
                self.assertFalse(done, info)

    def test_reward_prefers_ball_progress_in_both_directions(self):
        env = TeamEnv(EnvConfig(shaping_stage=0, gamma_shaping=1))
        for team, sign in ((0, 1), (1, -1)):
            env.ball[:] = 0
            before = env._team_potentials()
            env.ball[0] = sign * 0.1
            env.prev_phi = before.copy()
            toward = env._reward([], [])[2*team]
            env.ball[0] = -sign * 0.1
            env.prev_phi = before.copy()
            away = env._reward([], [])[2*team]
            self.assertGreater(toward, 0)
            self.assertLess(away, 0)

    def test_team_stage_reward_does_not_pull_assist_onto_ball(self):
        env = TeamEnv(EnvConfig(shaping_ball=0, gamma_shaping=1))
        env.ball[:] = 0
        env.pos[0] = (-0.38, 0)
        env.pos[1] = (-1.8, 1.1)
        before = env._team_potentials()[0]
        env.pos[1] = (-0.39, 0)
        self.assertAlmostEqual(before, env._team_potentials()[0])

    def test_terminal_potential_is_zero(self):
        env = TeamEnv(EnvConfig(shaping_stage=0, reward_goal=0, reward_concede=0,
                                reward_step=0))
        env.ball[:] = 0
        before = env._team_potentials()
        env.prev_phi = before.copy()
        env.ball[0] = 4.7
        reward = env._reward([('goal', 0)], [])
        np.testing.assert_allclose(reward[[0, 2]], -before, atol=1e-6)

    def test_timeout_reset_starts_live_episode_with_clean_tracks(self):
        env = TeamEnv(EnvConfig(max_steps=1), seed=11)
        _, _, done, _ = env.step(np.zeros((4, 3)))
        self.assertTrue(done)
        self.assertEqual(env.pending, ('timeout', None))
        env.reset()
        self.assertEqual(env._events(env.ball.copy()), [])
        for track in env.tracks + [env.ball_track]:
            np.testing.assert_allclose(track.velocity, (0, 0, 0), atol=1e-9)

    def test_training_deployment_observations_both_goals(self):
        cfg = Config(goal_x=4.585, goal_width=1.5, field_min_x=-5, field_max_x=5,
                     field_min_y=-2.75, field_max_y=2.75)
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for role in ('attack', 'assist'):
                path = Path(tmp) / (role + '.npz')
                np.savez(path, actor_0_weight=np.zeros((3, 41), np.float32),
                         actor_0_bias=np.zeros(3, np.float32), role=np.str_(role),
                         action_scale=np.array([.45, .2, .7], np.float32),
                         contact_offset=np.array([.1, 0], np.float32))
                paths.append(str(path))
            for team in (0, 1):
                env = TeamEnv(EnvConfig(randomize=False, mocap_noise=0, yaw_noise=0))
                env.pos[:] = [(-.5, .4), (-1.5, -.9), (2, .1), (.7, 1.2)]
                env.yaw[:] = [.2, .8, -1., -2.]
                env.ball[:] = (.3, -.5)
                env._prime_tracks()
                index = team * 2
                goal = env.attack_goal_of(index)
                c = RLController(cfg._replace(goal_x=goal[0], goal_yaw=team*math.pi,
                                              field_yaw=team*math.pi),
                                 speed_limit=.45, policy='shared_2v2',
                                 attack_model_path=paths[0], assist_model_path=paths[1])
                enemies = [j for j in range(4) if j // 2 != team]
                # Deliberately give the farther enemy the first topic slot.
                enemies.sort(key=lambda j: -np.linalg.norm(env.pos[j]-env.pos[index]))
                for t in (.9, .95, 1.):
                    for name, j in (('dog', index), ('teammate', index+1),
                                    ('opponent', enemies[0]), ('opponent2', enemies[1])):
                        c.update(name, *env.pos[j], env.yaw[j], t, 'sim', t)
                    c.update('ball', *env.ball, 0., t, 'sim', t)
                actual = c.observation(1.075)
                expected = env.observations()[index]
                np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_shared_default_stays_shared(self):
        c = RLController(Config(), speed_limit=.45, policy='shared_2v2')
        self.assertEqual(c.actor.role, 'shared_2v2')
        self.assertEqual(c.actor.peer_order, 'teammate_first')
        with self.assertRaises((ValueError, FileNotFoundError)):
            RLController(Config(), speed_limit=.45, policy='shared_2v2',
                         attack_model_path='/missing/attack.npz',
                         assist_model_path='/missing/assist.npz')


if __name__ == '__main__':
    unittest.main()
