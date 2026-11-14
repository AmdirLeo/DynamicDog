"""Stage-2 PPO deployment: 23 training observations and NumPy actor inference.

The checkpoint uses attack +X, 20 Hz and body-frame actions scaled by
[0.45, 0.20, 0.70]. No online training or stochastic action sampling.
"""

import math
from pathlib import Path

import numpy as np

from .control import Controller, wrap_angle
from .measurement import Track


class Actor:
    """Load numeric weights only; compatible with ROS Python without PyTorch."""

    scale = np.array([0.45, 0.20, 0.70], dtype=np.float32)

    def __init__(self, path=''):
        path = Path(path) if path else Path(__file__).with_name('policy_stage2_actor.npz')
        shapes = [(128, 23), (128, 128), (3, 128)]
        self.layers = []
        with np.load(path, allow_pickle=False) as weights:
            for index, shape in zip((0, 2, 4), shapes):
                w = weights[f'actor_{index}_weight'].copy()
                b = weights[f'actor_{index}_bias'].copy()
                if (w.shape != shape or b.shape != (shape[0],) or
                        not np.isfinite(w).all() or not np.isfinite(b).all()):
                    raise ValueError('RL 模型权重形状或数值无效')
                self.layers.append((w, b))

    def __call__(self, observation):
        x = np.asarray(observation, dtype=np.float32)
        if x.shape != (23,) or not np.isfinite(x).all():
            raise ValueError('RL 输入必须为 23 个有限数值')
        for w, b in self.layers:
            x = np.tanh(w @ x + b)
        if not np.isfinite(x).all():
            raise ValueError('RL 输出包含非有限数值')
        return x


class RLController(Controller):
    """Reuse goal detection, field checks, turn feedback and ROS posture gate."""

    def __init__(self, config, model_path='', speed_limit=0.30):
        super().__init__(config)
        if not math.isfinite(speed_limit) or not 0 < speed_limit <= 0.45:
            raise ValueError('rl_speed_limit 必须在 (0, 0.45] m/s')
        self.actor = Actor(model_path)
        self.speed_limit = speed_limit
        self.tracks = {'dog': Track(3.0), 'ball': Track(12.0)}
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.contact_seen = False

    def update(self, kind, x, y, yaw, now, frame, stamp):
        super().update(kind, x, y, yaw, now, frame, stamp)
        self.tracks[kind].update(now, stamp, x, y, yaw, frame)

    def observation(self, now):
        """Match SoccerEnv2.make_obs; world vectors rotate into the dog frame."""
        d, b, c = self.dog, self.ball, self.cfg
        co, si = math.cos(d.yaw), math.sin(d.yaw)

        def body(x, y):
            return (co*x + si*y, -si*x + co*y)

        rel = body(b.x-d.x, b.y-d.y)
        goal = body(c.goal_x-d.x, c.goal_y-d.y)
        dog_v = self.tracks['dog'].velocity
        ball_v = self.tracks['ball'].velocity
        dv, bv = body(*dog_v[:2]), body(*ball_v[:2])
        fc, fs = math.cos(c.field_yaw), math.sin(c.field_yaw)
        dx, dy = d.x-c.field_center_x, d.y-c.field_center_y
        px, py = fc*dx + fs*dy, -fs*dx + fc*dy
        half_length = ((c.goal_x-c.field_center_x)*fc +
                       (c.goal_y-c.field_center_y)*fs)
        yaw = wrap_angle(d.yaw-c.field_yaw)
        ages = (max(0., now-d.received), max(0., now-b.received))
        obs = [rel[0]/3., rel[1]/3., bv[0]/2., bv[1]/2.,
               goal[0]/5., goal[1]/5., dv[0], dv[1], dog_v[2],
               *self.previous_action, math.sin(yaw), math.cos(yaw),
               (px+half_length)/9., (half_length+.6-px)/9.,
               (py-c.field_min_y)/6., (c.field_max_y-py)/6.,
               math.hypot(*rel)/3., float(self.contact_seen),
               ages[0]/.5, ages[1]/.5, float(max(ages) <= .5)]
        return np.clip(np.asarray(obs, dtype=np.float32), -5., 5.)

    def command(self, now):
        try:
            velocity = super().command(now)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            self.stop('RL 推理失败：' + str(exc))
            velocity = self.ZERO
        # The next observation reflects the command after limits/protection.
        self.previous_action = np.asarray(velocity, dtype=np.float32)/self.actor.scale
        return velocity

    def _command(self, now):
        if self.finished:
            return self.ZERO
        if self.started_at is not None and now-self.started_at >= self.cfg.max_run_time:
            self.stop('RL 达到运行时间上限')
            return self.ZERO
        if (not self.poses_valid(now) or
                not all(track.valid(now) for track in self.tracks.values())):
            self.wait_for_pose('RL 等待连续有效的狗和球位置、速度')
            return self.ZERO
        if self.scored:
            self.state, self.reason = 'GOAL', '球已从场内越过球门线，进球'
            return self.ZERO
        d, b, c = self.dog, self.ball, self.cfg
        if not self.inside_field(d.x, d.y) or not self.inside_field(b.x, b.y):
            self.stop('狗或球已超出设定场地')
            return self.ZERO
        if self.goal_coordinates(b.x, b.y)[0] >= c.ball_radius:
            self.stop('球已在门线后，未确认有效进球，请重新摆球')
            return self.ZERO
        if self.state == 'WAIT_POSE':
            pair = (d.sequence, b.sequence)
            if pair[0] != self.confirm_pair[0] and pair[1] != self.confirm_pair[1]:
                self.confirmed += 1
                self.confirm_pair = pair
            if self.confirmed < 3:
                return self.ZERO
        if self.started_at is None:
            self.started_at = now
        action = self.actor(self.observation(now))
        # Training accumulates estimated contact from the previous observation.
        self.distance = math.hypot(b.x-d.x, b.y-d.y)
        self.contact_seen |= self.distance < .43
        velocity = action*self.actor.scale
        speed = float(np.linalg.norm(velocity[:2]))
        if speed > self.speed_limit:
            velocity[:2] *= self.speed_limit/speed
        velocity[2] = np.clip(velocity[2], -c.turn_speed, c.turn_speed)
        co, si = math.cos(d.yaw), math.sin(d.yaw)
        wx, wy = co*velocity[0]-si*velocity[1], si*velocity[0]+co*velocity[1]
        # Include observed momentum when checking the next half second.
        for vx, vy in ((wx, wy), self.tracks['dog'].velocity[:2]):
            if not self.inside_field(d.x+float(vx)*.5, d.y+float(vy)*.5):
                self.stop('RL 预计运动将越过场地边界')
                return self.ZERO
        self.state, self.reason = 'RL', 'PPO 第二阶段策略控制'
        self.target_yaw = None  # PPO outputs angular velocity, not a target heading.
        return tuple(float(v) for v in velocity)
