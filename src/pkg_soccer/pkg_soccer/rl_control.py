"""PPO deployment: single-dog (23) and opponent-aware duel (29) inference.

The checkpoint uses attack +X, 20 Hz and body-frame actions scaled by
[0.45, 0.20, 0.70]. No online training or stochastic action sampling.
"""

import math
from pathlib import Path

import numpy as np

from .control import Controller, Pose, wrap_angle
from .measurement import Track


class Actor:
    """Load numeric weights only; compatible with ROS Python without PyTorch."""

    scale = np.array([0.45, 0.20, 0.70], dtype=np.float32)

    def __init__(self, path=''):
        path = Path(path) if path else Path(__file__).with_name('policy_stage2_actor.npz')
        with np.load(path, allow_pickle=False) as weights:
            first = weights['actor_0_weight']
            if first.ndim != 2 or first.shape[0] != 128:
                raise ValueError('RL 模型第一层形状无效')
            self.obs_dim = int(first.shape[1])
            if self.obs_dim not in (23, 29):
                raise ValueError('RL 模型观测维度必须为 23 或 29')
            self.role = str(weights['role'].item()) if 'role' in weights else 'stage2'
            if self.obs_dim == 29:
                if (self.role not in ('attacker', 'defender') or
                        'action_scale' not in weights or
                        not np.allclose(weights['action_scale'], self.scale)):
                    raise ValueError('29 维权重缺少角色或动作比例不匹配')
            shapes = [(128, self.obs_dim), (128, 128), (3, 128)]
            self.layers = []
            for index, shape in zip((0, 2, 4), shapes):
                w = weights[f'actor_{index}_weight'].copy()
                b = weights[f'actor_{index}_bias'].copy()
                if (w.shape != shape or b.shape != (shape[0],) or
                        not np.isfinite(w).all() or not np.isfinite(b).all()):
                    raise ValueError('RL 模型权重形状或数值无效')
                self.layers.append((w, b))

    def __call__(self, observation):
        x = np.asarray(observation, dtype=np.float32)
        if x.shape != (self.obs_dim,) or not np.isfinite(x).all():
            raise ValueError(f'RL 输入必须为 {self.obs_dim} 个有限数值')
        for w, b in self.layers:
            x = np.tanh(w @ x + b)
        if not np.isfinite(x).all():
            raise ValueError('RL 输出包含非有限数值')
        return x


class RLController(Controller):
    """Reuse goal detection, field checks, turn feedback and ROS posture gate."""

    def __init__(self, config, model_path='', speed_limit=0.30, policy='stage2'):
        super().__init__(config)
        if policy not in ('stage2', 'attacker', 'defender'):
            raise ValueError('rl_policy 必须为 stage2/attacker/defender')
        self.policy = policy
        self.duel = policy != 'stage2'
        if not math.isfinite(speed_limit) or not 0 < speed_limit <= 0.45:
            raise ValueError('rl_speed_limit 必须在 (0, 0.45] m/s')
        if not model_path and self.duel:
            model_path = Path(__file__).with_name(f'policy_duel_{policy}_actor.npz')
        self.actor = Actor(model_path)
        if (self.actor.obs_dim != (29 if self.duel else 23) or
                self.actor.role != policy):
            raise ValueError('rl_policy 与模型观测维度或角色不一致')
        self.speed_limit = speed_limit
        self.tracks = {'dog': Track(3.0), 'ball': Track(12.0)}
        if self.duel:
            self.tracks['opponent'] = Track(3.0)
        self.opponent = None
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.contact_seen = False
        self.referee_origin = None
        self.referee_target = None
        self.referee_moved = False
        self.referee_settled = None
        self.deadlock_since = None
        from .attack import AttackPlanner
        self.attack_planner = AttackPlanner() if policy == 'attacker' else None

    def update(self, kind, x, y, yaw, now, frame, stamp):
        if kind == 'opponent':
            self.update_opponent(x, y, yaw, now, frame, stamp)
            return
        if kind == 'ball' and self.duel:
            previous = self.ball
            # Referee placement can jump across a goal line. Do not count a
            # discontinuous or stale trajectory as a goal or retain touch state.
            if previous is not None:
                dt = now-previous.received
                discontinuous = (dt <= 0 or dt > self.tracks['ball'].gap or
                                 previous.frame != frame or
                                 math.hypot(x-previous.x, y-previous.y) > 12.*dt+.03)
                if discontinuous:
                    self.contact_seen = False
                    self.wait_for_pose('球轨迹不连续，等待重发球/定位恢复')
        super().update(kind, x, y, yaw, now, frame, stamp)
        self.tracks[kind].update(now, stamp, x, y, yaw, frame)

    def update_opponent(self, x, y, yaw, now, frame, stamp):
        if not self.duel:
            raise ValueError('单狗模式不接收对手位姿')
        if not all(math.isfinite(v) for v in (x, y, yaw, now, stamp)) or stamp < 0:
            self.opponent = None
            self.tracks['opponent'].invalidate('invalid_pose')
            return
        previous = self.opponent
        sequence = previous.sequence + 1 if previous is not None else 1
        self.opponent = Pose(x, y, yaw, now, frame, stamp, sequence)
        self.tracks['opponent'].update(now, stamp, x, y, yaw, frame)

    def poses_valid(self, now):
        if not super().poses_valid(now):
            return False
        if not self.duel:
            return True
        o = self.opponent
        return (o is not None and o.frame == self.dog.frame and
                0 <= now-o.received <= self.cfg.pose_timeout)

    def observation(self, now):
        """Match SoccerEnv2.make_obs; world vectors rotate into the dog frame."""
        d, b, c = self.dog, self.ball, self.cfg
        co, si = math.cos(d.yaw), math.sin(d.yaw)

        def body(x, y):
            return (co*x + si*y, -si*x + co*y)

        rel = body(b.x-d.x, b.y-d.y)
        # attack_goal always denotes the opponent goal, including for defense.
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
        if self.actor.obs_dim == 29:
            o = self.opponent
            if o is None or not self.tracks['opponent'].valid(now):
                raise ValueError('等待连续有效的对手位姿')
            op = body(o.x-d.x, o.y-d.y)
            ov_world = (self.tracks['opponent'].velocity[0] - dog_v[0],
                        self.tracks['opponent'].velocity[1] - dog_v[1])
            ov = body(*ov_world)
            oyaw = wrap_angle(o.yaw-d.yaw)
            # DuelEnv uses the outer walls, not the goal lines, for these four
            # distances. Its two age slots share one tracking packet age.
            obs[14:18] = ((px-c.field_min_x)/9., (c.field_max_x-px)/9.,
                          (py-c.field_min_y)/6., (c.field_max_y-py)/6.)
            age = max(*ages, max(0., now-o.received))
            obs[20:23] = (age/.5, age/.5, float(age <= .5))
            obs.extend((op[0]/3., op[1]/3., ov[0]/2., ov[1]/2.,
                        math.sin(oyaw), math.cos(oyaw)))
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

    def field_coordinates(self, x, y):
        c = self.cfg
        dx, dy = x-c.field_center_x, y-c.field_center_y
        co, si = math.cos(c.field_yaw), math.sin(c.field_yaw)
        return co*dx+si*dy, -si*dx+co*dy

    def world_coordinates(self, fx, fy):
        """Convert field axes (attack direction, lateral) to mocap axes."""
        c = self.cfg
        co, si = math.cos(c.field_yaw), math.sin(c.field_yaw)
        return (c.field_center_x + co*fx - si*fy,
                c.field_center_y + si*fx + co*fy)

    def referee_wait(self, now):
        """Stop for physical referee replacement; never teleport real observations.

        Training assumptions: ball center in dead strip, goal corridor excluded;
        8 seconds of close, slow dog contact. Resume after the ball is placed
        near the nearest service point and tracked stationary for 0.5 seconds.
        """
        c, b, d, o = self.cfg, self.ball, self.dog, self.opponent
        bx, by = self.field_coordinates(b.x, b.y)
        dead = (by < c.field_min_y+.8 or by > c.field_max_y-.8 or
                ((bx < c.field_min_x+.8 or bx > c.field_max_x-.8) and
                 abs(by) > c.goal_width/2))
        stuck = (math.hypot(o.x-d.x, o.y-d.y) < .56 and
                 self.tracks['dog'].speed < .08 and self.tracks['opponent'].speed < .08)
        if self.referee_origin is None:
            self.deadlock_since = (now if self.deadlock_since is None else
                                   self.deadlock_since) if stuck else None
            deadlock = self.deadlock_since is not None and now-self.deadlock_since >= 8.
            if not dead and not deadlock:
                return False
            self.referee_origin = (b.x, b.y)
            points = [(x, y) for x in (-2.5, 0., 2.5) for y in (-1.25, 0., 1.25)]
            self.referee_target = min(points, key=lambda p: (p[0]-bx)**2+(p[1]-by)**2)
            self.referee_moved = False
            self.referee_settled = None
            self.contact_seen = False
        moved = math.hypot(b.x-self.referee_origin[0], b.y-self.referee_origin[1]) > .15
        self.referee_moved |= moved
        target_distance = math.hypot(bx-self.referee_target[0], by-self.referee_target[1])
        placed = (self.referee_moved and not dead and
                  target_distance < .50 and
                  self.tracks['ball'].speed < .08)
        self.referee_settled = (now if self.referee_settled is None else
                                self.referee_settled) if placed else None
        if self.referee_settled is not None and now-self.referee_settled >= .5:
            self.referee_origin = self.referee_target = None
            self.referee_settled = self.deadlock_since = None
            self.wait_for_pose('重发球已确认，重新确认连续定位')
        else:
            wx, wy = self.world_coordinates(*self.referee_target)
            self.state = 'WAIT_REFEREE'
            self.reason = ('死区/僵持：等待裁判将球放到最近发球点 '
                           f'mocap=({wx:.2f},{wy:.2f}) 米（与 ball 日志同坐标系）；'
                           f'当前偏差={target_distance:.2f}m，要求<0.50m；'
                           f'球速={self.tracks["ball"].speed:.2f}m/s，要求<0.08m/s 持续0.5s；'
                           f'球场坐标=({self.referee_target[0]:.2f},'
                           f'{self.referee_target[1]:.2f})')
        return True

    def _command(self, now):
        if self.finished:
            return self.ZERO
        if self.started_at is not None and now-self.started_at >= self.cfg.max_run_time:
            self.stop('RL 达到运行时间上限')
            return self.ZERO
        if (not self.poses_valid(now) or
                not all(track.valid(now) for track in self.tracks.values())):
            missing = [name for name, track in self.tracks.items() if not track.valid(now)]
            self.wait_for_pose('RL 等待同坐标系连续有效位姿、速度：' + ', '.join(missing))
            self.deadlock_since = None
            if self.attack_planner:
                self.attack_planner.reset()
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
        if self.duel:
            bx, _ = self.field_coordinates(b.x, b.y)
            gx, _ = self.field_coordinates(c.goal_x, c.goal_y)
            if bx <= -gx-c.ball_radius:
                self.stop('球已越过本方门线，请确认结果并重新摆球')
                return self.ZERO
        if self.duel and self.referee_wait(now):
            if self.attack_planner:
                self.attack_planner.reset()
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
        mode, reason = 'RL', 'PPO ' + self.policy + ' 策略控制'
        if self.attack_planner:
            override = self.attack_planner.command(self, now)
            if override is not None:
                velocity = np.asarray(override, dtype=np.float32)
            mode, reason = self.attack_planner.state, self.attack_planner.reason
            velocity = self.attack_planner.limit_contact(self, velocity)
        # Rules and RL share the same body-axis envelope and total speed cap.
        velocity = np.asarray(velocity, dtype=np.float32)
        axis_ratio = max(1., abs(velocity[0])/.45, abs(velocity[1])/.20)
        velocity[:2] /= axis_ratio
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
        self.state, self.reason = mode, reason
        self.target_yaw = None  # PPO outputs angular velocity, not a target heading.
        return tuple(float(v) for v in velocity)
