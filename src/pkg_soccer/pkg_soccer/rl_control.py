"""PPO deployment for single-dog, 1v1 and shared 2v2 policies.

The checkpoint uses attack +X, 20 Hz and body-frame actions scaled by
[0.45, 0.20, 0.70]. No online training or stochastic action sampling.
"""

import math
import re
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
            # 不用海象运算符：狗上若是 Python 3.6 会直接 SyntaxError。
            indices = []
            for key in weights.files:
                match = re.fullmatch(r'actor_(\d+)_weight', key)
                if match:
                    indices.append(int(match.group(1)))
            indices.sort()
            if not indices or indices[0] != 0 or any(i % 2 for i in indices):
                raise ValueError('RL 模型 actor 层索引无效')
            if indices != list(range(0, indices[-1] + 1, 2)):
                raise ValueError('RL 模型 actor 层不连续')
            first = weights['actor_0_weight']
            if first.ndim != 2 or first.shape[0] < 1:
                raise ValueError('RL 模型第一层形状无效')
            self.obs_dim = int(first.shape[1])
            if self.obs_dim not in (23, 29, 41):
                raise ValueError('RL 模型观测维度必须为 23、29 或 41')
            self.role = str(weights['role'].item()) if 'role' in weights else 'stage2'
            if self.obs_dim in (29, 41):
                allowed = ('attacker', 'defender') if self.obs_dim == 29 else ('shared_2v2', 'shared_2v2_large')
                if (self.role not in allowed or
                        'action_scale' not in weights or
                        not np.allclose(weights['action_scale'], self.scale)):
                    raise ValueError('29 维权重缺少角色或动作比例不匹配')
            self.layers = []
            previous = self.obs_dim
            for index in indices:
                w = weights[f'actor_{index}_weight'].copy()
                b = weights[f'actor_{index}_bias'].copy()
                if (w.ndim != 2 or w.shape[1] != previous or b.shape != (w.shape[0],) or
                        not np.isfinite(w).all() or not np.isfinite(b).all()):
                    raise ValueError('RL 模型权重形状或数值无效')
                self.layers.append((w, b))
                previous = w.shape[0]
            if previous != 3:
                raise ValueError('RL 模型输出维度必须为 3')

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

    def __init__(self, config, model_path='', speed_limit=0.30, policy='stage2',
                 speed_gain=1.0, continue_deadzone=True,
                 deadzone_speed_limit=0.35, collision_radius=0.0,
                 teammate_margin=0.10):
        super().__init__(config)
        # 队友避让：collision_radius 是单只狗的等效半径（米），0=关闭。
        # 两只狗中心距 < 2*radius + margin 时，去掉速度中"靠近队友"的分量。
        self.collision_radius = float(collision_radius)
        self.teammate_margin = float(teammate_margin)
        if not (math.isfinite(self.collision_radius) and
                0.0 <= self.collision_radius <= 0.6):
            raise ValueError('collision_radius 必须在 [0, 0.6] m（0=关闭避让）')
        if not (math.isfinite(self.teammate_margin) and
                0.0 <= self.teammate_margin <= 1.0):
            raise ValueError('teammate_margin 必须在 [0, 1.0] m')
        self.collision_active = False
        if policy not in ('stage2', 'attacker', 'defender', 'shared_2v2', 'shared_2v2_large'):
            raise ValueError('rl_policy 必须为 stage2/attacker/defender/shared_2v2/shared_2v2_large')
        self.policy = policy
        self.duel = policy != 'stage2'
        self.team2v2 = policy in ('shared_2v2', 'shared_2v2_large')
        if not math.isfinite(speed_limit) or not 0 < speed_limit <= 1.5:
            raise ValueError('rl_speed_limit 必须在 (0, 1.5] m/s（协议 vel_des 上限 1.5）')
        # 策略的动作缩放来自权重（shared_2v2 是 0.45/0.20/0.70）。想要更高的
        # 最高速度就把 m/s 两维按增益放大；转向（rad/s）不放大，否则会失控。
        self.speed_gain = float(speed_gain)
        # 0 = 自动：把"满舵前进"标定成正好等于 speed_limit（推荐，换权重也不用改）。
        if not (math.isfinite(self.speed_gain) and 0.0 <= self.speed_gain <= 4.0):
            raise ValueError('rl_speed_gain 必须在 [0, 4.0]（0=按 rl_speed_limit 自动标定）')
        if not model_path and self.team2v2:
            filename = ('policy_2v2_shared_large_actor.npz' if policy == 'shared_2v2_large'
                        else 'policy_2v2_shared_actor.npz')
            model_path = Path(__file__).with_name(filename)
        elif not model_path and self.duel:
            model_path = Path(__file__).with_name(f'policy_duel_{policy}_actor.npz')
        self.actor = Actor(model_path)
        expected_dim = 41 if self.team2v2 else (29 if self.duel else 23)
        if (self.actor.obs_dim != expected_dim or
                self.actor.role != policy):
            raise ValueError('rl_policy 与模型观测维度或角色不一致')
        self.speed_limit = speed_limit
        self.continue_deadzone = bool(continue_deadzone)
        self.deadzone_speed_limit = float(deadzone_speed_limit)
        if (not math.isfinite(self.deadzone_speed_limit) or
                not 0.0 < self.deadzone_speed_limit <= self.speed_limit):
            raise ValueError('deadzone_speed_limit 必须在 (0, rl_speed_limit]')
        self.deadzone_active = False
        scale = np.asarray(self.actor.scale, np.float32)
        if self.speed_gain <= 0.0:
            self.speed_gain = (float(speed_limit)/scale[0]
                               if scale[0] > 0 else 1.0)
            self.speed_gain = min(max(self.speed_gain, 0.2), 4.0)
        self.effective_scale = np.array(
            [scale[0]*self.speed_gain, scale[1]*self.speed_gain, scale[2]], np.float32)
        # 短时动捕断流外推：断流超过 gap 后用最后速度/加速度外推 prediction_timeout。
        pg = float(config.prediction_timeout)
        self.tracks = {'dog': Track(3.0, prediction_gap=pg),
                       'ball': Track(12.0, prediction_gap=pg)}
        if self.duel:
            self.tracks['opponent'] = Track(3.0, prediction_gap=pg)
        if self.team2v2:
            self.tracks['teammate'] = Track(3.0, prediction_gap=pg)
            self.tracks['opponent2'] = Track(3.0, prediction_gap=pg)
        for name in self.tracks:
            self._observed.setdefault(name, None)
            self._velocity.setdefault(name, (0.0, 0.0, 0.0))
            self._acceleration.setdefault(name, (0.0, 0.0, 0.0))
        self.opponent = None
        self.teammate = None
        self.opponent2 = None
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
        if kind in ('opponent', 'opponent2', 'teammate'):
            self.update_peer(kind, x, y, yaw, now, frame, stamp)
            return
        if kind == 'ball' and self.duel:
            previous = self._observed.get('ball')
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

    def update_peer(self, kind, x, y, yaw, now, frame, stamp):
        if kind not in self.tracks:
            raise ValueError('单狗模式不接收对手位姿')
        if not all(math.isfinite(v) for v in (x, y, yaw, now, stamp)) or stamp < 0:
            setattr(self, kind, None)
            self._observed[kind] = None
            self.tracks[kind].invalidate('invalid_pose')
            return
        previous = self._observed.get(kind)
        sequence = previous.sequence + 1 if previous is not None else 1
        pose = Pose(x, y, yaw, now, frame, stamp, sequence)
        if previous is not None and previous.frame == frame and now > previous.received:
            dt = now - previous.received
            old = self._velocity.get(kind, (0.0, 0.0, 0.0))
            velocity = ((x-previous.x)/dt, (y-previous.y)/dt,
                        wrap_angle(yaw-previous.yaw)/dt)
            self._acceleration[kind] = tuple((velocity[j]-old[j])/dt
                                             for j in range(3))
            self._velocity[kind] = velocity
        self._observed[kind] = pose
        self._predicted.discard(kind)
        setattr(self, kind, pose)
        self.tracks[kind].update(now, stamp, x, y, yaw, frame)
        if kind == 'opponent':
            self.opponent = getattr(self, kind)

    def poses_valid(self, now):
        if not super().poses_valid(now):
            return False
        if not self.duel:
            return True
        peer_names = ('teammate', 'opponent', 'opponent2') if self.team2v2 else ('opponent',)
        # 断流时 super().poses_valid 已用 _predict_missing 外推 peer 位姿，
        # 外推帧的 received 会被更新，故容差加上 prediction_timeout。
        peer_max_age = min(0.20, self.cfg.pose_timeout) + self.cfg.prediction_timeout
        return all((getattr(self, name) is not None and
                    getattr(self, name).frame == self.dog.frame and
                    0 <= now-getattr(self, name).received <= peer_max_age)
                   for name in peer_names)

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
        elif self.actor.obs_dim == 41:
            peer_names = ('teammate', 'opponent', 'opponent2')
            peer_features = []
            # TeamEnv2v2 uses the outer field half-length (5 m) for its
            # longitudinal boundary features. Keep this branch identical to
            # training instead of the 1v1 safety margin convention.
            outer_half = max(abs(c.field_min_x), abs(c.field_max_x))
            obs[14:18] = ((px + outer_half)/9., (outer_half-px)/9.,
                          (py-c.field_min_y)/6., (c.field_max_y-py)/6.)
            age = max(*ages, *(max(0., now-getattr(self, name).received)
                               for name in peer_names))
            obs[20:23] = (age/.5, age/.5, float(age <= .5))
            for name in peer_names:
                peer = getattr(self, name)
                rel_peer = body(peer.x-d.x, peer.y-d.y)
                peer_v = self.tracks[name].velocity
                rel_v = body(peer_v[0]-dog_v[0], peer_v[1]-dog_v[1])
                peer_yaw = wrap_angle(peer.yaw-d.yaw)
                peer_features.extend((rel_peer[0]/3., rel_peer[1]/3.,
                                      rel_v[0]/2., rel_v[1]/2.,
                                      math.sin(peer_yaw), math.cos(peer_yaw)))
            obs.extend(peer_features)
        return np.clip(np.asarray(obs, dtype=np.float32), -5., 5.)

    def command(self, now):
        try:
            velocity = super().command(now)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            self.stop('RL 推理失败：' + str(exc))
            velocity = self.ZERO
        velocity = self.avoid_teammate(velocity, now)
        # The next observation reflects the command after limits/protection.
        self.previous_action = np.asarray(velocity, dtype=np.float32)/self.effective_scale
        return velocity

    def avoid_teammate(self, velocity, now):
        """去掉机身系速度中"朝队友靠近"的分量，防止两只狗贴在一起。

        只在 2v2 已配置队友时生效；转向保留（可以边绕边走）。
        这是安全层，不改变策略输出的其它分量，也不参与训练观测以外的状态。
        """
        self.collision_active = False
        if (self.collision_radius <= 0 or not self.team2v2 or
                self.dog is None or self.teammate is None):
            return velocity
        if ('teammate' not in self.tracks or
                not self.tracks['teammate'].valid(now)):
            return velocity
        dx, dy = self.teammate.x - self.dog.x, self.teammate.y - self.dog.y
        distance = math.hypot(dx, dy)
        limit = 2.0*self.collision_radius + self.teammate_margin
        if distance >= limit or distance < 1e-6:
            return velocity
        # 世界系相对位置 → 机身系单位向量（机身 +x 朝狗头）。
        yaw = self.dog.yaw
        ux = (math.cos(yaw)*dx + math.sin(yaw)*dy)/distance
        uy = (-math.sin(yaw)*dx + math.cos(yaw)*dy)/distance
        vx, vy = float(velocity[0]), float(velocity[1])
        approach = vx*ux + vy*uy
        if approach <= 0.0:
            # 已经在远离/只是擦边经过，不干预策略输出。
            return velocity
        if distance <= 2.0*self.collision_radius:
            # 已经贴上且还在靠近：只保留转向，平移清零，避免继续挤压。
            vx = vy = 0.0
        else:
            # 还没贴上：去掉径向接近分量，保留绕行（切向）分量。
            vx -= approach*ux
            vy -= approach*uy
        self.collision_active = True
        self.reason += f'；队友避让(d={distance:.2f}m)'
        return np.array([vx, vy, velocity[2]], dtype=np.float32)

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
            # 裁判发球点是靠近出界位置的随机点，不假设九宫格标记一定存在。
            self.referee_target = None
            self.referee_moved = False
            self.referee_settled = None
            self.contact_seen = False
        moved = math.hypot(b.x-self.referee_origin[0], b.y-self.referee_origin[1]) > .15
        self.referee_moved |= moved
        placed = (self.referee_moved and self.tracks['ball'].speed < .08)
        self.referee_settled = (now if self.referee_settled is None else
                                self.referee_settled) if placed else None
        if self.referee_settled is not None and now-self.referee_settled >= .5:
            self.referee_origin = self.referee_target = None
            self.referee_settled = self.deadlock_since = None
            self.wait_for_pose('重发球已确认，重新确认连续定位')
        else:
            self.state = 'WAIT_REFEREE'
            self.reason = ('死区/僵持：等待裁判将球放到最近发球点 '
                           '附近的随机位置（与 ball 日志同坐标系）；'
                           f'球已移动={self.referee_moved}；'
                           f'球速={self.tracks["ball"].speed:.2f}m/s，要求<0.08m/s 持续0.5s；'
                           f'球场坐标=({bx:.2f},{by:.2f})')
        return True

    def _command(self, now):
        if self.finished:
            return self.ZERO
        # 进球/球出界/球停门线后都不结束比赛：暂停等裁判把球放回发球点，
        # 球静止在任一发球点且狗也静止后 rearm() 自动继续（规则 4.2/4.4）。
        if self.paused:
            if self._resume_ready(now):
                self.rearm(now, '球已回到发球点，重新开始比赛')
            else:
                self.state = 'WAIT_PLAY'
            if self.attack_planner:
                self.attack_planner.reset()
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
            self.goals += 1
            self.pause_play(now, f'进球（本场第 {self.goals} 个），等待裁判发球')
            return self.ZERO
        d, b, c = self.dog, self.ball, self.cfg
        if not self.inside_field(d.x, d.y):
            self.stop('狗已超出设定场地')
            return self.ZERO
        if not self.inside_field(b.x, b.y):
            self.pause_play(now, '球已出界，等待裁判就近发球')
            return self.ZERO
        if self.goal_coordinates(b.x, b.y)[0] >= c.ball_radius:
            self.pause_play(now, '球在球门线后，等待裁判发球')
            return self.ZERO
        if self.duel:
            bx, _ = self.field_coordinates(b.x, b.y)
            gx, _ = self.field_coordinates(c.goal_x, c.goal_y)
            if bx <= -gx-c.ball_radius:
                self.pause_play(now, '球已越过本方门线，等待裁判发球')
                return self.ZERO
        deadzone_wait = self.duel and self.referee_wait(now)
        self.deadzone_active = bool(deadzone_wait)
        if deadzone_wait and not self.continue_deadzone:
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
        velocity = action*self.effective_scale
        mode, reason = 'RL', 'PPO ' + self.policy + ' 策略控制'
        if self.attack_planner:
            override = self.attack_planner.command(self, now)
            if override is not None:
                velocity = np.asarray(override, dtype=np.float32)
            mode, reason = self.attack_planner.state, self.attack_planner.reason
            velocity = self.attack_planner.limit_contact(self, velocity)
        # Rules and RL share the same body-axis envelope and total speed cap.
        velocity = np.asarray(velocity, dtype=np.float32)
        axis_ratio = max(1., abs(velocity[0])/self.effective_scale[0],
                         abs(velocity[1])/self.effective_scale[1])
        velocity[:2] /= axis_ratio
        speed = float(np.linalg.norm(velocity[:2]))
        if speed > self.speed_limit:
            velocity[:2] *= self.speed_limit/speed
        if self.deadzone_active and speed > self.deadzone_speed_limit:
            velocity[:2] *= self.deadzone_speed_limit/max(speed, 1e-9)
        velocity[2] = np.clip(velocity[2], -c.turn_speed, c.turn_speed)
        co, si = math.cos(d.yaw), math.sin(d.yaw)
        wx, wy = co*velocity[0]-si*velocity[1], si*velocity[0]+co*velocity[1]
        # Include observed momentum when checking the next half second.
        for vx, vy in ((wx, wy), self.tracks['dog'].velocity[:2]):
            if not self.inside_field(d.x+float(vx)*.5, d.y+float(vy)*.5):
                self.stop('RL 预计运动将越过场地边界')
                return self.ZERO
        if self.deadzone_active:
            self.state = 'DEADZONE_MOVE'
            self.reason = ('球在死区，继续低速运动；等待裁判将球放回发球点，'
                           f'死区速度上限={self.deadzone_speed_limit:.2f}m/s')
        else:
            self.state, self.reason = mode, reason
        self.target_yaw = None  # PPO outputs angular velocity, not a target heading.
        return tuple(float(v) for v in velocity)
