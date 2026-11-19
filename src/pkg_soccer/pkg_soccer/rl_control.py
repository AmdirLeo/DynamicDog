"""PPO 推理端：单狗 23 维 / 1v1 攻防 29 维 / 2v2 攻防 41 维。

checkpoint 约定：进攻 +X、20 Hz、机身系动作。**动作缩放随权重走**：
npz 里的 `action_scale` 就是这一档的速度上限，`Actor` 会读出来用，
所以"常速 0.45 / 高速 0.90 / 冲刺 1.35"只需要换 npz，不用改代码。
没有 `action_scale` 的旧权重按 [0.45, 0.20, 0.70] 处理。
不做在线训练，不做随机采样。

可选"高速短冲推球"（`dash`）：与训练端 `TeamEnv._dash_commands` 逐条同规则 ——
前进指令 ≥0.80、球在机身前方 ±0.50 rad 内且 ≤0.90 m、冷却 1.2 s 已过时，
在 450 ms 内把机身系指令改成"球方向 × dash_speed、wz=0"，等价真机 push 模式。

2v2（41 维）：控制器同时跟踪一名队友和最多三个对手来源（旧的
opponent_pose_topic 以及 opponent1/opponent2_pose_topic）。
观测前半段与 29 维逐位相同（0-28 = 基础 23 维 + 最近合法对手），
后面再接"队友"和"次近对手"两个 6 维块：

    29-34  队友   相对位置/3、相对速度/2、相对朝向 sin/cos
    35-40  次近对手（同上）

槽位无效（未配置、断流、轨迹不连续）时该 6 维填 0；因为合法朝向的
sin/cos 不可能同时为 0，策略据此识别"看不到这个刚体"。
最近对手仍然是硬要求（与 1v1 一致）：没有它就不输出运动指令。
8 s 僵持判定看所有已配置的队友/对手刚体。
"""

import math
from pathlib import Path

import numpy as np

from .control import Controller, Pose, wrap_angle
from .measurement import Track

OPPONENT_KINDS = ('opponent', 'opponent1', 'opponent2')
BODY_KINDS = OPPONENT_KINDS + ('teammate',)
OBS_DIMS = (23, 29, 41)
TEAM_OBS_DIM = 41
BODY_BLOCK = 6
DEFAULT_SCALE = (0.45, 0.20, 0.70)
MAX_FORWARD = 1.5          # 协议 vel_des 前进上限
# 短冲触发参数（训练端 TeamEnv 用同一组默认值）
DASH_TRIGGER = 0.80        # 前进指令 ≥ 这个值才允许冲刺
DASH_RANGE = 0.90          # 球在机身前方这个距离内
DASH_CONE = 0.50           # 球方向与机身前方夹角上限（rad ≈ 28.6°）


def body_block(pose, kind, now, dog, tracks, cos_yaw, sin_yaw, dog_velocity):
    """一只刚体在狗机身系下的 6 维块；不可用则返回全 0。

    与 `train/obs.py::body_block` 逐位一致：位置 /3、速度 /2、朝向 sin/cos。
    """
    if pose is None or pose.frame != dog.frame or kind not in tracks:
        return (0.0,) * BODY_BLOCK
    track = tracks[kind]
    if not track.valid(now):
        return (0.0,) * BODY_BLOCK
    vx, vy, _ = track.velocity
    px, py = pose.x - dog.x, pose.y - dog.y
    wx, wy = vx - dog_velocity[0], vy - dog_velocity[1]
    yaw_rel = wrap_angle(pose.yaw - dog.yaw)
    return ((cos_yaw * px + sin_yaw * py) / 3.,
            (-sin_yaw * px + cos_yaw * py) / 3.,
            (cos_yaw * wx + sin_yaw * wy) / 2.,
            (-sin_yaw * wx + cos_yaw * wy) / 2.,
            math.sin(yaw_rel), math.cos(yaw_rel))


class Actor:
    """Load numeric weights only; compatible with ROS Python without PyTorch."""

    scale = np.array(DEFAULT_SCALE, dtype=np.float32)

    def __init__(self, path=''):
        path = Path(path) if path else Path(__file__).with_name('policy_stage2_actor.npz')
        with np.load(path, allow_pickle=False) as weights:
            first = weights['actor_0_weight']
            if first.ndim != 2 or first.shape[0] != 128:
                raise ValueError('RL 模型第一层形状无效')
            self.obs_dim = int(first.shape[1])
            if self.obs_dim not in OBS_DIMS:
                raise ValueError('RL 模型观测维度必须为 23、29 或 41')
            self.role = str(weights['role'].item()) if 'role' in weights else 'stage2'
            self.layout = (str(weights['obs_layout'].item())
                           if 'obs_layout' in weights else
                           ('2v2' if self.obs_dim == TEAM_OBS_DIM else '1v1'))
            # 动作缩放随权重走：档位（常速/高速/冲刺）写在 npz 里，不改代码。
            if 'action_scale' in weights:
                self.scale = np.asarray(weights['action_scale'],
                                        dtype=np.float32).reshape(3).copy()
            else:
                self.scale = np.array(DEFAULT_SCALE, dtype=np.float32)
            if (not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0) or
                    self.scale[0] > MAX_FORWARD + 1e-6):
                raise ValueError('RL 模型 action_scale 无效（前进分量需 ≤ 1.5 m/s）')
            if self.obs_dim != 23:
                if (self.role not in ('attacker', 'defender') or
                        'action_scale' not in weights):
                    raise ValueError('29/41 维权重缺少角色或动作比例')
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
    """复用进球/场地/转向反馈/姿态门控；2v2 只替换观测里的刚体槽位。"""

    def __init__(self, config, model_path='', speed_limit=0.50, policy='stage2',
                 dash=False, dash_speed=1.5, dash_ms=450, dash_cooldown=1.2,
                 dash_goal_cone=0.0):
        super().__init__(config)
        if policy not in ('stage2', 'attacker', 'defender'):
            raise ValueError('rl_policy 必须为 stage2/attacker/defender')
        self.policy = policy
        self.duel = policy != 'stage2'
        if not model_path and self.duel:
            # 2v2（41 维）优先：把 41 维权重按 policy_team_<角色>_actor.npz 放进本目录即可
            # 自动启用，没有则退回 29 维 1v1 权重。
            team = Path(__file__).with_name(f'policy_team_{policy}_actor.npz')
            model_path = (team if team.exists() else
                          Path(__file__).with_name(f'policy_duel_{policy}_actor.npz'))
        self.actor = Actor(model_path)
        allowed = (23,) if not self.duel else (29, TEAM_OBS_DIM)
        if (self.actor.obs_dim not in allowed or self.actor.role != policy):
            raise ValueError('rl_policy 与模型观测维度或角色不一致')
        # 平移限速：可以高于动作缩放（此时动作缩放才是真上限），也可以更低。
        # `rl_speed_limit:=0` 表示"跟着权重走" = max(0.50, 权重前进缩放)。
        if speed_limit is None or float(speed_limit) <= 0:
            speed_limit = max(0.50, float(self.actor.scale[0]))
        if not math.isfinite(speed_limit) or not 0 < speed_limit <= MAX_FORWARD:
            raise ValueError('rl_speed_limit 必须在 (0, 1.5] m/s')
        self.speed_limit = float(speed_limit)
        # 短冲推球（可选）：规则与训练端团队换人环境一致
        self.dash_enabled = bool(dash)
        self.dash_speed = float(dash_speed)
        if (not math.isfinite(self.dash_speed) or
                not 0 < self.dash_speed <= MAX_FORWARD):
            raise ValueError('rl_dash_speed 必须在 (0, 1.5] m/s')
        self.dash_duration = float(dash_ms) / 1000.0
        self.dash_cooldown = float(dash_cooldown)
        if not 0 < self.dash_duration <= 1.0:
            raise ValueError('rl_dash_ms 必须在 (0, 1000] ms')
        # >0 时只允许朝对方球门方向冲刺（把短冲当射门用，与训练端同规则）
        self.dash_goal_cone = float(dash_goal_cone)
        self.dash_until = -math.inf
        self.dash_ready_at = -math.inf
        self.dash_dir = (1.0, 0.0)
        self.dash_count = 0
        self.tracks = {'dog': Track(3.0), 'ball': Track(12.0)}
        self.bodies = {kind: None for kind in BODY_KINDS}
        for kind in BODY_KINDS:
            self.tracks[kind] = Track(3.0)
        self.body_sequence = 0
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.contact_seen = False
        self.deadlock_since = None
        self.deadlock_cooldown_until = -math.inf

    @property
    def opponent(self):
        """最近的对手位姿（供日志）；没有则 None。"""
        best, best_distance = None, math.inf
        for kind in OPPONENT_KINDS:
            pose = self.bodies.get(kind)
            if pose is None:
                continue
            distance = (math.hypot(pose.x - self.dog.x, pose.y - self.dog.y)
                        if self.dog is not None else 0.0)
            if distance < best_distance:
                best, best_distance = pose, distance
        return best

    @property
    def teammate(self):
        """队友位姿（2v2）；未配置或断流时为 None。"""
        return self.bodies.get('teammate')

    def update(self, kind, x, y, yaw, now, frame, stamp):
        if kind in BODY_KINDS:
            self.update_body(kind, x, y, yaw, now, frame, stamp)
            return
        if kind == 'ball' and self.duel:
            previous = self.ball
            # Referee placement can jump across a goal line. Do not count a
            # discontinuous or stale trajectory as a goal or retain touch state.
            if previous is not None:
                dt = now - previous.received
                discontinuous = (dt <= 0 or dt > self.tracks['ball'].gap or
                                 previous.frame != frame or
                                 math.hypot(x - previous.x, y - previous.y) > 12. * dt + .03)
                if discontinuous:
                    self.contact_seen = False
                    self.wait_for_pose('球轨迹不连续，等待重发球/定位恢复')
        super().update(kind, x, y, yaw, now, frame, stamp)
        self.tracks[kind].update(now, stamp, x, y, yaw, frame)

    def update_body(self, kind, x, y, yaw, now, frame, stamp):
        """队友/对手位姿：只存位姿和速度估计，不参与规则状态机。"""
        if kind not in BODY_KINDS:
            raise ValueError('未知刚体类型：' + str(kind))
        if not all(math.isfinite(v) for v in (x, y, yaw, now, stamp)) or stamp < 0:
            self.bodies[kind] = None
            self.tracks[kind].invalidate('invalid_pose')
            return
        self.body_sequence += 1
        self.bodies[kind] = Pose(x, y, yaw, now, frame, stamp, self.body_sequence)
        self.tracks[kind].update(now, stamp, x, y, yaw, frame)

    def invalidate_body(self, kind, reason):
        if kind in BODY_KINDS:
            self.bodies[kind] = None
            self.tracks[kind].invalidate(reason)

    def nearest_opponents(self, now):
        """按距离升序返回 [(kind, Pose)]；只接受同坐标系、新鲜且有速度估计的对手。"""
        if self.dog is None:
            return []
        ranked = []
        for kind in OPPONENT_KINDS:
            pose = self.bodies.get(kind)
            if pose is None or pose.frame != self.dog.frame:
                continue
            if not self.tracks[kind].valid(now):
                continue
            ranked.append((math.hypot(pose.x - self.dog.x,
                                      pose.y - self.dog.y), kind, pose))
        ranked.sort(key=lambda item: item[0])
        return [(kind, pose) for _distance, kind, pose in ranked]

    def nearest_opponent(self, now):
        """返回 (kind, Pose)；没有可用对手时为 (None, None)。"""
        ranked = self.nearest_opponents(now)
        return ranked[0] if ranked else (None, None)

    def poses_valid(self, now):
        if not super().poses_valid(now):
            return False
        if not self.duel:
            return True
        _, opponent = self.nearest_opponent(now)
        return (opponent is not None and opponent.frame == self.dog.frame and
                0 <= now - opponent.received <= self.cfg.pose_timeout)

    def observation(self, now):
        """Match SoccerEnv2.make_obs; world vectors rotate into the dog frame."""
        d, b, c = self.dog, self.ball, self.cfg
        # 2v2：对手按距离排序；最近对手进 23-28，第二近的进 35-40。
        ranked = self.nearest_opponents(now) if self.duel else []
        kind, o = ranked[0] if ranked else (None, None)
        co, si = math.cos(d.yaw), math.sin(d.yaw)

        def body(x, y):
            return (co * x + si * y, -si * x + co * y)

        rel = body(b.x - d.x, b.y - d.y)
        # attack_goal always denotes the opponent goal, including for defense.
        goal = body(c.goal_x - d.x, c.goal_y - d.y)
        dog_v = self.tracks['dog'].velocity
        ball_v = self.tracks['ball'].velocity
        dv, bv = body(*dog_v[:2]), body(*ball_v[:2])
        fc, fs = math.cos(c.field_yaw), math.sin(c.field_yaw)
        dx, dy = d.x - c.field_center_x, d.y - c.field_center_y
        px, py = fc * dx + fs * dy, -fs * dx + fc * dy
        half_length = ((c.goal_x - c.field_center_x) * fc +
                       (c.goal_y - c.field_center_y) * fs)
        yaw = wrap_angle(d.yaw - c.field_yaw)
        ages = (max(0., now - d.received), max(0., now - b.received))
        obs = [rel[0] / 3., rel[1] / 3., bv[0] / 2., bv[1] / 2.,
               goal[0] / 5., goal[1] / 5., dv[0], dv[1], dog_v[2],
               *self.previous_action, math.sin(yaw), math.cos(yaw),
               (px + half_length) / 9., (half_length + .6 - px) / 9.,
               (py - c.field_min_y) / 6., (c.field_max_y - py) / 6.,
               math.hypot(*rel) / 3., float(self.contact_seen),
               ages[0] / .5, ages[1] / .5, float(max(ages) <= .5)]
        if self.actor.obs_dim != 23:
            if o is None:
                raise ValueError('等待连续有效的对手位姿')
            # DuelEnv uses the outer walls, not the goal lines, for these four
            # distances. Its two age slots share one tracking packet age.
            obs[14:18] = ((px - c.field_min_x) / 9., (c.field_max_x - px) / 9.,
                          (py - c.field_min_y) / 6., (c.field_max_y - py) / 6.)
            age = max(*ages, max(0., now - o.received))
            obs[20:23] = (age / .5, age / .5, float(age <= .5))
            dog_velocity = (float(dog_v[0]), float(dog_v[1]))
            obs.extend(body_block(o, kind, now, d, self.tracks, co, si,
                                  dog_velocity))
            if self.actor.obs_dim == TEAM_OBS_DIM:
                # 队友与次近对手：不可用时整块填 0，不阻塞控制。
                obs.extend(body_block(self.bodies.get('teammate'), 'teammate',
                                      now, d, self.tracks, co, si,
                                      dog_velocity))
                second_kind, second = (ranked[1] if len(ranked) > 1
                                       else (None, None))
                obs.extend(body_block(second, second_kind, now, d, self.tracks,
                                      co, si, dog_velocity))
        return np.clip(np.asarray(obs, dtype=np.float32), -5., 5.)

    def command(self, now):
        try:
            velocity = super().command(now)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            self.stop('RL 推理失败：' + str(exc))
            velocity = self.ZERO
        # The next observation reflects the command after limits/protection.
        self.previous_action = np.asarray(velocity, dtype=np.float32) / self.actor.scale
        return velocity

    def field_coordinates(self, x, y):
        c = self.cfg
        dx, dy = x - c.field_center_x, y - c.field_center_y
        co, si = math.cos(c.field_yaw), math.sin(c.field_yaw)
        return co * dx + si * dy, -si * dx + co * dy

    def blocked_by_body(self, now):
        """狗几乎静止且与某个队友/对手贴近（8 s 僵持判据的一部分）。"""
        d = self.dog
        if d is None or not self.tracks['dog'].valid(now) or self.tracks['dog'].speed >= .08:
            return False
        for kind in BODY_KINDS:
            pose = self.bodies.get(kind)
            if pose is None or pose.frame != d.frame:
                continue
            if not 0 <= now - pose.received <= self.cfg.pose_timeout:
                continue
            track = self.tracks[kind]
            if not track.valid(now) or track.speed >= .08:
                continue
            if math.hypot(pose.x - d.x, pose.y - d.y) < .56:
                return True
        return False

    def referee_needed(self, now):
        """球进死区、或双方贴近僵持超过 8 s 时，需要裁判就近发球。

        与规则 4.2/4.4 对齐：判到就暂停等发球，球被放到任意发球点后由
        `_resume_ready` 自动恢复，不再靠"球必须移动到某个指定点"来判定。
        2v2：任一队友/对手在 0.56 m 内且双方都几乎静止，就算僵持。
        """
        c, b = self.cfg, self.ball
        bx, by = self.field_coordinates(b.x, b.y)
        dead = (by < c.field_min_y + .8 or by > c.field_max_y - .8 or
                ((bx < c.field_min_x + .8 or bx > c.field_max_x - .8) and
                 abs(by) > c.goal_width / 2))
        if dead:
            self.deadlock_since = None
            return '球进入死区，等待裁判就近发球'
        if now < self.deadlock_cooldown_until:
            return None
        stuck = self.blocked_by_body(now)
        self.deadlock_since = (now if self.deadlock_since is None else
                               self.deadlock_since) if stuck else None
        if self.deadlock_since is not None and now - self.deadlock_since >= 8.:
            return '双方僵持超过 8 秒，等待裁判就近发球'
        return None

    def rearm(self, now, reason):
        """发球确认后重启一轮：同时清掉 RL 的上一动作、接触与短冲状态。"""
        super().rearm(now, reason)
        self.deadlock_since = None
        # 刚恢复时双方可能还贴在一起，先给 5 s 再重新计僵持，避免立刻停第二次。
        self.deadlock_cooldown_until = now + 5.0
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.contact_seen = False
        self.dash_until = -math.inf
        self.dash_ready_at = -math.inf

    def _dash(self, action, velocity, now):
        """高速短冲推球；与 `train/team_env.py::_dash_commands` 逐条同规则。

        冲刺期间机身系指令 = 触发瞬间的球方向 × dash_speed、wz=0，
        等价真机 push 模式（1.5 m/s × 450 ms、步高不变、不检测相位）。
        """
        if not self.dash_enabled:
            return velocity
        d, b = self.dog, self.ball
        if now < self.dash_until:
            return (self.dash_dir[0] * self.dash_speed,
                    self.dash_dir[1] * self.dash_speed, 0.0)
        if now < self.dash_ready_at or float(action[0]) < DASH_TRIGGER:
            return velocity
        if d is None or b is None:
            return velocity
        c = self.cfg
        dx, dy = b.x - d.x, b.y - d.y
        distance = math.hypot(dx, dy)
        if not 1e-6 < distance <= DASH_RANGE:
            return velocity
        co, si = math.cos(d.yaw), math.sin(d.yaw)
        bx, by = co * dx + si * dy, -si * dx + co * dy
        if bx <= 0 or abs(math.atan2(by, bx)) > DASH_CONE:
            return velocity
        wx = co * bx / distance - si * by / distance
        wy = si * bx / distance + co * by / distance
        if self.dash_goal_cone > 0:
            aim = math.atan2(c.goal_y - b.y, c.goal_x - b.x)
            offset = math.atan2(math.sin(math.atan2(wy, wx) - aim),
                                math.cos(math.atan2(wy, wx) - aim))
            if abs(offset) > self.dash_goal_cone:
                return velocity
        # 冲刺位移会把自己推出场地时不下冲（训练端同样的拒绝规则）
        travel = self.dash_speed * self.dash_duration
        end_x = d.x + wx * travel
        end_y = d.y + wy * travel
        if not self.inside_field(end_x, end_y):
            return velocity
        self.dash_dir = (bx / distance, by / distance)
        self.dash_until = now + self.dash_duration
        self.dash_ready_at = now + self.dash_duration + self.dash_cooldown
        self.dash_count += 1
        return (self.dash_dir[0] * self.dash_speed,
                self.dash_dir[1] * self.dash_speed, 0.0)

    def _command(self, now):
        if self.finished:
            return self.ZERO
        if self.paused:
            if self._resume_ready(now):
                self.rearm(now, '球已回到发球点，重新开始比赛')
            else:
                self.state = 'WAIT_PLAY'
            return self.ZERO
        if self.started_at is not None and now - self.started_at >= self.cfg.max_run_time:
            self.stop('RL 达到运行时间上限')
            return self.ZERO
        if (not self.poses_valid(now) or
                not all(self.tracks[k].valid(now) for k in ('dog', 'ball'))):
            missing = [k for k in ('dog', 'ball')
                       if not self.tracks[k].valid(now)]
            if self.duel and self.nearest_opponent(now)[1] is None:
                missing.append('opponent')
            self.wait_for_pose('RL 等待同坐标系连续有效位姿、速度：' + ', '.join(missing))
            self.deadlock_since = None
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
            if bx <= -gx - c.ball_radius:
                self.pause_play(now, '球已越过本方球门线，等待裁判发球')
                return self.ZERO
            reason = self.referee_needed(now)
            if reason:
                self.pause_play(now, reason)
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
        self.distance = math.hypot(b.x - d.x, b.y - d.y)
        self.contact_seen |= self.distance < .43
        velocity = action * self.actor.scale
        speed = float(np.linalg.norm(velocity[:2]))
        if speed > self.speed_limit:
            velocity[:2] *= self.speed_limit / speed
        velocity[2] = np.clip(velocity[2], -c.turn_speed, c.turn_speed)
        velocity = self._dash(action, velocity, now)
        co, si = math.cos(d.yaw), math.sin(d.yaw)
        wx, wy = co * velocity[0] - si * velocity[1], si * velocity[0] + co * velocity[1]
        # Include observed momentum when checking the next half second.
        for vx, vy in ((wx, wy), self.tracks['dog'].velocity[:2]):
            if not self.inside_field(d.x + float(vx) * .5, d.y + float(vy) * .5):
                self.stop('RL 预计运动将越过场地边界')
                return self.ZERO
        self.state, self.reason = 'RL', ('PPO ' + self.policy + ' 策略控制')
        self.target_yaw = None  # PPO outputs angular velocity, not a target heading.
        return tuple(float(v) for v in velocity)
