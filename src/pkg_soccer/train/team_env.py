"""2v2 复赛规则对齐的运动学仿真环境（纯 numpy，不依赖 ROS / gymnasium）。

坐标系：场地中心为原点，+X 是 0 队（左半场）的进攻方向。
- 场地 10 x 5.5 m，围挡 x=±5、y=±2.75；
- 门线 x=±4.585（围挡内侧 0.415 m），门宽 1.5 m；
- 死区：距围挡 0.8 m 的条带，球门口（|y|<=0.75）除外；
- 9 个发球点：x∈{-2.5,0,2.5}, y∈{-1.25,0,1.25}；
- 出发点：四角内侧 0.5 m；
- 僵持 8 s、进球后 15 s 复位（训练里用"立即复位"近似，暂停由 match manager 负责）；
- 观测默认 41 维（2v2）：23 维基础 + 最近对手 + 队友 + 次近对手，
  与部署端 `rl_control.observation` 逐位一致；`obs_dim=29` 时退回"只有最近对手"。

每个 agent 是"一队里的一只狗"。41 维下三只其他狗都进观测：
最近对手进 23-28、队友进 29-34、次近对手进 35-40；后两个槽位按
`EnvConfig.body_dropout` 随机丢帧（整块填 0），与部署端"动捕断流填 0"对齐。
"""

import math

import numpy as np

try:  # 优先复用部署端实现，保证速度估计同源
    from pkg_soccer.measurement import Track
except Exception:  # pragma: no cover - 训练机没装 ROS 包时走副本
    from .track import Track

from .obs import (DEAD_STRIP, FIELD_LENGTH, FIELD_WIDTH, GOAL_LINE_X,
                  GOAL_WIDTH, MAX_X, MAX_Y, MIN_X, MIN_Y, OBS_DIM_TEAM,
                  SERVICE_POINTS, FieldSpec, GOAL_CENTER, make_obs, wrap_angle,
                  body_block)

DT = 0.05
ACTION_SCALE = np.array([0.45, 0.20, 0.70], np.float32)
BALL_RADIUS = 0.12
# Use a conservative footprint: the mocap rigid-body center is smaller than
# the robot's real body, so policies need visible clearance in simulation.
DOG_RADIUS = 0.22
CONTACT_DISTANCE = 0.50
STAGE_DISTANCE = 0.38
DEADLOCK_DISTANCE = 0.56
DEADLOCK_TIME = 8.0
PRESS_RADIUS = 1.5          # 干扰奖励的有效半径（米）：非持球狗贴近持球对手
TEAM_OF = np.array([0, 0, 1, 1])
HOME = {0: np.array([[-4.5, 2.25], [-4.5, -2.25]]),
        1: np.array([[4.5, 2.25], [4.5, -2.25]])}
ATTACK_GOAL = {0: (GOAL_LINE_X, 0.0), 1: (-GOAL_LINE_X, 0.0)}


class EnvConfig:
    """环境参数；默认值与复赛场地/部署端一致。"""

    def __init__(self, speed_limit=0.50, max_steps=600,
                 latency=(1, 3), tau=(0.08, 0.25), tau_w=(0.10, 0.30),
                 ball_drag=(0.3, 1.2), ball_restitution=(0.3, 0.7),
                 mocap_noise=0.008, yaw_noise=0.008,
                 obs_delay=(0, 2), randomize=True, gamma_shaping=0.995,
                 reward_goal=10.0, reward_concede=-10.0,
                 reward_dead_ball=-3.0, reward_deadlock=-2.0,
                 reward_contact=0.5, reward_step=-0.005,
                 reward_collision=-0.22,
                 reward_proximity=-0.10,
                 reward_half=0.0, reward_possess=0.0, reward_press=0.0,
                 shaping_ball=1.5, shaping_stage=0.8, reward_edge=0.0,
                 reward_aim=0.0, reward_own=0.0, reward_front=0.0,
                 reward_back=0.0,
                 scripted_speed=0.30,
                 obs_dim=OBS_DIM_TEAM, body_dropout=0.05, ablate=(),
                 contact_offset=(0.10, 0.0),
                 action_scale=ACTION_SCALE, dash=False, dash_speed=1.5,
                 dash_ms=450, dash_cooldown=1.2, dash_trigger=0.80,
                 dash_range=0.90, dash_cone=0.50, dash_goal_cone=0.0,
                 dash_mask=None):
        self.speed_limit = float(speed_limit)
        # 动作缩放就是"档位"：normal [0.45,0.20,0.70] / fast [0.90,...] / sprint。
        # 必须与部署端 npz 里的 action_scale 完全一致，否则速度对不上。
        self.action_scale = np.asarray(action_scale, np.float32).reshape(3)
        if not np.all(np.isfinite(self.action_scale)) or np.any(self.action_scale <= 0):
            raise ValueError('action_scale 必须是 3 个正数')
        if self.action_scale[0] > 1.5 + 1e-6:
            raise ValueError('action_scale 前进分量不得超过 1.5 m/s（协议 vel_des 上限）')
        # 高速短冲推球：与部署端 rl_control.RLController 同一套触发/执行规则
        self.dash = bool(dash)
        self.dash_speed = float(dash_speed)
        self.dash_steps = max(1, int(round(dash_ms / 1000.0 / DT)))
        self.dash_cooldown = float(dash_cooldown)
        self.dash_trigger = float(dash_trigger)
        self.dash_range = float(dash_range)
        self.dash_cone = float(dash_cone)
        # >0 时只允许"朝球门方向"冲刺（把短冲当射门用，显著减少把球推进死区）
        self.dash_goal_cone = float(dash_goal_cone)
        # 逐只狗的短冲开关（对打评估时两队训练配置不同）；None = 全部跟随 dash
        self.dash_mask = (np.ones(4, bool) if dash_mask is None
                          else np.asarray(dash_mask, bool).reshape(4))
        if not 0 < self.dash_speed <= 1.5:
            raise ValueError('dash_speed 必须在 (0, 1.5] m/s（协议上限）')
        self.max_steps = int(max_steps)
        self.latency = latency
        self.tau = tau
        self.tau_w = tau_w
        self.ball_drag = ball_drag
        self.ball_restitution = ball_restitution
        self.mocap_noise = float(mocap_noise)
        self.yaw_noise = float(yaw_noise)
        self.obs_delay = obs_delay
        self.randomize = bool(randomize)
        self.gamma_shaping = float(gamma_shaping)
        self.reward_goal = float(reward_goal)
        self.reward_concede = float(reward_concede)
        self.reward_dead_ball = float(reward_dead_ball)
        self.reward_deadlock = float(reward_deadlock)
        self.reward_contact = float(reward_contact)
        self.reward_step = float(reward_step)
        self.reward_collision = float(reward_collision)
        self.reward_proximity = float(reward_proximity)
        # 新增三项：半场推进 / 团队控球（两狗等价） / 非持球干扰
        self.reward_half = float(reward_half)
        self.reward_possess = float(reward_possess)
        self.reward_press = float(reward_press)
        self.shaping_ball = float(shaping_ball)
        self.shaping_stage = float(shaping_stage)
        # 远离死区势能项（默认 0=关闭）：Φ = reward_edge × min(1, 球到最近
        # "危险边界"的距离)。危险边界 = 两侧边线 + 门宽之外的端线，因此
        # 门前中路不会被惩罚。势能项（γΦ'−Φ）不改最优策略，只改信用分配。
        self.reward_edge = float(reward_edge)
        if not (math.isfinite(self.reward_edge) and self.reward_edge >= 0.0):
            raise ValueError('reward_edge 必须是非负有限数')
        # 方向感（球速层面，与 reward_edge 的位置层面互补）：
        #   reward_aim > 0：球速朝对方球门的投影 → 奖励"把球推向门前"
        #   reward_own < 0：球速朝自家球门的投影 → 惩罚"把球捅向自家门"（乌龙）
        # 都是 *投影(m/s) × DT*，所以量级是"米"，600 步、0.9 m/s、w=0.08 约 2 分/集。
        self.reward_aim = float(reward_aim)
        self.reward_own = float(reward_own)
        # 触球位置：每次"在球的球门一侧"（球在我与自家门之间）碰球都是一次坏触球
        # —— 这一脚只能把球顶回去。这是"出球方向"的物理源头，比事后奖励球速更直接。
        self.reward_front = float(reward_front)
        if not (math.isfinite(self.reward_front) and self.reward_front <= 0.0):
            raise ValueError('reward_front 必须是非正数（它是惩罚项）')
        # 好触球：在球后方（球在我与对方门之间）碰球，这一脚只能把球往前送
        self.reward_back = float(reward_back)
        if not (math.isfinite(self.reward_back) and self.reward_back >= 0.0):
            raise ValueError('reward_back 必须是非负数')
        if not (math.isfinite(self.reward_aim) and self.reward_aim >= 0.0):
            raise ValueError('reward_aim 必须是非负有限数')
        if not (math.isfinite(self.reward_own) and self.reward_own <= 0.0):
            raise ValueError('reward_own 必须是非正数（它是惩罚项）')
        # 脚本队友/对手的物理速度（m/s）。默认 0.30；想模拟"对手也快"时提到
        # 0.6 左右，否则策略只会打慢对手，学不到对抗下的推进与出球。
        self.scripted_speed = float(scripted_speed)
        if not 0.0 < self.scripted_speed <= 1.5:
            raise ValueError('scripted_speed 必须在 (0, 1.5] m/s')
        if obs_dim not in (29, OBS_DIM_TEAM):
            raise ValueError('obs_dim 只支持 29（1v1）或 41（2v2）')
        self.obs_dim = int(obs_dim)
        if not 0.0 <= float(body_dropout) < 1.0:
            raise ValueError('body_dropout 必须在 [0, 1) 内')
        # 队友/次近对手槽位的丢帧概率（模拟动捕断流 → 6 维填 0）
        self.body_dropout = float(body_dropout)
        self.contact_offset = np.asarray(contact_offset, np.float32).reshape(2)
        if not np.all(np.isfinite(self.contact_offset)) or np.linalg.norm(self.contact_offset) > 0.35:
            raise ValueError('contact_offset 必须是长度不超过 0.35m 的有限二维向量')
        # 消融：保持维度不变，只把某些信息抹掉（研究"这 2~6 维值不值得"）
        self.ablate = tuple(ablate if isinstance(ablate, (list, tuple))
                            else (ablate,) if ablate else ())


# 41 维下三只其他刚体块的起始索引
BODY_SLOT = {'opponent': 23, 'teammate': 29, 'opponent2': 35}
ABLATE_NAMES = ('body_vel', 'body_yaw', 'opponent', 'teammate', 'opponent2',
                'opponent.vel', 'opponent.yaw', 'teammate.vel', 'teammate.yaw',
                'opponent2.vel', 'opponent2.yaw')


def ablation_vectors(names, obs_dim):
    """把消融名字翻译成 (keep_mask, fill)：观测 = obs*keep + fill。

    - `body_vel`：三只刚体的相对速度归零（等价"假设它们静止"）；
    - `body_yaw`：三只刚体的相对朝向固定成同向（sin=0, cos=1，保留合法编码）；
    - `teammate` / `opponent2`：整个 6 维块归零（等价"看不到这只狗"）；
    - `opponent.vel` 这类：只抹单只刚体的某一项。
    """
    keep = np.ones(obs_dim, np.float32)
    fill = np.zeros(obs_dim, np.float32)
    slots = dict(BODY_SLOT)
    if obs_dim != OBS_DIM_TEAM:
        slots.pop('teammate', None)
        slots.pop('opponent2', None)

    def block(body, part):
        base = slots[body]
        if part == 'all':
            return range(base, base + 6)
        if part == 'vel':
            return (base + 2, base + 3)
        if part == 'yaw':
            return (base + 4, base + 5)
        raise ValueError(f'未知消融部位 {part!r}')

    for name in names:
        if name == 'body_vel':
            targets = [(body, 'vel') for body in slots]
        elif name == 'body_yaw':
            targets = [(body, 'yaw') for body in slots]
        elif name in slots:
            targets = [(name, 'all')]
        elif '.' in name and name.split('.')[0] in slots:
            body, part = name.split('.')
            targets = [(body, part)]
        else:
            raise ValueError(f'未知消融名 {name!r}，可选 {ABLATE_NAMES}')
        for body, part in targets:
            for index in block(body, part):
                keep[index] = 0.0
                fill[index] = 0.0
            if part == 'yaw':
                # 固定 yaw_rel=0：sin=0、cos=1，仍然是合法单位向量
                fill[slots[body] + 5] = 1.0
    return keep, fill


class TeamEnv:
    """4 只狗（0/1 为 0 队，2/3 为 1 队）的 2v2 环境。"""

    def __init__(self, cfg=None, seed=None):
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.tracks = [Track(3.0) for _ in range(4)]
        self.ball_track = Track(12.0)
        self.pos = np.zeros((4, 2))
        self.yaw = np.zeros(4)
        self.v_body = np.zeros((4, 2))
        self.w = np.zeros(4)
        self.prev_action = np.zeros((4, 3), np.float32)
        self.contact = np.zeros(4, dtype=bool)
        self.hist = [[] for _ in range(4)]
        self.latency = [1] * 4
        self.tau = np.full(4, 0.15)
        self.tau_w = np.full(4, 0.18)
        self.ball_drag = 0.6
        self.ball_e = 0.5
        self.ball = np.zeros(2)
        self.ball_v = np.zeros(2)
        self.t = 0.0
        self.steps = 0
        self.score = np.zeros(2, dtype=int)
        self.last_touch = None
        self.front_touch = np.zeros(2, dtype=bool)
        self.back_touch = np.zeros(2, dtype=bool)
        self._phys_now = np.zeros(4, dtype=bool)
        self._phys_prev = np.zeros(4, dtype=bool)
        self.stuck_time = 0.0
        self.pending = None
        self.obs_delay = 0
        self.prev_phi = None
        self.colliding = False
        # 每只狗两个"可选刚体槽位"：[队友, 次近对手] 是否可见（对角逐槽位）
        self.visible = [[True, True] for _ in range(4)]
        # 消融掩码：维度不变，只抹掉指定信息（默认全 1 = 不消融）
        self.obs_keep, self.obs_fill = ablation_vectors(self.cfg.ablate,
                                                        self.cfg.obs_dim)
        # 短冲推球状态：每只狗各自"冲刺结束时刻 / 冷却结束时刻 / 机身系方向"
        self.dash_until = np.zeros(4)
        self.dash_ready_at = np.zeros(4)
        self.dash_dir = np.zeros((4, 2))
        self.dash_count = 0
        self._fields = {0: FieldSpec(goal_x=ATTACK_GOAL[0][0]),
                        1: FieldSpec(goal_x=ATTACK_GOAL[1][0], yaw=math.pi)}
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self):
        """开始新的一幕：优先应用上次事件的复位（发球点/出发点）。"""
        self.steps = 0
        self.t = 1.0        # Track 拒绝负时间戳；从 1 s 起算
        self.contact[:] = False
        self.prev_action[:] = 0.0
        self.v_body[:] = 0.0
        self.w[:] = 0.0
        self.last_touch = None
        self.stuck_time = 0.0
        self.colliding = False
        self._phys_prev[:] = False
        self._phys_now[:] = False
        self.visible = [[True, True] for _ in range(4)]
        self.dash_until[:] = 0.0
        self.dash_ready_at[:] = 0.0
        if self.pending is not None:
            self._apply_restart(*self.pending)
            self.pending = None
        else:
            self._kickoff()
        self._sample_domain()
        self._prime_tracks()
        self.prev_phi = self._team_potentials()
        return self.observations()

    def _kickoff(self):
        self.ball = self.rng.uniform(-0.2, 0.2, 2)
        self.ball_v[:] = 0.0
        for i in range(4):
            home = HOME[int(TEAM_OF[i])][i % 2]
            self.pos[i] = self._clip_dog(home + self.rng.normal(0, 0.05, 2))
            goal = ATTACK_GOAL[int(TEAM_OF[i])]
            self.yaw[i] = math.atan2(goal[1] - self.pos[i][1],
                                     goal[0] - self.pos[i][0])

    def _apply_restart(self, kind, team=None):
        if kind == 'goal':
            conceding = 1 - int(team)
            x = -2.5 if conceding == 0 else 2.5
            y = float(self.rng.choice([-1.25, 0.0, 1.25]))
            self.ball = np.array([x, y])
            for i in range(4):
                home = HOME[int(TEAM_OF[i])][i % 2]
                self.pos[i] = self._clip_dog(home + self.rng.normal(0, 0.05, 2))
                self.yaw[i] = float(self.rng.uniform(-math.pi, math.pi))
        elif kind == 'timeout':
            self._kickoff()
        else:
            self.ball = self._random_boundary_restart()
            for i in range(4):
                self.pos[i] = self._clip_dog(self.pos[i])
        self.ball_v[:] = 0.0
        self.v_body[:] = 0.0
        self.w[:] = 0.0

    def _random_boundary_restart(self):
        """裁判将死区/出界球随机放在原出界点附近的场内位置。

        现场发球点是靠近边界的临时位置，不保证落在训练场地的九宫格标记上。
        保留原球最接近的边界方向，并在死区内侧随机化，避免策略只记住固定发球点。
        重发后的球不能仍落在死区，否则下一步会被立即再次判死球。
        """
        x, y = float(self.ball[0]), float(self.ball[1])
        side_gap = min(abs(y - MIN_Y), abs(MAX_Y - y))
        end_gap = min(abs(x - MIN_X), abs(MAX_X - x))
        # DEAD_STRIP 是事件判定的危险条带宽度；再向内留少量余量，确保
        # 重发球是有效比赛状态而不是“重发后立刻再次死球”。
        inward = float(self.rng.uniform(DEAD_STRIP + 0.08,
                                        DEAD_STRIP + 0.85))
        if side_gap <= end_gap:
            side = 1.0 if y >= 0.0 else -1.0
            rx = float(np.clip(x + self.rng.uniform(-0.75, 0.75),
                               MIN_X + DEAD_STRIP + 0.08,
                               MAX_X - DEAD_STRIP - 0.08))
            ry = (MAX_Y - inward) if side > 0 else (MIN_Y + inward)
            return np.array([rx, ry], dtype=float)
        side = 1.0 if x >= 0.0 else -1.0
        ry = float(np.clip(y + self.rng.uniform(-0.75, 0.75),
                           MIN_Y + DEAD_STRIP + 0.08,
                           MAX_Y - DEAD_STRIP - 0.08))
        rx = (MAX_X - inward) if side > 0 else (MIN_X + inward)
        return np.array([rx, ry], dtype=float)

    def _sample_domain(self):
        c = self.cfg
        if not c.randomize:
            self.latency = [2] * 4
            self.tau = np.full(4, 0.15)
            self.tau_w = np.full(4, 0.20)
            self.ball_drag = 0.7
            self.ball_e = 0.5
            self.obs_delay = 1
            return
        low, high = c.latency
        self.latency = [int(self.rng.integers(low, high + 1)) for _ in range(4)]
        self.tau = self.rng.uniform(*c.tau, size=4)
        self.tau_w = self.rng.uniform(*c.tau_w, size=4)
        self.ball_drag = float(self.rng.uniform(*c.ball_drag))
        self.ball_e = float(self.rng.uniform(*c.ball_restitution))
        self.obs_delay = int(self.rng.integers(c.obs_delay[0], c.obs_delay[1] + 1))

    def _prime_tracks(self):
        # Episode clocks restart at 1s; never regress across two episodes.
        self.tracks = [Track(3.0) for _ in range(4)]
        self.ball_track = Track(12.0)
        for k in range(3):
            ts = self.t - (2 - k) * DT
            for i in range(4):
                self.tracks[i].update(ts, ts, self.pos[i][0], self.pos[i][1],
                                      self.yaw[i], 'sim')
            self.ball_track.update(ts, ts, self.ball[0], self.ball[1], 0.0, 'sim')
        self.hist = [np.zeros((4, 3)) for _ in range(4)]

    # ------------------------------------------------------------------- step
    def step(self, actions):
        """actions: (4,3) 归一化动作；返回 (obs(4,obs_dim), rewards(4,), done, info)。"""
        actions = np.clip(np.asarray(actions, dtype=np.float32).reshape(4, 3),
                          -1.0, 1.0)
        self.t += DT
        self.steps += 1
        self.front_touch[:] = False       # 本步是否在球前方触球（按队）
        self.back_touch[:] = False
        self._phys_now[:] = False
        self._update_visibility()
        scale = self.cfg.action_scale
        cmds = actions * scale
        norms = np.linalg.norm(cmds[:, :2], axis=1)
        over = norms > self.cfg.speed_limit
        if np.any(over):
            cmds[over, :2] *= (self.cfg.speed_limit / norms[over])[:, None]
        if self.cfg.dash:
            cmds = self._dash_commands(actions, cmds)
        # 与部署端一致：上一动作存"限速/冲刺之后"的 world 指令除以动作缩放
        self.prev_action[:] = cmds / scale
        for i in range(4):
            self.hist[i] = np.vstack((self.hist[i], cmds[i]))
            if len(self.hist[i]) > 8:
                self.hist[i] = self.hist[i][-8:]
        prev_ball = self.ball.copy()
        newly_contact = []
        for i in range(4):
            delay = min(self.latency[i], len(self.hist[i]))
            cmd = self.hist[i][-delay]
            self.v_body[i] += DT * (cmd[:2] - self.v_body[i]) / self.tau[i]
            self.w[i] += DT * (cmd[2] - self.w[i]) / self.tau_w[i]
            self.yaw[i] = wrap_angle(self.yaw[i] + DT * self.w[i])
            co, si = math.cos(self.yaw[i]), math.sin(self.yaw[i])
            world = np.array([co * self.v_body[i][0] - si * self.v_body[i][1],
                              si * self.v_body[i][0] + co * self.v_body[i][1]])
            self.pos[i] = self._clip_dog(self.pos[i] + DT * world)
        self.colliding = self._dog_collisions()
        for i in range(4):
            co, si = math.cos(self.yaw[i]), math.sin(self.yaw[i])
            ox, oy = self.cfg.contact_offset
            cx = self.pos[i][0] + co * ox - si * oy
            cy = self.pos[i][1] + si * ox + co * oy
            if math.hypot(self.ball[0] - cx, self.ball[1] - cy) < CONTACT_DISTANCE:
                if not self.contact[i]:
                    newly_contact.append(i)
                    self.contact[i] = True
        self._ball_collisions()
        speed = float(np.linalg.norm(self.ball_v))
        if speed > 0:
            self.ball_v *= max(0.0, 1.0 - self.ball_drag * DT / speed)
        self.ball = self.ball + self.ball_v * DT
        self._update_tracks()
        events = self._events(prev_ball)
        if self._deadlocked():
            self.stuck_time += DT
        else:
            self.stuck_time = 0.0
        if self.stuck_time >= DEADLOCK_TIME:
            events = [('deadlock', None)]
        reward = self._reward(events, newly_contact)
        done = bool(events) or self.steps >= self.cfg.max_steps
        if events:
            self.pending = events[0]
            if events[0][0] == 'goal':
                self.score[int(events[0][1])] += 1
        elif self.steps >= self.cfg.max_steps:
            self.pending = ('timeout', None)
        info = {'events': events, 'score': self.score.copy(),
                'last_touch': self.last_touch}
        self._phys_prev[:] = self._phys_now
        return self.observations(), reward, done, info

    def _clip_dog(self, position):
        low = np.array([MIN_X + DOG_RADIUS, MIN_Y + DOG_RADIUS])
        high = np.array([MAX_X - DOG_RADIUS, MAX_Y - DOG_RADIUS])
        return np.clip(position, low, high)

    def _dash_commands(self, actions, cmds):
        """高速短冲推球：与部署端 `RLController._dash` 逐条同规则。

        触发（三条同时满足）：前进指令 `a[0] >= dash_trigger`；球在机身前方
        `±dash_cone` 锥内且距离 `<= dash_range`；冷却已过。
        执行：`dash_steps`（默认 450 ms）内机身系指令改成"触发瞬间的球方向 ×
        dash_speed、wz=0"，对应真机 push 模式（1.5 m/s × 450 ms、步高不变）。
        """
        c = self.cfg
        for i in range(4):
            if not c.dash_mask[i]:
                continue
            if self.t < self.dash_until[i]:
                cmds[i, 0] = self.dash_dir[i, 0] * c.dash_speed
                cmds[i, 1] = self.dash_dir[i, 1] * c.dash_speed
                cmds[i, 2] = 0.0
                continue
            if self.t < self.dash_ready_at[i] or actions[i, 0] < c.dash_trigger:
                continue
            dx = self.ball[0] - self.pos[i][0]
            dy = self.ball[1] - self.pos[i][1]
            distance = math.hypot(dx, dy)
            if not 1e-6 < distance <= c.dash_range:
                continue
            co, si = math.cos(self.yaw[i]), math.sin(self.yaw[i])
            bx, by = co * dx + si * dy, -si * dx + co * dy
            if bx <= 0 or abs(math.atan2(by, bx)) > c.dash_cone:
                continue
            wx = co * bx / distance - si * by / distance
            wy = si * bx / distance + co * by / distance
            # 只允许朝球门方向冲刺：把短冲当射门，而不是"往前乱推"
            if c.dash_goal_cone > 0:
                goal = ATTACK_GOAL[int(TEAM_OF[i])]
                aim = math.atan2(goal[1] - self.ball[1], goal[0] - self.ball[0])
                shot = math.atan2(wy, wx)
                offset = math.atan2(math.sin(shot - aim), math.cos(shot - aim))
                if abs(offset) > c.dash_goal_cone:
                    continue
            # 冲刺位移会把自己推出场地时不下冲（部署端同样会拒绝，见 rl_control）
            travel = c.dash_speed * c.dash_steps * DT
            end_x = self.pos[i][0] + wx * travel
            end_y = self.pos[i][1] + wy * travel
            if not (MIN_X + DOG_RADIUS <= end_x <= MAX_X - DOG_RADIUS and
                    MIN_Y + DOG_RADIUS <= end_y <= MAX_Y - DOG_RADIUS):
                continue
            self.dash_dir[i] = (bx / distance, by / distance)
            self.dash_until[i] = self.t + c.dash_steps * DT
            self.dash_ready_at[i] = self.dash_until[i] + c.dash_cooldown
            self.dash_count += 1
            cmds[i, 0] = self.dash_dir[i, 0] * c.dash_speed
            cmds[i, 1] = self.dash_dir[i, 1] * c.dash_speed
            cmds[i, 2] = 0.0
        return cmds

    def _dog_collisions(self):
        colliding = False
        for i in range(4):
            for j in range(i + 1, 4):
                delta = self.pos[j] - self.pos[i]
                distance = float(np.linalg.norm(delta))
                if not 1e-6 < distance < 2 * DOG_RADIUS:
                    continue
                colliding = True
                normal = delta / distance
                overlap = 2 * DOG_RADIUS - distance
                self.pos[i] = self._clip_dog(self.pos[i] - normal * overlap / 2)
                self.pos[j] = self._clip_dog(self.pos[j] + normal * overlap / 2)
        return colliding

    def _ball_collisions(self):
        for i in range(4):
            co, si = math.cos(self.yaw[i]), math.sin(self.yaw[i])
            ox, oy = self.cfg.contact_offset
            contact_center = self.pos[i] + np.array(
                [co * ox - si * oy, si * ox + co * oy])
            delta = self.ball - contact_center
            distance = float(np.linalg.norm(delta))
            if not 1e-6 < distance < BALL_RADIUS + DOG_RADIUS:
                continue
            normal = delta / distance
            self.ball = contact_center + normal * (BALL_RADIUS + DOG_RADIUS + 1e-4)
            world = np.array([co * self.v_body[i][0] - si * self.v_body[i][1],
                              si * self.v_body[i][0] + co * self.v_body[i][1]])
            relative = self.ball_v - world
            normal_speed = float(relative @ normal)
            if normal_speed < 0:
                self.ball_v -= (1.0 + self.ball_e) * normal_speed * normal
            self.last_touch = int(TEAM_OF[i])
            self._phys_now[i] = True
            # 只在"接触开始"的那一步判定位置：连续贴球推着走不该每帧都算一次
            if not self._phys_prev[i]:
                goal = ATTACK_GOAL[int(TEAM_OF[i])]
                gx, gy = goal[0] - self.ball[0], goal[1] - self.ball[1]
                norm = math.hypot(gx, gy)
                if norm > 1e-6:
                    front = -(normal[0] * gx + normal[1] * gy) / norm
                    side = int(TEAM_OF[i])
                    if front > 0.05:
                        self.front_touch[side] = True
                    elif front < -0.05:
                        self.back_touch[side] = True

    def _update_tracks(self):
        noise = self.cfg.mocap_noise
        for i in range(4):
            self.tracks[i].update(
                self.t, self.t,
                self.pos[i][0] + self.rng.normal(0, noise),
                self.pos[i][1] + self.rng.normal(0, noise),
                self.yaw[i] + self.rng.normal(0, self.cfg.yaw_noise), 'sim')
        self.ball_track.update(
            self.t, self.t,
            self.ball[0] + self.rng.normal(0, noise),
            self.ball[1] + self.rng.normal(0, noise), 0.0, 'sim')

    def _events(self, prev_ball):
        x, y = float(self.ball[0]), float(self.ball[1])
        for team, gx in ((0, GOAL_LINE_X), (1, -GOAL_LINE_X)):
            if (prev_ball[0] - gx) * (x - gx) <= 0 and abs(x - prev_ball[0]) > 1e-9:
                fraction = (gx - prev_ball[0]) / (x - prev_ball[0])
                y_cross = prev_ball[1] + fraction * (y - prev_ball[1])
                if abs(y_cross) <= GOAL_WIDTH / 2:
                    return [('goal', team)]
        if not (MIN_X <= x <= MAX_X and MIN_Y <= y <= MAX_Y):
            return [('out', None)]
        if (abs(y) > MAX_Y - DEAD_STRIP or
                (abs(x) > MAX_X - DEAD_STRIP and abs(y) > GOAL_WIDTH / 2)):
            return [('dead', None)]
        return []

    def _deadlocked(self):
        speeds = np.linalg.norm(self.v_body, axis=1)
        for i in range(4):
            for j in range(i + 1, 4):
                if (speeds[i] < 0.08 and speeds[j] < 0.08 and
                        math.hypot(*(self.pos[i] - self.pos[j])) < DEADLOCK_DISTANCE):
                    return True
        return False

    def _edge_margin(self):
        """球到最近"危险边界"的距离（米，截断到 [0, 1]）。

        危险边界 = 两侧边线（|y|=W/2）+ **门宽之外**的端线（|x|=L/2 且 |y|>门半宽）。
        门前中路（|y|<=门半宽）不算危险，否则势能项会把策略往球门外推。
        """
        half_l = FIELD_LENGTH / 2.0
        half_w = FIELD_WIDTH / 2.0
        x, y = abs(float(self.ball[0])), abs(float(self.ball[1]))
        margin = half_w - y
        if y > GOAL_WIDTH / 2.0:
            margin = min(margin, half_l - x)
        return min(max(margin, 0.0), 1.0)

    def _team_potentials(self):
        edge = self.cfg.reward_edge * self._edge_margin()
        values = []
        for team in (0, 1):
            goal = ATTACK_GOAL[team]
            ball_to_goal = math.hypot(self.ball[0] - goal[0],
                                      self.ball[1] - goal[1])
            norm = max(ball_to_goal, 1e-6)
            unit = np.array([(goal[0] - self.ball[0]) / norm,
                             (goal[1] - self.ball[1]) / norm])
            stage = self.ball - unit * STAGE_DISTANCE
            dogs = np.where(TEAM_OF == team)[0]
            # Only one dog needs to approach the ball; the other's role reward
            # guides it to support, rather than rewarding both for crowding it.
            stage_distance = float(np.min(
                [math.hypot(*(self.pos[i] - stage)) for i in dogs]))
            values.append(-self.cfg.shaping_ball * ball_to_goal -
                          self.cfg.shaping_stage * stage_distance + edge)
        return np.array(values)

    def _possession(self):
        """(控球队, 持球狗, 各狗到球距离)：0.43 m 内的最近者算持球，否则 None。"""
        distances = np.linalg.norm(self.pos - self.ball, axis=1)
        holder = int(np.argmin(distances))
        if distances[holder] > CONTACT_DISTANCE:
            return None, None, distances
        return int(TEAM_OF[holder]), holder, distances

    def _reward(self, events, newly_contact):
        c = self.cfg
        team_reward = np.full(2, c.reward_step)
        phi = self._team_potentials()
        if self.prev_phi is not None:
            # True terminal states have zero potential. Time limits are handled
            # as truncations by the Gym wrapper and keep their bootstrap value.
            team_reward += c.gamma_shaping * (0.0 if events else phi) - self.prev_phi
        for i in newly_contact:
            team_reward[int(TEAM_OF[i])] += c.reward_contact
        # 半场推进项：球每深入对方半场 1 m，每步多拿 reward_half*(1/5)*dt；
        # 停在己方半场则每步扣分。只和球的位置有关，与是谁推的无关。
        half = FIELD_LENGTH / 2
        for side in (0, 1):
            forward = float(self.ball[0]) * (1.0 if side == 0 else -1.0)
            team_reward[side] += c.reward_half * (forward / half) * DT
        # 控球与干扰：只在"进攻方在对方半场组织"时才结算 —— 控球给进攻方、
        # 干扰给防守方；球在己方半场时两项都不发，避免"抱球/贴人"就刷分。
        # 两项都按"队"结算，不区分哪只狗碰到 → 同队两狗拿到的数完全相同。
        team, holder, distances = self._possession()
        if team is not None:
            forward = float(self.ball[0]) * (1.0 if team == 0 else -1.0)
            if forward > 0.0:
                team_reward[team] += c.reward_possess
                defending = 1 - team
                press = 0.0
                for i in np.where(TEAM_OF == defending)[0]:
                    if distances[i] <= CONTACT_DISTANCE:
                        continue        # 这只狗自己在抢球，不算"无球干扰"
                    gap = float(np.linalg.norm(self.pos[i] - self.pos[holder]))
                    press = max(press, max(0.0, 1.0 - gap / PRESS_RADIUS))
                team_reward[defending] += c.reward_press * press
        # 方向感：球速在"球→对方门"方向上的投影。proj>0 = 球正被推向对方球门。
        # 只对**最后触球队**结算，并且只用它自己的符号（0 队朝 +X 为正向）：
        # 这样奖励归因到"做出这一脚的一方"，对手把球推向我们门时我们不会被连带惩罚，
        # 只有"自己把球捅向自家门"才吃 reward_own。
        if (c.reward_aim or c.reward_own) and self.last_touch is not None:
            goal = ATTACK_GOAL[0]
            dx, dy = goal[0] - self.ball[0], goal[1] - self.ball[1]
            distance = math.hypot(dx, dy)
            if distance > 1e-6:
                proj = (float(self.ball_v[0]) * dx +
                        float(self.ball_v[1]) * dy) / distance
                side = int(self.last_touch)
                signed = proj if side == 0 else -proj
                team_reward[side] += (c.reward_aim * max(signed, 0.0) +
                                      c.reward_own * max(-signed, 0.0)) * DT
        if c.reward_front or c.reward_back:
            for side in (0, 1):
                if self.front_touch[side]:
                    team_reward[side] += c.reward_front
                elif self.back_touch[side]:
                    team_reward[side] += c.reward_back
        for kind, team in events:
            if kind == 'goal':
                team_reward[int(team)] += self.cfg.reward_goal
                team_reward[1 - int(team)] += self.cfg.reward_concede
            elif kind in ('out', 'dead'):
                if self.last_touch is None:
                    team_reward += self.cfg.reward_dead_ball / 2.0
                else:
                    team_reward[self.last_touch] += self.cfg.reward_dead_ball
            elif kind == 'deadlock':
                team_reward += self.cfg.reward_deadlock
        if self.colliding:
            team_reward += self.cfg.reward_collision
        # 连续靠得过近时提前给惩罚，避免策略只在发生穿模碰撞的那一帧才学会避让。
        safe_gap = 2.0 * DOG_RADIUS + 0.18
        for i in range(4):
            for j in range(i + 1, 4):
                if TEAM_OF[i] != TEAM_OF[j]:
                    continue
                distance = float(np.linalg.norm(self.pos[i] - self.pos[j]))
                if distance < safe_gap:
                    closeness = (safe_gap - distance) / safe_gap
                    team_reward[int(TEAM_OF[i])] += c.reward_proximity * closeness
        if events:
            self.prev_phi = None
        else:
            self.prev_phi = phi
        return np.array([team_reward[int(TEAM_OF[i])] for i in range(4)],
                        dtype=np.float32)

    # -------------------------------------------------------------- 观测/接口
    def _update_visibility(self):
        """模拟动捕断流：队友/次近对手槽位随机不可见几帧（粘滞，约 0.3~0.5 s）。"""
        p = self.cfg.body_dropout if self.cfg.randomize else 0.0
        if p <= 0.0:
            self.visible = [[True, True] for _ in range(4)]
            return
        for i in range(4):
            for slot in range(2):
                if self.visible[i][slot]:
                    self.visible[i][slot] = bool(self.rng.random() >= p)
                else:
                    # 恢复概率 0.15 → 平均断开约 6~7 个周期（0.3 s）
                    self.visible[i][slot] = bool(self.rng.random() < 0.15)

    def teammate_of(self, index):
        """同队的另一只狗（2v2 里每队两只）。"""
        team = int(TEAM_OF[index])
        mates = [j for j in range(4) if j != index and int(TEAM_OF[j]) == team]
        return mates[0] if mates else None

    def _slot(self, index):
        """(x, y, yaw, (vx, vy))：优先用 Track 的含噪最新值，退化到真值。"""
        latest = self.tracks[index].latest
        pose = (latest[2], latest[3], latest[4]) if latest else (
            float(self.pos[index][0]), float(self.pos[index][1]),
            float(self.yaw[index]))
        velocity = self.tracks[index].velocity or (0.0, 0.0, 0.0)
        return (pose[0], pose[1], pose[2], (velocity[0], velocity[1]))

    def observations(self):
        return np.stack([self._obs_one(i) for i in range(4)])

    def observation_for_actor(self, i, actor):
        """Use the checkpoint's input contract for historical opponents/evaluation.

        Shared-policy checkpoints used teammate first and unrotated lateral
        wall distances for team B. TeamEnv checkpoints use nearest-opponent
        first. Physics stays fixed when comparing them.
        """
        obs = self._obs_one(i)
        if actor.obs_dim != self.cfg.obs_dim:
            raise ValueError('模型与环境观测维度不匹配')
        offset = np.asarray(actor.contact_offset or (0., 0.), np.float32)
        rel = obs[:2] * 3 + self.cfg.contact_offset - offset
        obs[:2], obs[18] = rel / 3, np.linalg.norm(rel) / 3
        if actor.obs_dim == 41 and actor.peer_order == 'teammate_first':
            x, y, yaw, velocity = self._slot(i)
            co, si = math.cos(yaw), math.sin(yaw)
            team = int(TEAM_OF[i])
            enemies = [j for j in range(4) if int(TEAM_OF[j]) != team]
            indices = [self.teammate_of(i)] + enemies
            for k, j in enumerate(indices):
                slot = self._slot(j)
                if (k == 0 and not self.visible[i][0] or
                        k == 2 and not self.visible[i][1]):
                    slot = None
                obs[23+6*k:29+6*k] = body_block(slot, x, y, yaw, velocity, co, si)
            obs[16:18] = ((y-MIN_Y)/6., (MAX_Y-y)/6.)
        return np.clip(obs, -5., 5.)

    def _obs_one(self, i):
        team = int(TEAM_OF[i])
        dog_track = self.tracks[i]
        dog_pose = dog_track.latest or (self.t, self.t, self.pos[i][0],
                                        self.pos[i][1], self.yaw[i], 'sim')
        dog_v = dog_track.velocity or (0.0, 0.0, 0.0)
        ball_pose = self.ball_track.latest or (self.t, self.t, self.ball[0],
                                               self.ball[1], 0.0, 'sim')
        ball_v = self.ball_track.velocity or (0.0, 0.0)
        # 对手按距离升序：最近的进 23-28，第二近的进 35-40（与部署端一致）
        slots = {j: self._slot(j) for j in range(4)}
        enemies = sorted(
            (j for j in range(4) if int(TEAM_OF[j]) != team),
            key=lambda j: math.hypot(slots[j][0] - dog_pose[2],
                                     slots[j][1] - dog_pose[3]))
        age = (0.5 + self.obs_delay) * DT
        mate = self.teammate_of(i)
        teammate = (slots[mate] if (self.cfg.obs_dim == OBS_DIM_TEAM and
                                    mate is not None and self.visible[i][0])
                    else None)
        opponent2 = (slots[enemies[1]]
                     if (self.cfg.obs_dim == OBS_DIM_TEAM and len(enemies) > 1 and
                         self.visible[i][1]) else None)
        obs = make_obs(
            dog_pose[2], dog_pose[3], dog_pose[4], dog_v[:2], dog_v[2],
            ball_pose[2], ball_pose[3], ball_v[:2],
            self.prev_action[i], bool(self.contact[i]), (age, age),
            field=self._fields[team], obs_dim=self.cfg.obs_dim,
            opponent=slots[enemies[0]],
            opponent_age=age,
            teammate=teammate, teammate_age=age,
            opponent2=opponent2, opponent2_age=age,
            contact_offset=self.cfg.contact_offset)
        if self.cfg.ablate:
            obs = obs * self.obs_keep + self.obs_fill
        return obs

    def dog_state(self, i):
        """脚本对手/队友用的真实状态：(x, y, yaw, (vx, vy))。"""
        co, si = math.cos(self.yaw[i]), math.sin(self.yaw[i])
        world = (co * self.v_body[i][0] - si * self.v_body[i][1],
                 si * self.v_body[i][0] + co * self.v_body[i][1])
        return float(self.pos[i][0]), float(self.pos[i][1]), float(self.yaw[i]), world

    def attack_goal_of(self, i):
        return ATTACK_GOAL[int(TEAM_OF[i])]

    def own_goal_of(self, i):
        """本方门中心（与进攻门关于场心对称）。"""
        goal = ATTACK_GOAL[int(TEAM_OF[i])]
        return (-goal[0], -goal[1])

    @staticmethod
    def field_limit():
        """狗心可达的场地内边界 (|x|, |y|)。"""
        return (MAX_X - DOG_RADIUS, MAX_Y - DOG_RADIUS)
