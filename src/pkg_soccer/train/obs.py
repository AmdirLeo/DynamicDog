"""与部署端 `pkg_soccer/rl_control.py` 的观测构造逐位一致。

训练与部署必须共用同一套公式；本文件是训练侧的权威实现，
`verify.py --contract` 会用真实的 RLController 逐位对比。

观测布局（23 = 单狗；29 = 1v1；41 = 2v2）：

| idx | 内容 | 归一化 |
|---|---|---|
| 0-1 | 球相对狗（机身系） | /3 |
| 2-3 | 球速（机身系） | /2 |
| 4-5 | 进攻球门相对狗（机身系） | /5 |
| 6-7 | 狗速（机身系） | 无 |
| 8 | 狗偏航角速度 | 无 |
| 9-11 | 上一步动作（限速后 velocity / scale） | 无 |
| 12-13 | 狗相对场地朝向 sin/cos | 无 |
| 14-17 | 到四墙距离（23 维用门线，攻防用外场） | /9,/9,/6,/6 |
| 18 | 球距 | /3 |
| 19 | contact_seen | 0/1 |
| 20-22 | 狗/球/对手年龄（取最大）+ 新鲜标志 | /0.5 |
| 23-28 | 最近对手：相对位置/相对速度/相对朝向 | /3,/2,sin/cos |
| 29-34 | **队友**（仅 41 维）：同上 6 维 | /3,/2,sin/cos |
| 35-40 | **次近对手**（仅 41 维）：同上 6 维 | /3,/2,sin/cos |

槽位无效（动捕未配置/断流/轨迹不连续）时该 6 维填 0：
合法朝向必定满足 sin²+cos²=1，所以 (0,0) 可以被策略识别为"看不到"。
训练时按 `EnvConfig.body_dropout` 随机丢掉这两个槽位，逼策略兼容部署端的 0 填充。
"""

import math

import numpy as np

ACTION_SCALE = np.array([0.45, 0.20, 0.70], np.float32)

# 复赛场地（与 pkg_soccer 的攻防默认参数一致）
FIELD_LENGTH = 10.0
FIELD_WIDTH = 5.5
GOAL_LINE_X = FIELD_LENGTH / 2 - 0.415      # 4.585
GOAL_WIDTH = 1.5
BALL_RADIUS = 0.12
DEAD_STRIP = 0.8
MIN_X = -FIELD_LENGTH / 2
MAX_X = FIELD_LENGTH / 2
MIN_Y = -FIELD_WIDTH / 2
MAX_Y = FIELD_WIDTH / 2
SERVICE_POINTS = [(x, y) for x in (-2.5, 0.0, 2.5) for y in (-1.25, 0.0, 1.25)]
GOAL_CENTER = {0: (GOAL_LINE_X, 0.0), 1: (-GOAL_LINE_X, 0.0)}

OBS_DIM_STAGE2 = 23
OBS_DIM_DUEL = 29
OBS_DIM_TEAM = 41
OBS_DIMS = (OBS_DIM_STAGE2, OBS_DIM_DUEL, OBS_DIM_TEAM)
# 41 维在 29 维基础上的两个新槽位
TEAMMATE_SLICE = (29, 35)
OPPONENT2_SLICE = (35, 41)
BODY_BLOCK = 6


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class FieldSpec:
    """部署端 Config 里与观测有关的那部分。"""

    __slots__ = ('goal_x', 'goal_y', 'min_x', 'max_x', 'min_y', 'max_y',
                 'center_x', 'center_y', 'yaw', 'goal_width')

    def __init__(self, goal_x=GOAL_LINE_X, goal_y=0.0,
                 min_x=MIN_X, max_x=MAX_X, min_y=MIN_Y, max_y=MAX_Y,
                 center_x=0.0, center_y=0.0, yaw=0.0, goal_width=GOAL_WIDTH):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.center_x = center_x
        self.center_y = center_y
        self.yaw = yaw
        self.goal_width = goal_width


DEFAULT_FIELD = FieldSpec()


def body_block(slot, dog_x, dog_y, dog_yaw, dog_vel, cos_yaw, sin_yaw,
               position_scale=3.0, velocity_scale=2.0):
    """一只刚体在狗机身系下的 6 维块；slot=None 表示该槽位不可见。

    slot = (x, y, yaw, (vx, vy))，速度为世界系。不可见时返回全 0，
    因为合法朝向的 sin/cos 不可能同时为 0，策略能区分"看不到"。
    """
    if slot is None:
        return (0.0,) * BODY_BLOCK
    x, y, yaw, velocity = slot
    px, py = x - dog_x, y - dog_y
    vx, vy = velocity[0] - dog_vel[0], velocity[1] - dog_vel[1]
    yaw_rel = wrap_angle(yaw - dog_yaw)
    return ((cos_yaw * px + sin_yaw * py) / position_scale,
            (-sin_yaw * px + cos_yaw * py) / position_scale,
            (cos_yaw * vx + sin_yaw * vy) / velocity_scale,
            (-sin_yaw * vx + cos_yaw * vy) / velocity_scale,
            math.sin(yaw_rel), math.cos(yaw_rel))


def make_obs(dog_x, dog_y, dog_yaw, dog_vel, dog_w,
             ball_x, ball_y, ball_vel,
             prev_action, contact_seen, ages,
             field=DEFAULT_FIELD, obs_dim=OBS_DIM_DUEL,
             opponent=None, opponent_age=0.0,
             teammate=None, teammate_age=0.0,
             opponent2=None, opponent2_age=0.0):
    """按 `rl_control.observation` 的顺序拼 23/29/41 维观测。

    opponent / teammate / opponent2 = (x, y, yaw, (vx, vy))；
    29 维必须给 opponent；41 维必须给 opponent，后两个可给 None（填 0）。
    ages=(dog_age, ball_age)，单位秒；与部署端一样除以 0.5。
    """
    if obs_dim not in OBS_DIMS:
        raise ValueError(f'obs_dim 必须为 {OBS_DIMS} 之一')
    c = field
    co, si = math.cos(dog_yaw), math.sin(dog_yaw)

    def body(x, y):
        return (co * x + si * y, -si * x + co * y)

    dog_vel = (float(dog_vel[0]), float(dog_vel[1]))
    rel = body(ball_x - dog_x, ball_y - dog_y)
    goal = body(c.goal_x - dog_x, c.goal_y - dog_y)
    dv = body(dog_vel[0], dog_vel[1])
    bv = body(ball_vel[0], ball_vel[1])
    fc, fs = math.cos(c.yaw), math.sin(c.yaw)
    dx, dy = dog_x - c.center_x, dog_y - c.center_y
    px, py = fc * dx + fs * dy, -fs * dx + fc * dy
    half_length = ((c.goal_x - c.center_x) * fc +
                   (c.goal_y - c.center_y) * fs)
    yaw_rel = wrap_angle(dog_yaw - c.yaw)
    obs = [rel[0] / 3., rel[1] / 3., bv[0] / 2., bv[1] / 2.,
           goal[0] / 5., goal[1] / 5., dv[0], dv[1], dog_w,
           float(prev_action[0]), float(prev_action[1]), float(prev_action[2]),
           math.sin(yaw_rel), math.cos(yaw_rel),
           (px + half_length) / 9., (half_length + .6 - px) / 9.,
           (py - c.min_y) / 6., (c.max_y - py) / 6.,
           math.hypot(*rel) / 3., float(contact_seen),
           ages[0] / .5, ages[1] / .5, float(max(ages) <= .5)]
    if obs_dim == OBS_DIM_STAGE2:
        return np.clip(np.asarray(obs, dtype=np.float32), -5., 5.)
    if opponent is None:
        raise ValueError(f'{obs_dim} 维观测需要最近对手')
    # 攻防环境用外场墙壁，而不是门线
    obs[14:18] = ((px - c.min_x) / 9., (c.max_x - px) / 9.,
                  (py - c.min_y) / 6., (c.max_y - py) / 6.)
    age = max(ages[0], ages[1], opponent_age)
    obs[20:23] = (age / .5, age / .5, float(age <= .5))
    obs.extend(body_block(opponent, dog_x, dog_y, dog_yaw, dog_vel, co, si))
    if obs_dim == OBS_DIM_TEAM:
        obs.extend(body_block(teammate, dog_x, dog_y, dog_yaw, dog_vel, co, si))
        obs.extend(body_block(opponent2, dog_x, dog_y, dog_yaw, dog_vel, co, si))
    return np.clip(np.asarray(obs, dtype=np.float32), -5., 5.)
