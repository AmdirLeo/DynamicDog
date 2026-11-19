"""脚本化的队友/对手：按角色分工，避免同队两只狗挤在同一个点。

两种角色（训练初期用它当队友/对手，稳定后再换历史权重做自博弈）：

- ``attacker``：绕到球后、朝对方门推球（原行为）；
- ``support``：守在球后侧方 1.8 m 的支援位，球逼近本方门时才回追解围，
  避免两只狗同时压球、互相挡路。

输出与策略相同的归一化动作 (3,)，由环境统一限速/限幅。
"""

import math

import numpy as np

STAGE_DISTANCE = 0.38
SPEED = 0.30
SUPPORT_GAP = 1.8        # 支援位在球朝本方门方向的退距
SUPPORT_LATERAL = 1.1    # 支援位相对球—门连线的横向偏移（靠场地中线一侧）
DANGER_DISTANCE = 2.6    # 球离本方门多近时，支援位改为直接回追
ROLES = ('attacker', 'support')


def default_role(index):
    """一队两只狗默认一攻一守：偶数号进攻，奇数号支援。"""
    return 'attacker' if index % 2 == 0 else 'support'


def _drive(x, y, yaw, target, speed, scale=(0.45, 0.20, 0.70)):
    """按**米/秒**规划，再按环境的动作缩放归一化。

    `scale` 必须传环境的 `action_scale`：高速档下同一套脚本仍然跑同样的
    物理速度（0.30 m/s），这样对照实验只改变受控狗的速度，不偷偷把对手也加速。
    """
    dx, dy = target[0] - x, target[1] - y
    distance = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)
    error = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
    vx = 0.0 if abs(error) > 0.6 else min(speed, max(0.04, 0.8 * distance))
    wz = 0.0 if abs(error) <= 0.05 else float(np.clip(1.5 * error, -0.7, 0.7))
    scale = np.asarray(scale, np.float32).reshape(3)
    return np.array([vx / scale[0], 0.0, wz / scale[2]], dtype=np.float32)


def _attacker(env, index, speed):
    goal = env.attack_goal_of(index)
    ball = env.ball
    x, y, yaw, _ = env.dog_state(index)
    to_goal = np.array([goal[0] - ball[0], goal[1] - ball[1]])
    norm = float(np.linalg.norm(to_goal))
    if norm < 1e-6:
        to_goal = np.array([1.0, 0.0])
        norm = 1.0
    unit = to_goal / norm
    stage = ball - unit * STAGE_DISTANCE
    forward = (ball[0] - x) * unit[0] + (ball[1] - y) * unit[1]
    lateral = -(ball[0] - x) * unit[1] + (ball[1] - y) * unit[0]
    to_ball = math.hypot(ball[0] - x, ball[1] - y)
    to_stage = math.hypot(stage[0] - x, stage[1] - y)
    # 已在球后走廊/贴近球：直接往球门方向推过去，而不是停在球旁
    if ((to_ball < 0.55 and forward > 0.05 and abs(lateral) < 0.35) or
            (to_stage < 0.30 and forward > 0.15)):
        target = ball + unit * 0.50
    else:
        target = stage
    return _drive(x, y, yaw, target, speed, env.cfg.action_scale)


def _support(env, index, speed):
    """支援位：站在球的己方侧后 1.8 m、偏向场地中线；危险时回追解围。"""
    attack = env.attack_goal_of(index)
    own = env.own_goal_of(index)
    ball = env.ball
    x, y, yaw, _ = env.dog_state(index)
    axis = np.array([attack[0] - own[0], attack[1] - own[1]])
    norm = float(np.linalg.norm(axis))
    unit = axis / norm if norm > 1e-6 else np.array([1.0, 0.0])
    near_own = math.hypot(ball[0] - own[0], ball[1] - own[1])
    if near_own < DANGER_DISTANCE:
        # 球逼近本方门：不再守位，直接按"解围"思路追球并朝对方门推
        return _attacker(env, index, speed)
    perpendicular = np.array([-unit[1], unit[0]])
    side = -1.0 if ball[1] > 0 else 1.0        # 往场地中线那一侧收
    target = (ball - unit * SUPPORT_GAP + perpendicular * SUPPORT_LATERAL * side)
    limit_x, limit_y = env.field_limit()
    target = np.clip(target, [-limit_x, -limit_y], [limit_x, limit_y])
    return _drive(x, y, yaw, target, speed * 0.9, env.cfg.action_scale)


def act(env, index, role=None, speed=SPEED):
    role = role or default_role(index)
    if role not in ROLES:
        raise ValueError(f'未知角色 {role!r}，应为 {ROLES}')
    if role == 'support':
        return _support(env, index, speed)
    return _attacker(env, index, speed)
