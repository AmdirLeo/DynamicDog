"""速度档位诊断：把"跑多快 / 推球多快 / 短冲值不值"变成可比的数字。

三组测量（全部用 `train/team_env.py` 的真实动力学，不靠估算）：

1. **单次推球**：狗以该档最高速正面顶球，记录出脚球速与滚到静止的距离；
2. **带球推进速率**：一只"追球脚本狗"连续 6 s 顶球，统计球沿进攻方向
   前进的米数 / 秒 —— 这是比赛里真正值钱的指标（不是瞬时球速）；
3. **控制代价**：指令延迟 + 一阶惯性带来的过冲距离、以及从全速刹停
   需要的距离，用来判断"高速档在 10×5.5 m 场地上会不会把自己顶出界"。

用法：
    python -m train.speed                    # 默认档位表
    python -m train.speed --seconds 8

读法：
- 出脚球速 > 1.0 m/s 且滚距 ≥ 1.0 m 才谈得上"短传/射门"；
- 带球推进速率是"这档速度能把球运多快"的上限；
- 过冲距离 > 死区宽度 0.8 m 时，高速档要额外依赖观测里的墙距槽位。
"""

import argparse
import math

import numpy as np

from .team_env import (ACTION_SCALE, BALL_RADIUS, CONTACT_DISTANCE,
                       DOG_RADIUS, EnvConfig, TeamEnv)

TIERS = ((1.0, '常速 0.45'), (1.5, '中速 0.68'), (2.0, '高速 0.90'),
         (2.7, '快速 1.22'), (3.3, '极速 1.49'))


def _scale(k):
    return np.array([ACTION_SCALE[0] * k, ACTION_SCALE[1] * k, ACTION_SCALE[2]],
                    np.float32)


def _fresh_env(kind, dash=False):
    k = kind
    cfg = EnvConfig(randomize=False, action_scale=_scale(k),
                    speed_limit=0.50 * k, dash=dash, dash_speed=1.5)
    env = TeamEnv(cfg, seed=0)
    env.reset()
    env.pos[0] = [-2.0, 0.0]
    env.yaw[0] = 0.0
    env.ball = np.array([0.0, 0.0])
    env.ball_v[:] = 0.0
    env.pos[1] = [-4.5, -2.2]
    env.pos[2] = [4.6, 2.2]
    env.pos[3] = [4.6, -2.2]
    return env


def single_push(k, dash=False, steps=200):
    """正面顶球一次（触球后松脚），返回 (出脚峰值球速, 滚动距离)。"""
    env = _fresh_env(k, dash)
    origin = env.ball.copy()
    peak, launched = 0.0, None
    for step in range(steps):
        actions = np.zeros((4, 3), np.float32)
        if launched is None:
            actions[0, 0] = 1.0
        env.step(actions)
        speed = float(np.linalg.norm(env.ball_v))
        peak = max(peak, speed)
        if launched is None and float(np.linalg.norm(env.ball - origin)) > 0.05:
            launched = step
        if launched is not None and speed < 0.08:
            break
    return peak, float(np.linalg.norm(env.ball - origin))


def chase_action(env, index):
    """最简单的人肉追球：朝球转、全速前进（不预测）。"""
    x, y, yaw, _ = env.dog_state(index)
    dx, dy = env.ball[0] - x, env.ball[1] - y
    error = math.atan2(dy, dx) - yaw
    error = math.atan2(math.sin(error), math.cos(error))
    return np.array([1.0, 0.0, np.clip(1.5 * error, -1.0, 1.0)], np.float32)


def advance_rate(k, dash=False, seconds=6.0, seed=0):
    """连续追球顶球：返回 (球前进米/秒, 触球次数/秒, 狗平均速度)。"""
    env = _fresh_env(k, dash)
    env.rng = np.random.default_rng(seed)
    env.pos[0] = [-2.0, 0.0]
    env.ball = np.array([0.0, 0.0])
    start = env.ball.copy()
    steps = int(seconds / 0.05)
    touches, speed_sum = 0, 0.0
    was_contact = env.contact[0]
    for _ in range(steps):
        actions = np.zeros((4, 3), np.float32)
        actions[0] = chase_action(env, 0)
        env.step(actions)
        if env.contact[0] and not was_contact:
            touches += 1
        was_contact = env.contact[0]
        speed_sum += float(np.linalg.norm(env.v_body[0]))
    return ((float(env.ball[0] - start[0]) / seconds), touches / seconds,
            speed_sum / steps)


def main(argv=None):
    parser = argparse.ArgumentParser(description='速度档位诊断')
    parser.add_argument('--seconds', type=float, default=6.0)
    args = parser.parse_args(argv)

    print(f'接触判定：狗心-球心 < {DOG_RADIUS + BALL_RADIUS:.2f} m 发生碰撞，'
          f'触球圈 {CONTACT_DISTANCE:.2f} m；场地 10 × 5.5 m，死区宽 0.8 m\n')

    print('=== 1) 单次推球：出脚球速与滚距 ===')
    print(f'{"档位":<12}{"前进 m/s":>10}{"出脚球速":>10}{"滚距 m":>9}'
          f'{"+短冲 出脚":>12}{"+短冲 滚距":>11}')
    for k, label in TIERS:
        peak, travel = single_push(k)
        dpeak, dtravel = single_push(k, dash=True)
        print(f'{label:<12}{0.45 * k:>10.2f}{peak:>10.2f}{travel:>9.2f}'
              f'{dpeak:>12.2f}{dtravel:>11.2f}')

    print('\n=== 2) 连续带球：球沿进攻方向前进速率（6 s，无对手）===')
    print(f'{"档位":<12}{"球推进 m/s":>12}{"触球次/s":>11}{"狗均速 m/s":>12}'
          f'{"+短冲 球推进":>14}')
    for k, label in TIERS:
        rate, touch, dog = advance_rate(k, seconds=args.seconds)
        drate, _, _ = advance_rate(k, dash=True, seconds=args.seconds)
        print(f'{label:<12}{rate:>12.2f}{touch:>11.2f}{dog:>12.2f}'
              f'{drate:>14.2f}')

    print('\n=== 3) 控制代价（延迟 1~3 帧 + 惯性 τ=0.08~0.25 s）===')
    print(f'{"档位":<12}{"过冲(典型)":>12}{"过冲(最坏)":>12}{"跨半场 s":>10}'
          f'{"刹停 m":>9}{"追球 2 m 耗时 s":>16}')
    for k, label in TIERS:
        v = 0.45 * k
        typical = v * (2 * 0.05 + 0.15)
        worst = v * (3 * 0.05 + 0.25)
        cross = 5.0 / v
        stop = 0.5 * v * 0.3        # 一阶惯性回落约 0.3 s 的有效距离
        chase = 2.0 / v
        print(f'{label:<12}{typical:>12.2f}{worst:>12.2f}{cross:>10.1f}'
              f'{stop:>9.2f}{chase:>16.1f}')

    print('\n=== 4) 高速档的死球风险：球以出脚速度冲向死区 ===')
    print(f'{"档位":<12}{"出脚球速":>10}{"剩余 0.8 m 时滚距":>18}{"结论":>10}')
    for k, label in TIERS:
        peak, travel = single_push(k)
        verdict = '安全' if travel < 0.8 else ('临界' if travel < 1.3 else '会进死区')
        print(f'{label:<12}{peak:>10.2f}{travel:>18.2f}{verdict:>10}')
    print('\n注：滚距按单次推球（触球后松脚）测；实战里狗会继续追，'
          '所以"球推进 m/s"才是可比的战术量。')


if __name__ == '__main__':
    main()
