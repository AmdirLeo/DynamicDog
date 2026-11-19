"""抢球竞赛诊断：自己比对手快，到底能换来什么。

只测"速度本身能决定的部分"，把策略水平的影响剥掉。做法是**单狗计时**：
一只狗从静止出发，全速冲向指定位置静止的球，记录首次进入触球圈的时间；
两只狗的"竞赛"就是比较各自的时间（不互相阻挡，避免纠缠干扰读数）。

三张表：

1. **到球耗时**：不同档位跑到 2 / 3 / 5 m 的球要多久；
2. **等距 / 带劣势竞赛**：我方速度倍率 → 能容忍多大距离劣势还先触球；
3. **救球（第二落点）**：球被弹到侧前方，双方等距起步谁先到。

用法：
    python -m train.race
    python -m train.race --lateral 1.5
"""

import argparse
import math

import numpy as np

from .team_env import ACTION_SCALE, CONTACT_DISTANCE, EnvConfig, TeamEnv

DT = 0.05


def _scale(k):
    return np.array([ACTION_SCALE[0] * k, ACTION_SCALE[1] * k, ACTION_SCALE[2]],
                    np.float32)


def time_to_ball(k, distance, lateral=0.0, steps=600):
    """单狗从静止出发触到球的时间（秒）；触不到返回 None。

    用比例转向 + 按航向误差折减前进（和真机上"先转头再走"一致），
    避免"朝反方向也全速冲"这种仿真里才会出现的假动作。
    """
    world = _scale(k)
    cfg = EnvConfig(randomize=False, action_scale=tuple(world),
                    speed_limit=float(world[0]))
    env = TeamEnv(cfg, seed=0)
    env.reset()
    ball = np.array([0.0, lateral])
    env.ball = ball.copy()
    env.pos[0] = np.array([-distance, 0.0])
    env.pos[1] = np.array([-4.6, 2.4])
    env.pos[2] = np.array([4.6, -2.4])
    env.pos[3] = np.array([4.6, 2.4])
    env.yaw[0] = math.atan2(lateral, distance)
    speed = 0.45 * k
    for step in range(steps):
        x, y, yaw, _ = env.dog_state(0)
        error = math.atan2(env.ball[1] - y, env.ball[0] - x) - yaw
        error = math.atan2(math.sin(error), math.cos(error))
        forward = speed * max(0.0, math.cos(error)) if abs(error) < 1.0 else 0.0
        actions = np.zeros((4, 3), np.float32)
        actions[0] = np.array([forward / world[0], 0.0,
                               np.clip(1.5 * error / world[2], -1, 1)],
                              np.float32)
        env.step(actions)
        if float(np.linalg.norm(env.pos[0] - env.ball)) < CONTACT_DISTANCE:
            return (step + 1) * DT
    return None


def handicap(our_k, their_k, their_distance, lateral=0.0, ceiling=6.0):
    """我方还能先触球的最大距离。"""
    their_time = time_to_ball(their_k, their_distance, lateral)
    if their_time is None:
        return 0.0
    best = 0.0
    distance = 0.3
    while distance <= ceiling:
        ours = time_to_ball(our_k, distance, lateral)
        if ours is None or ours > their_time:
            break
        best = distance
        distance += 0.1
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description='抢球竞赛诊断')
    parser.add_argument('--lateral', type=float, default=1.5,
                        help='救球场景里球的横向偏移（米）')
    parser.add_argument('--their-distance', type=float, default=1.5)
    args = parser.parse_args(argv)

    tiers = (1.0, 1.5, 2.0, 2.7, 3.3)
    print('到球耗时（秒；从静止出发、含 1~3 帧指令延迟与一阶惯性）')
    print(f'{"档 m/s":>8}{"1.5 m":>9}{"3.0 m":>9}{"5.0 m":>9}{"相对 0.45":>11}')
    base = time_to_ball(1.0, 3.0)
    for k in tiers:
        times = [time_to_ball(k, d) for d in (1.5, 3.0, 5.0)]
        label = lambda t: '—' if t is None else f'{t:.2f}'
        gain = f'{base / times[1]:.2f}×' if times[1] else '—'
        print(f'{0.45 * k:>8.2f}{label(times[0]):>9}{label(times[1]):>9}'
              f'{label(times[2]):>9}{gain:>11}')

    print(f'\n带距离劣势（对手距球 {args.their_distance:.1f} m、同为 0.45 m/s 档）')
    print(f'{"我方档 m/s":>10}{"临界距离 m":>12}{"距离比":>8}{"纯匀速理论比":>13}')
    for k in tiers:
        best = handicap(k, 1.0, args.their_distance)
        print(f'{0.45 * k:>10.2f}{best:>12.2f}'
              f'{best / args.their_distance:>8.2f}{k:>13.2f}')

    print(f'\n救球（球被弹到侧前方，横向 {args.lateral:.1f} m，双方等距 1.6 m）')
    print(f'{"我方档 m/s":>10}{"我方耗时":>10}{"对手 0.45":>11}{"赢多少 ms":>11}')
    their = time_to_ball(1.0, 1.6, args.lateral)
    for k in tiers:
        ours = time_to_ball(k, 1.6, args.lateral)
        if ours is None or their is None:
            print(f'{0.45 * k:>10.2f}{"—":>10}{"—":>11}{"—":>11}')
            continue
        print(f'{0.45 * k:>10.2f}{ours:>10.2f}{their:>11.2f}'
              f'{(their - ours) * 1000:>11.0f}')

    print('\n读法：前两张表是"速度本身"的收益（不含触球之后的控球质量）。'
          '实测 0.90 档在完整评估里没赢球，输在触球之后自己把球推进死区，'
          '见 train/README.md §14.4。')


if __name__ == '__main__':
    main()
