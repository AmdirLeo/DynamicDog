"""短程传球可行性诊断：用仿真本身的物理量出脚速度、滚动距离和口径误差。

做法：让一只脚本狗以最大前进速度（动作 vx=1 → 0.45 m/s）正面顶球，
记录球的出脚速度、滚到静止的距离和到达 0.6/1.0/1.5 m 的时间；
再按"出脚方向 = 狗心→球心法线"这一事实，把站位横向误差换算成传球角度误差。

用法：
    python -m train.passing                 # 3 种恢复系数 × 3 种摩擦
    python -m train.passing --trials 3      # 每组重复几次取均值（真机会抖）

结论怎么读：
- 滚动距离 ≥ 想传的距离才谈得上"传到位"；
- 滚动时间要和队友到位时间比：狗最快 0.45 m/s，1 m 要 2.2 s；
- 出脚角度误差决定需要多大的接球容差。
"""

import argparse
import math

import numpy as np

from .team_env import (BALL_RADIUS, CONTACT_DISTANCE, DOG_RADIUS, EnvConfig,
                       TeamEnv)

BALL_STOP_SPEED = 0.08
PUSH_DISTANCE = 0.60        # 顶球前狗到球的初始距离
GOAL_MARKS = (0.6, 1.0, 1.5, 2.0)


def one_push(ball_e, drag, steps=260):
    """让 0 号狗正面顶球，返回 (出脚速度, 滚动距离, 各里程耗时)。"""
    env = TeamEnv(EnvConfig(randomize=False), seed=0)
    env.reset()
    env.ball_e = float(ball_e)
    env.ball_drag = float(drag)
    env.pos[0] = [-PUSH_DISTANCE, 0.0]
    env.yaw[0] = 0.0
    env.pos[1] = [-4.5, -2.2]
    env.pos[2] = [4.6, 2.2]
    env.pos[3] = [4.6, -2.2]
    env.ball = np.array([0.0, 0.0])
    env.ball_v[:] = 0.0
    env.pos[1] = [-4.5, -2.2]

    origin = env.ball.copy()
    peak = 0.0
    launched_at = None
    marks = {}
    for k in range(steps):
        actions = np.zeros((4, 3), np.float32)
        # 触球后立刻松脚，让球自由滚动（模拟"推一下就收"的最省力踢法）
        if launched_at is None:
            actions[0, 0] = 1.0
        _, _, done, _ = env.step(actions)
        speed = float(np.linalg.norm(env.ball_v))
        if speed > peak:
            peak = speed
        travelled = float(np.linalg.norm(env.ball - origin))
        if launched_at is None and travelled > 0.05:
            launched_at = k
        for mark in GOAL_MARKS:
            if mark not in marks and travelled >= mark:
                marks[mark] = (k - (launched_at or k)) * 0.05
        if launched_at is not None and speed < BALL_STOP_SPEED and travelled > 0.10:
            return peak, travelled, marks, (k - launched_at) * 0.05
        if done:
            break
    return peak, float(np.linalg.norm(env.ball - origin)), marks, None


def main(argv=None):
    parser = argparse.ArgumentParser(description='短程传球可行性诊断')
    parser.add_argument('--trials', type=int, default=1)
    args = parser.parse_args(argv)

    print(f'狗心到球心接触距离 = {DOG_RADIUS + BALL_RADIUS:.2f} m，'
          f'触球判定圈 = {CONTACT_DISTANCE:.2f} m，狗速上限 0.45 m/s（动作饱和）\n')
    print(f'{"恢复系数e":>9}{"摩擦":>7}{"出脚速度":>10}{"滚动距离":>10}'
          f'{"到0.6m":>8}{"到1.0m":>8}{"到1.5m":>8}{"滚停耗时":>10}')
    for ball_e in (0.3, 0.5, 0.7):
        for drag in (0.3, 0.8, 1.2):
            peaks, travels, times, rolls = [], [], [], []
            for _ in range(max(1, args.trials)):
                peak, travel, marks, roll = one_push(ball_e, drag)
                peaks.append(peak)
                travels.append(travel)
                times.append(marks)
                rolls.append(roll if roll is not None else float('nan'))
            mean = lambda xs: sum(xs) / len(xs)
            def mark_time(mark):
                vals = [t[mark] for t in times if mark in t]
                return mean(vals) if vals else float('nan')
            print(f'{ball_e:>9.1f}{drag:>7.1f}{mean(peaks):>10.2f}'
                  f'{mean(travels):>10.2f}{mark_time(0.6):>8.2f}'
                  f'{mark_time(1.0):>8.2f}{mark_time(1.5):>8.2f}'
                  f'{np.nanmean(rolls):>10.2f}')

    print('\n站位误差 → 传球角度误差 → 1 m / 2 m 处的横向偏差'
          '（出脚方向就是狗心→球心法线）')
    print(f'{"横向站位误差":>14}{"角度误差":>10}{"1 m 偏差":>10}{"2 m 偏差":>10}')
    for offset in (0.02, 0.05, 0.10, 0.15):
        angle = math.degrees(math.atan2(offset, DOG_RADIUS + BALL_RADIUS))
        print(f'{offset:>14.2f}{angle:>10.1f}{math.tan(math.radians(angle)):>10.2f}'
              f'{2 * math.tan(math.radians(angle)):>10.2f}')

    print('\n参照：队友接球容差 = 触球圈 0.43 m；狗横向移动 0.5 m 需约 '
          f'{0.5 / 0.45:.1f} s（全速直行）')

    print('\n要把球送到 d 米外，出脚速度至少要 sqrt(2·a·d)（a=摩擦减速度）：')
    print(f'{"距离 m":>8}{"a=0.3":>9}{"a=0.8":>9}{"a=1.2":>9}'
          f'{"最乐观(0.75)结论":>20}')
    for distance in (0.6, 1.0, 2.0, 3.0, 5.0):
        row = [math.sqrt(2 * a * distance) for a in (0.3, 0.8, 1.2)]
        best = row[0]
        verdict = ('够' if best <= 0.70 else
                   '勉强' if best <= 0.85 else '不够，需 kick 通道')
        print(f'{distance:>8.1f}{row[0]:>9.2f}{row[1]:>9.2f}{row[2]:>9.2f}'
              f'{verdict:>20}')
    print('注：0.75 m/s 是 e=0.7、狗速 0.45 m/s 时的出脚上限；'
          '真机 push 模式 1.5 m/s 脉冲可让球出脚约 2.2 m/s（够 2~3 m 传球）')


if __name__ == '__main__':
    main()
