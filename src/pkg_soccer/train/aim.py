"""量化"方向感"：球被推出去时，是朝对方球门还是朝边线/自家门。

只看进球数无法区分"方向对但没进"和"乱捅却蒙进"。本脚本统计三个直接指标：

- **朝门角**：球速 > `--min-speed` 时，球速方向与"球→对方球门"的夹角（度）。
  越小说明推球越朝门；`<20°` 记为"射正"。
- **朝门投影**：球速在"球→对方门"单位向量上的分量（m/s），正=朝对方门，
  负=朝自家门（乌龙风险）。
- **靠边占比**：球处于"距边线 0.8 m"条带内的步数比例（规则里的死区）。

用法：
    python -m train.aim --model runs/sp_k/policy.npz --dash --scripted-speed 0.60
    python -m train.aim --model runs/sp_e/policy.npz --dash --halves 4 --seed 1

口径与 `train.eval_match` 一致（同一套 TeamEnv、同一份 `Actor`），只多统计方向。
"""

import argparse
import math

import numpy as np

from . import scripted
from .team_env import (DEAD_STRIP, FIELD_WIDTH, GOAL_LINE_X, TEAM_OF,
                       EnvConfig, TeamEnv)

DT = 0.05
HALF_STEPS = int(240.0 / DT)
SHOT_ANGLE = 20.0        # 夹角小于它算"射正"


def run_half(env, actors, roles, speed, min_speed, steps=HALF_STEPS):
    """只统计"我方触球后 0.15 s 内"的球速方向 —— 那才是我们的出球。

    否则指标会被"对手推的球"和"球自由滚动"污染（基线看起来永远 ~90°）。
    触球判定：球到我方任一只狗 < 0.43 m，且球速方向在一步内变化 > 25°
    （撞击必然改变方向；球自然滚动时方向基本不变）。
    """
    stats = dict(goals=0, conceded=0, own_goal=0, dead=0, moving=0,
                 aimed=0, away=0, angle=0.0, projection=0.0, edge=0)
    window = 0                 # 触球后还剩几帧计入统计
    prev_dir = None
    for _ in range(steps):
        obs = env.observations()
        actions = np.stack([scripted.act(env, i, roles.get(i), speed=speed)
                            for i in range(4)])
        for index, actor in actors.items():
            actions[index] = np.clip(np.asarray(actor(obs[index]), np.float32),
                                     -1.0, 1.0)
        _, _, done, info = env.step(actions)

        ball_v = env.ball_v
        speed_now = float(np.linalg.norm(ball_v))
        direction = (math.atan2(ball_v[1], ball_v[0])
                     if speed_now > 1e-6 else None)
        near = float(np.min(np.linalg.norm(env.pos[:2] - env.ball, axis=1)))
        if (near < 0.43 and speed_now > min_speed and prev_dir is not None and
                abs(math.atan2(math.sin(direction - prev_dir),
                               math.cos(direction - prev_dir))) > math.radians(25)):
            window = 3             # 0.15 s
        prev_dir = direction

        if window > 0 and speed_now > min_speed:
            window -= 1
            gx, gy = GOAL_LINE_X - env.ball[0], -env.ball[1]
            norm = math.hypot(gx, gy)
            if norm > 1e-6:
                projection = (ball_v[0] * gx + ball_v[1] * gy) / norm
                cos = max(-1.0, min(1.0, projection / speed_now))
                angle = math.degrees(math.acos(cos))
                stats['moving'] += 1
                stats['projection'] += projection
                stats['angle'] += angle
                if angle < SHOT_ANGLE:
                    stats['aimed'] += 1
                if projection < 0:
                    stats['away'] += 1
        if abs(float(env.ball[1])) > FIELD_WIDTH / 2 - DEAD_STRIP:
            stats['edge'] += 1
        for kind, team in info['events']:
            if kind == 'goal':
                if int(team) == 0:
                    stats['goals'] += 1
                else:
                    stats['conceded'] += 1
                    touch = info['last_touch']
                    if touch is not None and int(TEAM_OF[touch]) == 0:
                        stats['own_goal'] += 1
            elif kind in ('dead', 'out'):
                stats['dead'] += 1
        if done:
            env.reset()
            window = 0
            prev_dir = None
    return stats


def summarize(rows):
    total = {k: sum(r[k] for r in rows) for k in rows[0]}
    moving = max(total['moving'], 1)
    n = max(len(rows), 1)
    return dict(
        goals=total['goals'] / n, conceded=total['conceded'] / n,
        own_goal=total['own_goal'] / n, dead=total['dead'] / n,
        angle=total['angle'] / moving, projection=total['projection'] / moving,
        aimed=total['aimed'] / moving, away=total['away'] / moving,
        edge=total['edge'] / (n * HALF_STEPS))


def main(argv=None):
    parser = argparse.ArgumentParser(description='方向感评估：朝门还是朝边线')
    parser.add_argument('--model', required=True, help='部署端 npz')
    parser.add_argument('--obs-dim', type=int, default=41, choices=(29, 41))
    parser.add_argument('--agent', type=int, default=None,
                        help='只控这只狗（训练形态）；不给则控本队两只（部署形态）')
    parser.add_argument('--halves', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dash', action='store_true')
    parser.add_argument('--speed-limit', type=float, default=None)
    parser.add_argument('--scripted-speed', type=float, default=0.60)
    parser.add_argument('--min-speed', type=float, default=0.30,
                        help='只统计球速大于它的步（默认 0.30 m/s）')
    parser.add_argument('--ball-restitution', type=float, default=None,
                        help='与训练 --ball-restitution 对齐')
    parser.add_argument('--ball-drag', type=float, default=None,
                        help='与训练 --ball-drag 对齐')
    parser.add_argument('--no-randomize', action='store_true')
    args = parser.parse_args(argv)

    from pkg_soccer.rl_control import Actor
    actor = Actor(args.model)
    if actor.obs_dim != args.obs_dim:
        raise SystemExit(f'权重是 {actor.obs_dim} 维，与 --obs-dim 不一致')
    scale = tuple(float(v) for v in np.asarray(actor.scale).reshape(3))
    limit = (max(0.50, scale[0]) if args.speed_limit is None
             else float(args.speed_limit))

    ball = {}
    if args.ball_restitution is not None:
        ball['ball_restitution'] = (args.ball_restitution, args.ball_restitution)
    if args.ball_drag is not None:
        ball['ball_drag'] = (args.ball_drag, args.ball_drag)
    env = TeamEnv(EnvConfig(**ball, randomize=not args.no_randomize,
                            obs_dim=args.obs_dim, action_scale=scale,
                            speed_limit=limit, dash=args.dash), seed=args.seed)
    env.reset()
    roles = {0: 'attacker', 1: 'support'}
    actors = {0: actor} if args.agent is not None else {0: actor, 1: actor}
    rows = []
    for half in range(args.halves):
        row = run_half(env, actors, roles,
                       args.scripted_speed, args.min_speed)
        rows.append(row)
        print(f'半场 {half + 1}: 进球 {row["goals"]} 失球 {row["conceded"]} '
              f'(乌龙 {row["own_goal"]}) 死球 {row["dead"]}')
    s = summarize(rows)
    print(f'合计（{args.halves} 个半场）：进球 {s["goals"]:.1f} '
          f'失球 {s["conceded"]:.1f}（乌龙 {s["own_goal"]:.1f}）'
          f'死球 {s["dead"]:.1f}')
    print(f'方向感：朝门角中位 {s["angle"]:.0f}°，射正率 {s["aimed"]:.1%}，'
          f'朝己方门占比 {s["away"]:.1%}，朝门投影 {s["projection"]:+.2f} m/s，'
          f'靠边时间 {s["edge"]:.1%}')
    print(f'口径：速度档 {scale[0]:.2f} m/s，限速 {limit:.2f}，'
          f'短冲 {"开" if args.dash else "关"}，脚本对手 {args.scripted_speed:.2f} m/s')


if __name__ == '__main__':
    main()
