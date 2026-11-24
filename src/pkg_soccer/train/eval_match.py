"""按复赛规则跑 4 分钟半场评估（只用 numpy + 部署端 Actor，训练机也能跑）。

用法：
    python -m train.eval_match                                   # 全脚本基线
    python -m train.eval_match --model pkg_soccer/policy_duel_attacker_actor.npz
    python -m train.eval_match --model ... --agent 0             # 只控一只狗
    python -m train.eval_match --model ... --team 1 --halves 2    # 控 1 队两只狗
    python -m train.eval_match --model ... --obs-dim 41 --no-randomize

默认（不给 --agent）用同一套权重**同时控制本队两只狗**，这才是真实的 2v2 形态：
每只狗各自吃自己的 41 维观测（含队友与双对手槽位），互不共享状态。
评估口径对齐规则：半场 4 min（4800 步 @20 Hz），进球后由环境复位到
被进球方发球点/出发点，死球/出界就近发球；score 累计。
"""

import argparse
import math

import numpy as np

from . import scripted
from .team_env import DOG_RADIUS, EnvConfig, TeamEnv

DT = 0.05
HALF_STEPS = int(240.0 / DT)


def load_actor(path):
    from pkg_soccer.rl_control import Actor
    return Actor(path)


def controlled_indices(args):
    if args.agent is not None:
        return [args.agent]
    return [0, 1] if args.team == 0 else [2, 3]


def run_half(env, actors, roles, steps=HALF_STEPS, speed=scripted.SPEED,
             metrics=None, team=0):
    events = {}
    for _ in range(steps):
        obs = env.observations()
        actions = np.stack([scripted.act(env, i, roles.get(i), speed=speed)
                            for i in range(4)])
        for index, actor in actors.items():
            actions[index] = np.clip(np.asarray(actor(env.observation_for_actor(index, actor)), np.float32),
                                     -1.0, 1.0)
        _, _, done, info = env.step(actions)
        if metrics is not None:
            first, second = team * 2, team * 2 + 1
            near = np.linalg.norm(env.pos[first] - env.pos[second]) < 2 * DOG_RADIUS + .18
            both_ball = all(np.linalg.norm(env.pos[i] - env.ball) < .8
                            for i in (first, second))
            metrics['steps'] = metrics.get('steps', 0) + 1
            metrics['teammate_near'] = metrics.get('teammate_near', 0) + int(near)
            metrics['both_near_ball'] = metrics.get('both_near_ball', 0) + int(both_ball)
        for kind, _team in info['events']:
            events[kind] = events.get(kind, 0) + 1
        if done:
            env.reset()
    return events


def main(argv=None):
    parser = argparse.ArgumentParser(description='2v2 半场评估')
    parser.add_argument('--model', default='', help='部署端 npz；留空则全部脚本')
    parser.add_argument('--assist-model', default='', help='本队第二只狗的独立 assist 权重')
    parser.add_argument('--agent', type=int, default=None,
                        help='只控这只狗；不给则控本队两只')
    parser.add_argument('--team', type=int, default=0, choices=(0, 1),
                        help='受控球队（0=左半场先攻 +X）')
    parser.add_argument('--obs-dim', type=int, default=41, choices=(29, 41))
    parser.add_argument('--halves', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ball-restitution', type=float, default=None,
                        help='与训练 --ball-restitution 对齐')
    parser.add_argument('--ball-drag', type=float, default=None,
                        help='与训练 --ball-drag 对齐')
    parser.add_argument('--no-randomize', action='store_true')
    parser.add_argument('--ablate', default='',
                        help='与训练一致的消融设置（body_vel / body_yaw / teammate …）')
    parser.add_argument('--dash', action='store_true',
                        help='开启短冲推球（必须与该权重训练时一致）')
    parser.add_argument('--scripted-speed', type=float, default=scripted.SPEED,
                        help='脚本对手速度（m/s），与训练 --scripted-speed 对齐')
    parser.add_argument('--speed-limit', type=float, default=None,
                        help='平移限速（m/s）。默认 max(0.50, scale[0])；'
                             '0.90 档权重训练时常用 1.0，不给会裁掉对角指令')
    args = parser.parse_args(argv)
    if args.halves < 1:
        parser.error('--halves 必须为正整数')
    if args.assist_model and (not args.model or args.agent is not None):
        parser.error('--assist-model 需要 --model，且不能指定 --agent')

    ablate = tuple(part.strip() for part in args.ablate.split(',') if part.strip())
    actors = {}
    roles = {}
    # 速度档从权重里读：评估必须用与训练相同的 action_scale，否则 m/s 对不上
    scale, limit = (0.45, 0.20, 0.70), 0.50
    if args.model:
        actor = load_actor(args.model)
        scale = tuple(float(v) for v in np.asarray(actor.scale).reshape(3))
        limit = max(0.50, scale[0])
    if args.speed_limit is not None:
        if not (math.isfinite(args.speed_limit) and args.speed_limit > 0):
            raise SystemExit('--speed-limit 必须是正数')
        limit = float(args.speed_limit)
    ball = {}
    if args.ball_restitution is not None:
        ball['ball_restitution'] = (args.ball_restitution, args.ball_restitution)
    if args.ball_drag is not None:
        ball['ball_drag'] = (args.ball_drag, args.ball_drag)
    env = TeamEnv(EnvConfig(**ball, randomize=not args.no_randomize,
                            obs_dim=args.obs_dim, ablate=ablate,
                            action_scale=scale, speed_limit=limit,
                            dash=args.dash), seed=args.seed)
    env.reset()
    if args.model:
        if actor.obs_dim != args.obs_dim:
            raise SystemExit(f'权重是 {actor.obs_dim} 维，与 --obs-dim '
                             f'{args.obs_dim} 不一致')
        actors = {i: actor for i in controlled_indices(args)}
        if args.assist_model:
            assist = load_actor(args.assist_model)
            if (assist.obs_dim != args.obs_dim or assist.role != 'assist' or
                    not np.allclose(assist.scale, actor.scale)):
                raise SystemExit('assist 权重维度、角色或速度档不匹配')
            actors[controlled_indices(args)[1]] = assist
        # 受控队：一只按角色进攻、一只按角色支援；未被控的狗按默认角色
        for index in actors:
            roles[index] = 'attacker' if index % 2 == 0 else 'support'
    events = {}
    scores = []
    metrics = {}
    for half in range(args.halves):
        # 独立种子和全新开场，不能让上半场的死球状态污染下一次评估。
        env = TeamEnv(env.cfg, seed=args.seed + half)
        half_events = run_half(env, actors, roles,
                               speed=args.scripted_speed, metrics=metrics,
                               team=args.team)
        scores.append(tuple(int(v) for v in env.score))
        for kind, count in half_events.items():
            events[kind] = events.get(kind, 0) + count
        print(f'半场 {half + 1}: 比分 0队={int(env.score[0])} '
              f'1队={int(env.score[1])} 事件={half_events}')
    totals = np.sum(scores, axis=0)
    scored = sum(s[args.team] > 0 for s in scores)
    won = sum(s[args.team] > s[1-args.team] for s in scores)
    print(f'至少进一球的半场={scored}/{args.halves}；获胜半场={won}/{args.halves}；'
          '这不是单次射门命中率；固定 attack/assist，不含实机角色切换。')
    print('队友距离<0.62m 时间比例={:.2%}；同时距球<0.8m 时间比例={:.2%}；'
          '这两项是接近程度，不是碰撞次数。'.format(
              metrics['teammate_near'] / metrics['steps'],
              metrics['both_near_ball'] / metrics['steps']))
    print(f'合计：0队={int(totals[0])} 1队={int(totals[1])} '
          f'事件={events} 受控狗={sorted(actors) or "无"} 模式='
          f'{"策略" if actors else "全脚本"} 观测={args.obs_dim} 维 '
          f'速度档={scale[0]:.2f}m/s 短冲={"开" if args.dash else "关"}')


if __name__ == '__main__':
    main()
