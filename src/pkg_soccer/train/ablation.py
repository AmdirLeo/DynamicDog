"""消融评估：同算力、只改"喂不喂某项信息"时的半场对比。

用法：
    python -m train.ablation \
      --case "full=runs/abl_full/policy.npz|" \
      --case "no_vel=runs/abl_no_vel/policy.npz|body_vel" \
      --case "no_yaw=runs/abl_no_yaw/policy.npz|body_yaw" \
      --case "no_mate=runs/abl_no_mate/policy.npz|teammate" \
      --seeds 0,1,2 --halves 2

`--case` 格式：`标签=权重路径|消融设置`（消融设置必须与该权重训练时一致，
因为被抹掉的维度在训练时就一直是常量；`|` 后面留空表示不消融）。

**为什么要看稠密指标**：短预算（几十万步）下策略往往还进不了球，"进球数"全是 0，
区分不出信息价值。所以主指标看抢球效率：

- 球推进：半场内球在场地 x 轴上的平均位置（0 队进攻 +X，越大越靠前）；
- 贴球率：球在受控狗 0.43 m 触球圈内的步数占比（真实"控球"）；
- 最近率：受控狗是全场离球最近的那只的步数占比（抢球权）；
- 进球/失球/死球：仍然记录，用于确认没有学歪。
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import scripted
from .eval_match import HALF_STEPS, load_actor
from .team_env import CONTACT_DISTANCE, TEAM_OF, EnvConfig, TeamEnv

EVENT_KINDS = ('goal', 'dead', 'out', 'deadlock')


def parse_case(text):
    """'标签=路径|消融|附加' → (标签, 路径, 消融名, 附加标志集合)。

    附加标志目前只认 `dash`（评估时给环境开短冲，必须与该权重训练时一致）。
    """
    if '=' not in text:
        raise SystemExit(f'--case 需要 "标签=路径|消融"，实得 {text!r}')
    label, rest = text.split('=', 1)
    parts = rest.split('|')
    path = parts[0]
    ablate = parts[1] if len(parts) > 1 else ''
    flags = {p.strip() for p in (parts[2] if len(parts) > 2 else '').split(',')
             if p.strip()}
    names = tuple(p.strip() for p in ablate.split(',') if p.strip())
    return label.strip(), path.strip(), names, flags


def run_case(case, seeds, halves, agent, obs_dim, randomize,
             scripted_speed=0.30):
    label, path, ablate, flags = case
    actor = load_actor(path)
    if actor.obs_dim != obs_dim:
        raise SystemExit(f'{label} 权重是 {actor.obs_dim} 维，与 --obs-dim 不符')
    # 速度档写在 npz 里；短冲参数写在同目录的 train_config.json 里
    # （评估必须与训练用同一套动力学，否则不是在测同一个策略）
    meta = {}
    config_path = Path(path).with_name('train_config.json')
    if config_path.exists():
        with open(config_path, encoding='utf-8') as handle:
            meta = json.load(handle)
    scale = tuple(float(v) for v in np.asarray(actor.scale).reshape(3))
    if 'action_scale' in meta:
        scale = tuple(float(v) for v in meta['action_scale'])
    dash = bool(meta.get('dash', False)) or 'dash' in flags
    goal_cone = float(meta.get('dash_goal_cone', 0.0))
    limit = float(meta.get('speed_limit', max(0.50, scale[0])))
    team = int(agent // 2)
    totals = {kind: 0 for kind in EVENT_KINDS}
    goals_for = goals_against = 0
    episodes = 0
    ball_x = []
    wrap_rate = []
    nearest_rate = []
    possess_rate = []
    press_gap = []
    for seed in seeds:
        env = TeamEnv(EnvConfig(randomize=randomize, obs_dim=obs_dim,
                                ablate=ablate, action_scale=scale,
                                speed_limit=limit, dash=dash,
                                dash_goal_cone=goal_cone), seed=seed)
        env.reset()
        roles = {agent: 'attacker'}
        for _ in range(halves):
            before = env.score.copy()
            for _ in range(HALF_STEPS):
                obs = env.observations()
                actions = np.stack([scripted.act(env, i, roles.get(i),
                                                 speed=scripted_speed)
                                    for i in range(4)])
                actions[agent] = np.clip(
                    np.asarray(actor(obs[agent]), np.float32), -1.0, 1.0)
                _, _, done, info = env.step(actions)
                for kind, _team in info['events']:
                    totals[kind] = totals.get(kind, 0) + 1
                if done:
                    env.reset()
                    continue
                # 稠密指标：只统计场上正常推进的步（复位那一步跳过）
                ball_x.append(float(env.ball[0]) * (1 if team == 0 else -1))
                distances = np.linalg.norm(env.pos - env.ball, axis=1)
                wrap_rate.append(float(distances[agent] < CONTACT_DISTANCE))
                nearest_rate.append(float(np.argmin(distances) == agent))
                # 新奖励瞄准的行为：团队控球率 + 无球狗对持球对手的贴近程度
                holder_team, holder, _ = env._possession()
                possess_rate.append(float(holder_team == team))
                if holder_team is not None and holder_team != team:
                    gaps = [np.linalg.norm(env.pos[i] - env.pos[holder])
                            for i in np.where(TEAM_OF == team)[0]
                            if distances[i] > CONTACT_DISTANCE]
                    if gaps:
                        press_gap.append(float(min(gaps)))
            delta = env.score - before
            goals_for += int(delta[team])
            goals_against += int(delta[1 - team])
            episodes += 1
    return {
        'label': label,
        'ablate': ','.join(ablate) or '—',
        'scale': scale[0],
        'dash': bool(dash),
        'cone': goal_cone,
        'ball_x': float(np.mean(ball_x)) if ball_x else 0.0,
        'wrap': float(np.mean(wrap_rate)) if wrap_rate else 0.0,
        'nearest': float(np.mean(nearest_rate)) if nearest_rate else 0.0,
        'possession': float(np.mean(possess_rate)) if possess_rate else 0.0,
        'press': float(np.mean(press_gap)) if press_gap else float('nan'),
        'goals_for': goals_for / episodes,
        'goals_against': goals_against / episodes,
        'dead': totals['dead'] / episodes,
        'episodes': episodes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='观测信息量消融对比')
    parser.add_argument('--case', action='append', required=True,
                        help='"标签=权重路径|消融设置"，可重复')
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--halves', type=int, default=2)
    parser.add_argument('--agent', type=int, default=0)
    parser.add_argument('--obs-dim', type=int, default=41, choices=(29, 41))
    parser.add_argument('--no-randomize', action='store_true')
    parser.add_argument('--scripted-speed', type=float, default=0.30,
                        help='脚本队友/对手的物理速度（m/s）；调高可模拟'
                             '"对手也一样快"的场景')
    args = parser.parse_args(argv)

    seeds = [int(s) for s in str(args.seeds).split(',') if s.strip()]
    cases = [parse_case(text) for text in args.case]
    rows = [run_case(case, seeds, args.halves, args.agent, args.obs_dim,
                     not args.no_randomize, args.scripted_speed)
            for case in cases]

    header = (f'{"配置":<14}{"消融":<10}{"档 m/s":>7}{"冲":>3}{"球推进":>8}'
              f'{"门锥":>6}{"控球率":>8}{"贴球率":>8}{"干扰距":>8}{"进球/场":>9}'
              f'{"失球/场":>9}{"死球/场":>9}')
    print(header)
    print('-' * len(header))
    for row in rows:
        print(f'{row["label"]:<14}{row["ablate"]:<10}{row["scale"]:>7.2f}'
              f'{"是" if row["dash"] else "否":>3}'
              f'{row["cone"] if row["cone"] else "—":>6}'
              f'{row["ball_x"]:>8.2f}{row["possession"]:>8.3f}{row["wrap"]:>8.3f}'
              f'{row["press"]:>8.2f}{row["goals_for"]:>9.2f}'
              f'{row["goals_against"]:>9.2f}{row["dead"]:>9.2f}')
    print(f'\n每个配置 = {len(seeds)} 个种子 × {args.halves} 个 4 min 半场'
          f'（受控狗 {args.agent}，其余脚本；随机化='
          f'{"关" if args.no_randomize else "开"}，观测 {args.obs_dim} 维，'
          f'脚本狗 {args.scripted_speed:.2f} m/s）')
    print('球推进 = 球在进攻方向上的平均位置（米）；控球率 = 本队持球的步数占比；'
          '干扰距 = 对方持球时我方无球狗到持球者的距离（越小越贴）')


if __name__ == '__main__':
    main()
