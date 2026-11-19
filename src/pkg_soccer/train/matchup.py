"""两队都用真策略对打：速度档到底值不值，要看"对等对手"下的比分。

单边评估里对手是 0.30 m/s 的脚本，0.45 m/s 的狗已经够用，所以测不出
"提速能抢到更多球"。这里让左右两队各拿一份权重（各自的速度档/短冲配置），
在同一块场地上对打，直接比进球。

用法：
    python -m train.matchup \
      --left runs/sp_e/policy.npz --right runs/rwd_old/policy.npz \
      --seeds 0,1,2 --halves 2

实现要点：
- 环境只有一套 `action_scale`，所以取两队里较大的那档作为分母，
  另一队按 `动作 × 自己档位 / 环境档位` 换算，等价于各自用真实速度；
- 短冲用 `dash_mask` 逐只狗开：每队按自己 `train_config.json` 的配置；
- 两队都用"偶数号进攻、奇数号支援"的同一套角色，避免角色差异污染比较。
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .eval_match import HALF_STEPS, load_actor
from .team_env import TEAM_OF, EnvConfig, TeamEnv


def load_side(path):
    actor = load_actor(path)
    meta = {}
    config_path = Path(path).with_name('train_config.json')
    if config_path.exists():
        with open(config_path, encoding='utf-8') as handle:
            meta = json.load(handle)
    scale = np.asarray(meta.get('action_scale', actor.scale), np.float32)
    return {
        'actor': actor,
        'scale': scale,
        'dash': bool(meta.get('dash', False)),
        'goal_cone': float(meta.get('dash_goal_cone', 0.0)),
        'label': Path(path).parent.name or Path(path).name,
    }


def play(left, right, seeds=(0, 1, 2), halves=2, dash_speed=1.5,
         dash_ms=450, cooldown=1.2):
    """返回 (左队进球, 右队进球, 左队控球占比, 死球数)。"""
    world_scale = np.maximum(left['scale'], right['scale'])
    mask = np.array([left['dash'], left['dash'], right['dash'], right['dash']])
    # 两队门锥取较严的那个（环境里只有一个全局值）：对"更保守"的一队不利，
    # 所以只在两队都开门控时才启用。
    cone = min(left['goal_cone'], right['goal_cone']) \
        if (left['goal_cone'] and right['goal_cone']) else 0.0
    goals = np.zeros(2, int)
    holds = 0
    steps = 0
    dead = 0
    for seed in seeds:
        env = TeamEnv(EnvConfig(randomize=True,
                                action_scale=tuple(world_scale),
                                speed_limit=float(max(world_scale[0], 0.50)),
                                dash=bool(mask.any()), dash_mask=mask,
                                dash_speed=dash_speed,
                                dash_ms=dash_ms, dash_cooldown=cooldown,
                                dash_goal_cone=cone), seed=seed)
        env.reset()
        for _ in range(halves):
            for _ in range(HALF_STEPS):
                obs = env.observations()
                actions = np.zeros((4, 3), np.float32)
                for i in range(4):
                    side = left if TEAM_OF[i] == 0 else right
                    raw = np.asarray(side['actor'](obs[i]), np.float32)
                    actions[i] = np.clip(raw * side['scale'] / world_scale, -1, 1)
                _, _, done, info = env.step(actions)
                for kind, team in info['events']:
                    if kind == 'goal':
                        goals[int(team)] += 1
                    elif kind in ('dead', 'out'):
                        dead += 1
                if done:
                    env.reset()
                    continue
                holder_team, _, _ = env._possession()
                holds += int(holder_team == 0)
                steps += 1
    return (int(goals[0]), int(goals[1]),
            holds / steps if steps else 0.0, dead)


def main(argv=None):
    parser = argparse.ArgumentParser(description='两份策略对打')
    parser.add_argument('--left', required=True, help='0 队权重（进攻 +X）')
    parser.add_argument('--right', required=True, help='1 队权重')
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--halves', type=int, default=2)
    parser.add_argument('--swap', action='store_true',
                        help='额外打一遍交换场地（消除先攻优势）')
    args = parser.parse_args(argv)

    seeds = [int(s) for s in str(args.seeds).split(',') if s.strip()]
    left, right = load_side(args.left), load_side(args.right)
    for side in (left, right):
        print(f'{side["label"]}: 档位={tuple(float(v) for v in side["scale"])} '
              f'短冲={"开" if side["dash"] else "关"} 门锥={side["goal_cone"] or "—"}')
    l_goal, r_goal, hold, dead = play(left, right, seeds, args.halves)
    print(f'\n{left["label"]} vs {right["label"]}：{l_goal} : {r_goal}，'
          f'左队控球 {hold:.2%}，死球 {dead} 次'
          f'（{len(seeds)} 种子 × {args.halves} 个 4 min 半场）')
    if args.swap:
        l2, r2, hold2, dead2 = play(right, left, seeds, args.halves)
        print(f'交换场地后（{right["label"]} 为 0 队）：{l2} : {r2}，'
              f'原左队作为 1 队进 {r2} 球、失 {l2} 球，死球 {dead2} 次')
        print(f'合计：{left["label"]} {l_goal + r2} : {r_goal + l2} {right["label"]}')
    print('\n注意：这些策略只在"进攻 +X"上训练过，作为 1 队（进攻 -X）时行为明显变差'
          '（自对自实测 rwd_old 26:0、sp_a 9:5），所以对打比分只能当参考。'
          '判定速度/短冲是否划算，以 train/ablation.py 的单边同分配评估为准'
          '（--scripted-speed 可把对手也提速）。要修好对打，需要给 1 队做真正的'
          '镜像观测（横轴反射 + 侧向动作反号），属于另一轮的结构性改动。')


if __name__ == '__main__':
    main()
