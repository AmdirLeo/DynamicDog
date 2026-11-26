"""交替训练：attack → assist → 再 attack，队友和对手逐步换成历史权重。

用法（需要 gymnasium + sb3 + torch）：
    python -m train.selfplay --rounds 4 --steps 1000000 --envs 8

每轮产出：
    runs/2v2/r1_attacker/final.zip + policy.npz
    runs/2v2/r1_assist/final.zip + policy.npz
最后一份会按 `--copy-to`（默认 pkg_soccer）落成
`policy_2v2_attack_actor.npz` / `policy_2v2_assist_actor.npz`，
**不会覆盖** 29 维的 `policy_duel_*_actor.npz`（旧的 1v1 权重继续可用）。

为什么是"交替"而不是"两狗联合训练"：sb3 是单智能体实现，两狗共用一组
权重做联合 rollout 需要自造多智能体采样器；而"我 vs 上一版的我 + 队友也是
我的上一版"这种交替自博弈在同等算力下能拿到相近的协作行为，且完全复用
现成的验证/导出链路（观测契约不变）。
"""

import argparse
import os
import shutil
import subprocess
import sys


def run(module, *args):
    command = [sys.executable, '-m', module, *[str(a) for a in args]]
    print('$ ' + ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def train_round(role, out, args, peer_teammate='', peer_opponents=''):
    common = ['--role', role, '--obs-dim', args.obs_dim,
              '--steps', args.steps, '--envs', args.envs, '--out', out]
    extra = []
    if args.no_randomize:
        extra.append('--no-randomize')
    if getattr(args, 'speed_scale', 1.0) != 1.0:
        extra += ['--speed-scale', args.speed_scale]
    extra += ['--net-width', args.net_width, '--net-depth', args.net_depth]
    if getattr(args, 'dash', False):
        extra += ['--dash', '--dash-goal-cone', args.dash_goal_cone]
    if peer_teammate:
        extra += ['--peer-teammate', peer_teammate]
    if peer_opponents:
        extra += ['--peer-opponents', peer_opponents]
    run('train.train_ppo', *common, *extra)
    policy = os.path.join(out, 'policy.npz')
    run('train.export_actor', '--model', os.path.join(out, 'final.zip'),
        '--out', policy, '--role', role, '--obs-dim', args.obs_dim,
        '--config', os.path.join(out, 'train_config.json'))
    return policy


def main(argv=None):
    parser = argparse.ArgumentParser(description='2v2 交替自博弈')
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--obs-dim', type=int, default=41, choices=(29, 41))
    parser.add_argument('--out', default='runs/2v2')
    parser.add_argument('--copy-to', default='pkg_soccer')
    parser.add_argument('--no-randomize', action='store_true')
    parser.add_argument('--speed-scale', type=float, default=1.0,
                        help='速度档位（与 train_ppo 同义）')
    parser.add_argument('--dash', action='store_true',
                        help='开启朝门门控的高速短冲推球')
    parser.add_argument('--dash-goal-cone', type=float, default=0.9)
    parser.add_argument('--net-width', type=int, default=256)
    parser.add_argument('--net-depth', type=int, default=6)
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    attacker = assist = ''
    for round_index in range(1, args.rounds + 1):
        # attack：队友使用上一轮 assist 权重，第一轮使用脚本支援。
        a_out = os.path.join(args.out, f'r{round_index}_attacker')
        attacker = train_round('attacker', a_out, args,
                               peer_teammate=assist or '',
                               peer_opponents=assist or '')
        # assist：对手和队友使用刚训好的 attack 权重，保持一攻一支援。
        d_out = os.path.join(args.out, f'r{round_index}_assist')
        assist = train_round('assist', d_out, args,
                               peer_teammate=attacker,
                               peer_opponents=attacker)
        print(f'第 {round_index} 轮完成：attack={attacker} assist={assist}')

    if args.copy_to:
        os.makedirs(args.copy_to, exist_ok=True)
        for src, name in ((attacker, f'policy_2v2_attack_actor.npz'),
                          (assist, f'policy_2v2_assist_actor.npz')):
            if src:
                target = os.path.join(args.copy_to, name)
                shutil.copyfile(src, target)
                print(f'已部署权重 → {target}')


if __name__ == '__main__':
    main()
