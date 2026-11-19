"""观测槽位敏感度：看训练好的策略到底有没有在用某项信息。

做法：用该策略自己跑一段 rollout 采观测（保证在训练分布内），对指定槽位
做有限差分，统计"输入变化 → 动作变化"的增益：

    敏感度 = mean( ||a(o + δ) - a(o)|| ) / δ_物理单位

两块信息用同一物理尺度比较（位置 0.1 m、速度 0.1 m/s、朝向 0.1 rad），
就能回答"这 2 维是被真用了还是摆设"。敏感度接近 0 说明网络基本忽略它。

用法：
    python -m train.sensitivity --model runs/w_full/policy.npz --agent 0
    python -m train.sensitivity --model runs/w_full/policy.npz --steps 300
"""

import argparse

import numpy as np

from . import scripted
from .eval_match import load_actor
from .team_env import EnvConfig, TeamEnv

# (名字, 观测索引, 物理步长, 观测步长)
SLOTS = (
    ('球位置', (0, 1), 0.1, 0.1 / 3),
    ('球速度', (2, 3), 0.1, 0.1 / 2),
    ('狗自身速度', (6, 7), 0.1, 0.1),
    ('最近对手位置', (23, 24), 0.1, 0.1 / 3),
    ('最近对手速度', (25, 26), 0.1, 0.1 / 2),
    ('最近对手朝向', (27, 28), 0.1, 0.1),
    ('队友位置', (29, 30), 0.1, 0.1 / 3),
    ('队友速度', (31, 32), 0.1, 0.1 / 2),
    ('队友朝向', (33, 34), 0.1, 0.1),
    ('次近对手位置', (35, 36), 0.1, 0.1 / 3),
    ('次近对手速度', (37, 38), 0.1, 0.1 / 2),
    ('次近对手朝向', (39, 40), 0.1, 0.1),
)


def collect(actor, steps, agent, seed):
    env = TeamEnv(EnvConfig(randomize=True), seed=seed)
    env.reset()
    samples = []
    for _ in range(steps):
        obs = env.observations()
        samples.append(obs[agent])
        actions = np.stack([scripted.act(env, i) for i in range(4)])
        actions[agent] = actor(obs[agent])
        _, _, done, _ = env.step(actions)
        if done:
            env.reset()
    return np.asarray(samples, np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description='观测槽位敏感度')
    parser.add_argument('--model', required=True)
    parser.add_argument('--agent', type=int, default=0)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    actor = load_actor(args.model)
    if actor.obs_dim != 41:
        raise SystemExit('敏感度表按 41 维定义，请用 --obs-dim 41 的权重')
    obs = collect(actor, args.steps, args.agent, args.seed)
    base = np.stack([actor(o) for o in obs])
    print(f'采集 {len(obs)} 帧；策略输出 ||a|| 均值 = '
          f'{np.linalg.norm(base, axis=1).mean():.3f}\n')
    print(f'{"槽位":<14}{"敏感度":>10}{"占比":>9}')

    gains = []
    for name, index, physical, delta in SLOTS:
        shifted = obs.copy()
        for column in index:
            shifted[:, column] += delta
        moved = np.stack([actor(o) for o in shifted])
        gain = float(np.linalg.norm(moved - base, axis=1).mean() / physical)
        gains.append((name, gain))
    total = sum(g for _, g in gains) or 1.0
    for name, gain in gains:
        print(f'{name:<14}{gain:>10.3f}{100 * gain / total:>8.1f}%')


if __name__ == '__main__':
    main()
