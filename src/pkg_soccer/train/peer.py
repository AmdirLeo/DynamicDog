"""用历史权重当队友/对手（自博弈的最小实现）。

部署端 npz 里的 `Actor` 只做 numpy 前向，不需要 torch，因此训练时可以直接把
上一轮的 attacker/defender 权重塞给"另一只狗"，形成"我 vs 上一版的我"的对手池。
它的输出是 tanh 后的归一化动作，交给 `TeamEnv` 统一 ×scale 与限速。

注意：训练与部署的观测必须同维（41 维权重只能喂 41 维观测），
所以导出后权重会带 obs_dim/obs_layout 元数据，这里会显式校验。
"""

import numpy as np


class PeerPolicy:
    """把一个 npz 权重包装成"另一只狗"的动作源。"""

    def __init__(self, path, obs_dim, label='peer'):
        from pkg_soccer.rl_control import Actor

        self.actor = Actor(path)
        self.path = str(path)
        self.label = label
        if self.actor.obs_dim != int(obs_dim):
            raise ValueError(
                f'{label} 权重 {path} 是 {self.actor.obs_dim} 维，'
                f'与环境观测 {obs_dim} 维不一致；请先重训或换用同维权重')
        self.calls = 0

    def act(self, observation):
        self.calls += 1
        return np.asarray(self.actor(observation), dtype=np.float32)

    def __repr__(self):
        return (f'PeerPolicy({self.label}, obs_dim={self.actor.obs_dim}, '
                f'role={self.actor.role})')


class PeerPool:
    """一组历史权重；每次 `sample()` 随机抽一个当对手，实现自博弈的对手池。"""

    def __init__(self, paths, obs_dim, label='opponent'):
        if not paths:
            raise ValueError('对手池至少需要一个权重文件')
        self.pool = [PeerPolicy(path, obs_dim, f'{label}{i}')
                     for i, path in enumerate(paths)]

    def sample(self, rng):
        return self.pool[int(rng.integers(len(self.pool)))]

    def __len__(self):
        return len(self.pool)
