"""把 sb3 PPO 模型导出成部署端可加载的 npz（并做确定性前向一致性检查）。

用法：
    python -m train.export_actor --model runs/a41/final.zip \
        --out pkg_soccer/policy_duel_attacker_actor.npz \
        --role attacker --obs-dim 41

导出布局（与 pkg_soccer/rl_control.py::Actor 一致）：
    actor_0_weight/bias -> 128 x obs_dim
    actor_2_weight/bias -> 128 x 128
    actor_4_weight/bias -> 3 x 128
    29/41 维还写 role / action_scale / obs_dim / obs_layout / field_geometry。
    obs_layout = 1v1（29 维）或 2v2（41 维），仅供部署端日志核对。
"""

import argparse

import numpy as np

ACTION_SCALE = np.array([0.45, 0.20, 0.70], np.float32)
# 复赛场地：长、宽、门刚体间距、门宽、死区、僵持秒数
FIELD_GEOMETRY = np.array([10.0, 5.5, 9.17, 1.5, 0.8, 8.0], np.float32)
OBS_DIMS = (23, 29, 41)
LAYOUT_OF = {23: 'single', 29: '1v1', 41: '2v2'}
LAYERS = (('actor_0', 'mlp_extractor.policy_net.0'),
          ('actor_2', 'mlp_extractor.policy_net.2'),
          ('actor_4', 'action_net'))


def deploy_forward(weights, observation):
    """与 pkg_soccer.rl_control.Actor.__call__ 完全相同的 numpy 前向。"""
    x = np.asarray(observation, dtype=np.float32)
    for index in (0, 2, 4):
        x = np.tanh(weights[f'actor_{index}_weight'] @ x +
                    weights[f'actor_{index}_bias'])
    return x


def export(model_path, out_path, role='attacker', obs_dim=29, check=True,
           action_scale=ACTION_SCALE):
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device='cpu')
    state = model.policy.state_dict()
    weights = {}
    for index, prefix in enumerate(('mlp_extractor.policy_net.0',
                                    'mlp_extractor.policy_net.2',
                                    'action_net')):
        name = f'actor_{index * 2}'
        weights[f'{name}_weight'] = state[f'{prefix}.weight'].numpy().astype(np.float32)
        weights[f'{name}_bias'] = state[f'{prefix}.bias'].numpy().astype(np.float32)
    shape = weights['actor_0_weight'].shape
    if shape != (128, obs_dim):
        raise SystemExit(f'第一层形状 {shape} 与 obs_dim={obs_dim} 不符')
    if weights['actor_4_weight'].shape != (3, 128):
        raise SystemExit('输出层形状应为 (3, 128)')

    payload = dict(weights)
    action_scale = np.asarray(action_scale, np.float32).reshape(3)
    if obs_dim != 23:
        payload.update(role=np.str_(role), action_scale=action_scale,
                       obs_dim=np.int64(obs_dim),
                       obs_layout=np.str_(LAYOUT_OF[obs_dim]),
                       field_geometry=FIELD_GEOMETRY)
    np.savez(out_path, **payload)

    if check:
        observation = np.random.default_rng(0).uniform(
            -5, 5, size=obs_dim).astype(np.float32)
        exported = deploy_forward(weights, observation)
        predicted = model.predict(observation[None], deterministic=True)[0]
        difference = float(np.max(np.abs(exported - predicted)))
        print(f'确定性动作一致性 max|Δ| = {difference:.3e}')
        if difference > 1e-5:
            raise SystemExit(
                '导出与 sb3 predict 不一致：检查训练策略类是否在输出端做了 '
                'tanh（SquashedDiagGaussian），以及 _predict 是否返回 tanh(mean)。')
    print(f'已导出 {out_path}（obs_dim={obs_dim} / {LAYOUT_OF[obs_dim]}，'
          f'role={role}）')
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description='导出 PPO 权重为 npz')
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--role', default='attacker',
                        choices=('attacker', 'defender', 'stage2'))
    parser.add_argument('--obs-dim', type=int, default=41, choices=OBS_DIMS)
    parser.add_argument('--speed-scale', type=float, default=1.0,
                        help='速度档位，必须与训练时一致（动作缩放 = '
                             '[0.45,0.20,0.70] × 系数，转向不变）')
    parser.add_argument('--config', default='',
                        help='可选：直接读训练输出的 train_config.json')
    parser.add_argument('--no-check', action='store_true')
    args = parser.parse_args(argv)
    if args.obs_dim == 23 and args.role != 'stage2':
        raise SystemExit('23 维只用于单狗 stage2')
    if args.obs_dim != 23 and args.role == 'stage2':
        raise SystemExit('29/41 维必须指定 attacker 或 defender')
    scale = np.array([0.45 * args.speed_scale, 0.20 * args.speed_scale, 0.70],
                     np.float32)
    if args.config:
        import json
        with open(args.config, encoding='utf-8') as handle:
            meta = json.load(handle)
        scale = np.asarray(meta['action_scale'], np.float32)
        print(f'从 {args.config} 读到 action_scale='
              f'{tuple(float(v) for v in scale)}')
    if not 0 < scale[0] <= 1.5 + 1e-6:
        raise SystemExit('前进动作缩放必须在 (0, 1.5] m/s')
    export(args.model, args.out, args.role, args.obs_dim, not args.no_check,
           scale)


if __name__ == '__main__':
    main()
