"""用已经验证的 shared 2v2 策略蒸馏一个较大的角色网络。

这一步只复制可迁移的控球行为，不改变旧权重。学生网络使用部署端相同的
``tanh`` 层，因而可以直接导出到机器狗；随后再用较小学习率做 PPO 微调。

示例（在 src/pkg_soccer 目录执行）：
    python -m train.distill_actor \
      --teacher pkg_soccer/policy_2v2_shared_actor.npz \
      --out runs/2v2_warm/attack_distilled.npz \
      --steps 80000 --epochs 25
"""

import argparse
import hashlib
import os

import numpy as np

from . import scripted
from pkg_soccer.rl_control import Actor
from .export_actor import deploy_forward
from .team_env import EnvConfig, TeamEnv


def collect(teacher, steps, seed, noise, role='attacker'):
    """在教师自己的闭环轨迹上采样，标签是教师的归一化动作。"""
    env = TeamEnv(EnvConfig(randomize=True, obs_dim=41,
                            action_scale=(.45, .20, .70), speed_limit=.45),
                  seed=seed)
    rng = np.random.default_rng(seed + 991)
    x = np.empty((steps, 41), np.float32)
    y = np.empty((steps, 3), np.float32)
    agent = 1 if role == 'assist' else 0
    shared_mate = True
    obs = env.reset()
    for n in range(steps):
        # _obs_one 是角色网络的 opponent-first 合约；教师用
        # observation_for_actor 得到历史 shared 的 teammate-first 合约。
        student_obs = np.asarray(obs[agent], np.float32)
        if role == 'assist':
            # 支援网络学侧后方接应/回防；不把 shared 的追球行为复制给它。
            action = scripted.act(env, agent, 'support', speed=.30)
        else:
            teacher_obs = env.observation_for_actor(agent, teacher)
            action = np.asarray(teacher(teacher_obs), np.float32)
        x[n] = student_obs
        y[n] = action
        applied = action.copy()
        if noise and rng.random() < .25:
            applied += rng.normal(0., noise, 3).astype(np.float32)
            applied = np.clip(applied, -1., 1.)
        actions = np.stack([scripted.act(env, i, scripted.default_role(i),
                                         speed=.30) for i in range(4)])
        actions[agent] = applied
        if shared_mate or role == 'assist':
            mate = env.teammate_of(agent)
            actions[mate] = teacher(env.observation_for_actor(mate, teacher))
        _, _, done, _ = env.step(actions)
        if done:
            obs = env.reset()
            agent = (agent + 2) % 4
            shared_mate = bool(rng.integers(2))
        else:
            obs = env.observations()
    return x, y


def main(argv=None):
    p = argparse.ArgumentParser(description='shared→大网络角色策略蒸馏')
    p.add_argument('--teacher', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--role', choices=('attacker', 'assist'), default='attacker',
                   help='attack 模仿 shared；assist 模仿支援脚本，teacher 控制进攻队友')
    p.add_argument('--steps', type=int, default=80000)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--seed', type=int, default=20261124)
    p.add_argument('--noise', type=float, default=.08,
                   help='采样时给教师动作的探索噪声；0 关闭')
    args = p.parse_args(argv)
    if (args.steps < 1000 or args.epochs < 1 or not 32 <= args.width <= 1024 or
            not 2 <= args.depth <= 12 or args.batch_size < 1 or
            not np.isfinite(args.noise) or args.noise < 0):
        p.error('检查 steps/epochs/width/depth/batch-size/noise 参数')
    if not args.out.endswith('.npz'):
        p.error('--out 应使用 .npz 扩展名')
    if os.path.realpath(args.teacher) == os.path.realpath(args.out):
        p.error('输出不能覆盖教师权重')

    teacher = Actor(args.teacher)
    if teacher.obs_dim != 41:
        raise SystemExit('教师/进攻队友必须是 41 维权重')
    if args.role == 'attacker' and teacher.peer_order != 'teammate_first':
        raise SystemExit('teacher 必须是 41 维 shared_2v2 权重')
    if not np.allclose(teacher.scale, [.45, .20, .70]):
        raise SystemExit('此迁移课程要求教师为 0.45 m/s 速度档')
    print(f'收集 {args.role} 训练轨迹（双向进攻）', flush=True)
    x, y = collect(teacher, args.steps, args.seed, args.noise, args.role)
    # 独立种子生成整条验证轨迹，避免相邻帧随机切分造成泄漏。
    print('收集独立验证轨迹', flush=True)
    valid_x, valid_y = collect(teacher, max(3000, args.steps // 10),
                               args.seed + 100000, 0., args.role)

    import torch
    from torch import nn
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layers = []
    prev = 41
    for _ in range(args.depth):
        layers += [nn.Linear(prev, args.width), nn.Tanh()]
        prev = args.width
    layers += [nn.Linear(prev, 3), nn.Tanh()]
    net = nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    tx = torch.from_numpy(x).to(device)
    ty = torch.from_numpy(y).to(device)
    vx = torch.from_numpy(valid_x).to(device)
    vy = torch.from_numpy(valid_y).to(device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    best_loss, best_state = float('inf'), None
    for epoch in range(args.epochs):
        perm = torch.randperm(len(tx), generator=gen, device=device)
        net.train()
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start:start + args.batch_size]
            loss = torch.mean((net(tx[idx]) - ty[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            net.eval()
            with torch.no_grad():
                pred = net(vx)
                mae = torch.mean(torch.abs(pred - vy)).item()
                val_loss = torch.mean((pred - vy) ** 2).item()
                component = torch.mean(torch.abs(pred - vy), dim=0).cpu().tolist()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
            print(f'epoch {epoch + 1}/{args.epochs}: val_mse={val_loss:.6f} '
                  f'val_mae={mae:.5f} component_mae={component}', flush=True)

    net.load_state_dict(best_state)
    net.eval()
    weights = {}
    actor_index = 0
    for module in net:
        if not isinstance(module, nn.Linear):
            continue
        weights[f'actor_{actor_index}_weight'] = module.weight.detach().cpu().numpy().astype(np.float32)
        weights[f'actor_{actor_index}_bias'] = module.bias.detach().cpu().numpy().astype(np.float32)
        actor_index += 2
    with open(args.teacher, 'rb') as handle:
        teacher_hash = hashlib.sha256(handle.read()).hexdigest()
    payload = dict(weights, role=np.str_('assist' if args.role == 'assist' else 'attack'),
                   action_scale=np.array([.45, .20, .70], np.float32),
                   obs_dim=np.int64(41), obs_layout=np.str_('2v2'),
                   contact_offset=np.array([.10, 0.], np.float32),
                   field_geometry=np.array([10., 5.5, 9.17, 1.5, .8, 8.], np.float32),
                   teacher_sha256=np.str_(teacher_hash), training_seed=np.int64(args.seed),
                   distill_steps=np.int64(args.steps), validation_mse=np.float64(best_loss),
                   initialization=np.str_('shared_imitation' if args.role == 'attacker'
                                          else 'script_support_imitation'))
    with torch.no_grad():
        expected = net(vx[:64]).cpu().numpy()
    actual = np.stack([deploy_forward(weights, observation) for observation in valid_x[:64]])
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **payload)
    print(f'已保存蒸馏权重：{args.out}（device={device}, samples={len(x)}）')


if __name__ == '__main__':
    main()
