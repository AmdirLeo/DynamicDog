"""PPO 训练入口（2v2，默认 41 维观测）。

需要 gymnasium + stable-baselines3 + torch：
    pip install gymnasium "stable-baselines3>=2.2" torch

    # 1) 先训进攻狗（队友/对手是脚本角色）
    python -m train.train_ppo --role attacker --steps 5000000 --envs 8 --out runs/a41

    # 2) 训练支援狗；它的目标位由 assist shaping 固定在球侧后方
    python -m train.train_ppo --role assist --steps 5000000 --envs 8 \
        --peer-opponents pkg_soccer/policy_2v2_attack_actor.npz \
        --out runs/assist41

    # 3) 从 29 维 1v1 权重热启动（新增的两个槽位权重先置 0，行为与旧策略一致）
    python -m train.train_ppo --role attacker --init pkg_soccer/policy_duel_attacker_actor.npz \
        --steps 5000000 --out runs/a41_warm

**只训练一只狗**：受控狗（attacker=0 号，defender=1 号）由 PPO 决策，
同队另一只与对手由脚本角色或历史权重驱动。sb3 是单智能体实现，
两狗共用权重的联合训练需要自造多智能体 rollout，收益不如"轮流训练 +
对方/队友用历史权重"的交替自博弈（见 `python -m train.selfplay`）。
"""

import argparse
import json
import os

import numpy as np

from . import scripted
from .team_env import CONTACT_DISTANCE, TEAM_OF, EnvConfig, TeamEnv

# 角色 → 受控狗编号（0/1 属于 0 队）
AGENT_OF = {'attacker': 0, 'assist': 1, 'defender': 1}
# 常速档动作缩放；`--speed-scale k` 只放大前进/侧移，转向保持 0.70 rad/s
BASE_SCALE = (0.45, 0.20, 0.70)


def parse_ablate(text):
    """'body_vel,teammate' → ('body_vel', 'teammate')。"""
    return tuple(part.strip() for part in str(text).split(',') if part.strip())


def _deps():
    try:
        import gymnasium as gym
        import torch
        from torch import nn
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.distributions import (
            SquashedDiagGaussianDistribution)
        from stable_baselines3.common.policies import ActorCriticPolicy
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            '缺少训练依赖：请先 pip install gymnasium "stable-baselines3>=2.2" torch\n'
            f'原始错误：{exc}')
    return (gym, torch, nn, PPO, CheckpointCallback,
            SquashedDiagGaussianDistribution, ActorCriticPolicy, DummyVecEnv)


def role_map(agent, controlled_role):
    """4 只狗的角色：受控狗按命令行，同队另一只互补，对手一攻一守。"""
    team = int(TEAM_OF[agent])
    roles = {}
    for index in range(4):
        if index == agent:
            roles[index] = controlled_role
        elif int(TEAM_OF[index]) == team:
            roles[index] = 'assist' if controlled_role == 'attacker' else 'attacker'
        else:
            roles[index] = scripted.default_role(index)
    return roles


def build_env_class():
    """SingleAgentEnv 需要 gymnasium，延迟到调用时构造。"""
    gym, *_ = _deps()

    class SingleAgentEnv(gym.Env):
        """把 4 狗环境包装成"控制其中一只"的单智能体环境。

        peers = {'teammate': PeerPolicy|None, 'opponent': PeerPolicy|None}；
        给了 peer 就用历史权重，否则用 `scripted` 的角色策略。
        """

        metadata = {'render_modes': []}

        def __init__(self, agent=0, role='attacker', cfg=None, seed=None,
                     peers=None):
            super().__init__()
            self.env = TeamEnv(cfg, seed)
            self.agent = int(agent)
            self.roles = role_map(self.agent, role)
            self.peers = peers or {}
            self.role = role
            self.team = int(TEAM_OF[self.agent])
            obs_dim = self.env.cfg.obs_dim
            self.observation_space = gym.spaces.Box(
                -5.0, 5.0, shape=(obs_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Box(
                -1.0, 1.0, shape=(3,), dtype=np.float32)
            self._role_potential = None

        def _role_potential_value(self):
            """Role shaping keeps assist in a support lane instead of duplicating attack."""
            if self.role not in ('assist', 'attacker'):
                return 0.0
            goal = self.env.attack_goal_of(self.agent)
            own = self.env.own_goal_of(self.agent)
            ball = self.env.ball
            axis = np.asarray(goal, dtype=np.float32) - np.asarray(own, dtype=np.float32)
            norm = float(np.linalg.norm(axis))
            unit = axis / norm if norm > 1e-6 else np.array([1., 0.], dtype=np.float32)
            if self.role == 'attacker':
                target = ball - unit * scripted.STAGE_DISTANCE
            else:
                perpendicular = np.array([-unit[1], unit[0]])
                side = -1.0 if ball[1] > 0 else 1.0
                target = ball - unit * scripted.SUPPORT_GAP + perpendicular * scripted.SUPPORT_LATERAL * side
            return -float(np.linalg.norm(self.env.pos[self.agent] - target))

        def _other_action(self, index, observation):
            same_team = int(TEAM_OF[index]) == self.team
            peer = self.peers.get('teammate' if same_team else 'opponent')
            if peer is not None:
                return peer.act(observation)
            return scripted.act(self.env, index, self.roles[index],
                                speed=self.env.cfg.scripted_speed)

        def reset(self, *, seed=None, options=None):
            obs = self.env.reset()
            self._role_potential = self._role_potential_value()
            return obs[self.agent], {}

        def step(self, action):
            # 决策时刻一致：先用当前状态快照驱动其他三只狗，再统一结算一步
            snapshot = self.env.observations()
            actions = np.stack([self._other_action(i, snapshot[i])
                                for i in range(4)])
            actions[self.agent] = np.clip(np.asarray(action, np.float32), -1, 1)
            obs, reward, done, info = self.env.step(actions)
            role_potential = self._role_potential_value()
            role_reward = 0.45 * (role_potential - self._role_potential)
            if self.role == 'assist' and np.linalg.norm(self.env.pos[self.agent] - self.env.ball) < CONTACT_DISTANCE:
                role_reward -= 0.015
            self._role_potential = role_potential
            return (obs[self.agent], float(reward[self.agent] + role_reward),
                    bool(done), False, info)

    return SingleAgentEnv


def make_policy_class():
    """与部署端同构：tanh 隐藏层 + tanh 挤压高斯（确定性动作 = tanh(mean)）。"""
    gym, torch, nn, PPO, CheckpointCallback, SquashedDist, ActorCriticPolicy, _ = _deps()

    class TanhActorCriticPolicy(ActorCriticPolicy):
        def _build(self, lr_schedule):
            super()._build(lr_schedule)
            self.action_dist = SquashedDist(3)

        def _predict(self, observation, deterministic=False):
            features = self.extract_features(observation)
            return torch.tanh(self.action_net(
                self.mlp_extractor.forward_actor(features)))

    return TanhActorCriticPolicy


def tensorboard_dir(out):
    """装了 tensorboard 才写曲线；否则跳过（sb3 会直接报 ImportError）。"""
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        print('未安装 tensorboard，跳过曲线日志（pip install tensorboard 可启用）')
        return None
    return os.path.join(out, 'tb')


def load_initial_weights(model, npz_path, obs_dim, action_scale=None):
    """把部署端 npz 拷进 sb3 模型；第一层多出来的输入列置 0。

    29 维 → 41 维时，前 29 列照抄、后 12 列置 0，因此初始策略与旧权重
    完全等价（新槽位一开始被忽略），这是最稳的热启动方式。

    `action_scale` 与参考权重不同（换速度档）时，把输出层对应行的权重
    按 `旧缩放/新缩放` 缩小 —— tanh 在小信号下近似线性，所以初始"m/s 行为"
    与旧策略接近，网络只需学会"同样的动作要更早收脚"。
    """
    import torch

    reference = np.load(npz_path, allow_pickle=False)
    wanted = {}
    indices = sorted(int(key[6:-7]) for key in reference.files
                     if key.startswith('actor_') and key.endswith('_weight'))
    if not indices or indices[0] != 0:
        raise SystemExit('参考权重没有有效 actor 层')
    for index in indices:
        prefix = ('action_net' if index == indices[-1]
                  else 'mlp_extractor.policy_net.{}'.format(index))
        weight = reference[f'actor_{index}_weight']
        bias = reference[f'actor_{index}_bias']
        if index == 0 and weight.shape[1] != obs_dim:
            if weight.shape[1] > obs_dim:
                raise SystemExit(
                    f'参考权重是 {weight.shape[1]} 维，比目标 {obs_dim} 维还大')
            padded = np.zeros((weight.shape[0], obs_dim), dtype=np.float32)
            padded[:, :weight.shape[1]] = weight
            weight = padded
        if index == indices[-1] and action_scale is not None:
            old = np.asarray(reference['action_scale'], np.float32) \
                if 'action_scale' in reference else np.array([0.45, 0.20, 0.70],
                                                             np.float32)
            new = np.asarray(action_scale, np.float32)
            ratio = (old / new).astype(np.float32)
            if not np.allclose(ratio, 1.0):
                weight = weight * ratio[:, None]
                bias = bias * ratio
                print('换档热启动：输出层按 {} 缩放'.format(
                    np.round(ratio, 3).tolist()))
        wanted[f'{prefix}.weight'] = torch.as_tensor(weight, dtype=torch.float32)
        wanted[f'{prefix}.bias'] = torch.as_tensor(bias, dtype=torch.float32)
    state = model.policy.state_dict()
    with torch.no_grad():
        for name, tensor in wanted.items():
            if name not in state:
                raise SystemExit(f'{name} 形状不匹配：'
                                 f'{state.get(name).shape if name in state else None} '
                                 f'vs {tuple(tensor.shape)}')
            target = state[name]
            if target.shape != tensor.shape:
                # 扩大网络时保留旧矩阵的左上角，其余神经元用零初始化。
                expanded = torch.zeros_like(target)
                rows = min(target.shape[0], tensor.shape[0])
                cols = min(target.shape[1], tensor.shape[1]) if tensor.ndim == 2 else None
                if tensor.ndim == 2:
                    expanded[:rows, :cols].copy_(tensor[:rows, :cols])
                else:
                    expanded[:rows].copy_(tensor[:rows])
                tensor = expanded
            state[name].copy_(tensor)
    print(f'已用 {npz_path} 热启动（第一层输入 {obs_dim} 维，新增列置 0）')


def main(argv=None):
    parser = argparse.ArgumentParser(description='2v2 PPO 训练（单智能体课程）')
    parser.add_argument('--steps', type=int, default=5_000_000)
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--role', default='attacker',
                        choices=('attacker', 'assist', 'defender'))
    parser.add_argument('--obs-dim', type=int, default=41, choices=(29, 41),
                        help='41=2v2（队友+双对手），29=旧 1v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--speed-scale', type=float, default=1.0,
                        help='速度档位：动作缩放 = [0.45,0.20,0.70] × 该系数'
                             '（前进/侧移按比例放大，转向不变）。2.0 = 高速档'
                             '（0.90 m/s），3.0 = 接近协议上限 1.35 m/s')
    parser.add_argument('--speed-limit', type=float, default=None,
                        help='平移限速（m/s）。默认 = 0.45 × speed-scale')
    parser.add_argument('--dash', action='store_true',
                        help='开启高速短冲推球（450 ms × 1.5 m/s，冷却 1.2 s），'
                             '与部署端 rl_dash 同规则')
    parser.add_argument('--dash-speed', type=float, default=1.5)
    parser.add_argument('--dash-ms', type=int, default=450)
    parser.add_argument('--dash-cooldown', type=float, default=1.2)
    parser.add_argument('--dash-goal-cone', type=float, default=0.0,
                        help='>0 时只允许朝对方球门方向冲刺（rad，0.9≈52°）；'
                             '把短冲当射门，能大幅减少把球推进死区')
    parser.add_argument('--no-randomize', action='store_true')
    parser.add_argument('--body-dropout', type=float, default=0.05,
                        help='队友/次近对手槽位每步丢帧概率（0 关闭）')
    parser.add_argument('--ablate', default='',
                        help='消融（维度不变，只抹信息）：body_vel / body_yaw / '
                             'teammate / opponent2 / opponent.vel 等，逗号分隔')
    parser.add_argument('--reward-dead-ball', type=float, default=-3.0,
                        help='死球/出界时对"最后触球方"的惩罚；默认 -3。'
                             '从零训练时会诱使策略干脆不碰球，见 README')
    parser.add_argument('--reward-half', type=float, default=0.0,
                        help='半场推进项：球每深入对方半场 1 m 的每秒奖励'
                             '（己方半场为负）。默认 0=关闭，见 README §12 实测')
    parser.add_argument('--reward-possess', type=float, default=0.0,
                        help='控球奖励/步：只在进攻方于对方半场持球时结算，按队给。'
                             '默认 0=关闭；试 0.004')
    parser.add_argument('--reward-press', type=float, default=0.0,
                        help='干扰奖励/步：对方在对方半场持球时，我方无球狗贴近'
                             '持球者的程度。默认 0=关闭；试 0.004')
    parser.add_argument('--reward-edge', type=float, default=0.0,
                        help='远离死区势能项权重（Φ=w×球到最近危险边界的距离，'
                             '门前中路不计）。势能项不改最优策略，默认 0=关闭；'
                             '高速/短冲档死球多时试 0.8~1.5')
    parser.add_argument('--reward-aim', type=float, default=0.0,
                        help='方向感：球速朝对方球门的投影 × DT 的奖励/米。'
                             '默认 0=关闭；试 0.08（0.9 m/s 推 600 步约 2 分/集）')
    parser.add_argument('--reward-own', type=float, default=0.0,
                        help='方向感：球速朝**自家**球门的投影 × DT 的惩罚（填负数）。'
                             '只对最后触球队结算，默认 0=关闭；治乌龙试 -0.5 ~ -0.8')
    parser.add_argument('--reward-front', type=float, default=0.0,
                        help='坏触球惩罚（负值）：每次"在球的球门一侧"碰球（球在我与'
                             '自家门之间，只能把球顶回去）扣一次。默认 0=关闭；'
                             '试 -0.4 ~ -0.8（半场约 15~25 次坏触球）')
    parser.add_argument('--reward-back', type=float, default=0.0,
                        help='好触球奖励：在**球后方**碰球（球在我与对方门之间，'
                             '只能把球往前送）每次加这么多。默认 0=关闭；试 0.2~0.4')
    parser.add_argument('--ball-restitution', type=float, default=None,
                        help='球-狗碰撞恢复系数（默认随机 0.3~0.7）。真实推球是'
                             '"持续接触推着走"，仿真的弹性碰撞会把出脚速度高估约 1.5 倍，'
                             '球一下飞走、狗无法控制方向；试 0.10~0.20')
    parser.add_argument('--ball-drag', type=float, default=None,
                        help='球的滚动阻尼（1/s，默认随机 0.3~1.2）。调大后球滚得近，'
                             '狗来得及绕到球后再推；试 1.0~2.0')
    parser.add_argument('--shaping-ball', type=float, default=1.5,
                        help='势能项：球到对方门的距离权重（默认 1.5；'
                             '想更积极进攻试 2.0~2.5，仍是势能项，安全）')
    parser.add_argument('--shaping-stage', type=float, default=0.8,
                        help='势能项：狗到"球后点"距离的权重（默认 0.8）。'
                             '出球方向由触球前的站位决定，想纠正横向捅球可提到 1.5~2.0')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='学习率（默认 3e-4）。从权重热启动做微调时用 3e-5~1e-4，'
                             '大 lr 会把原策略直接推离局部最优')
    parser.add_argument('--ent-coef', type=float, default=0.005,
                        help='熵系数（默认 0.005）。微调时降到 0.0005~0.001，'
                             '否则探索噪声会让已经收敛的动作变抖')
    parser.add_argument('--scripted-speed', type=float, default=0.30,
                        help='脚本队友/对手的物理速度（m/s）。默认 0.30；'
                             '想模拟"对手也快"时给 0.6 左右')
    parser.add_argument('--peer', default='',
                        help='队友和对手都用这个 npz（自博弈起步）')
    parser.add_argument('--peer-teammate', default='', help='只给队友指定 npz')
    parser.add_argument('--peer-opponents', default='', help='只给对手指定 npz')
    parser.add_argument('--init', default='',
                        help='从部署端 npz 热启动（29 维可自动补零到 41 维）')
    parser.add_argument('--out', default='runs/attacker41')
    parser.add_argument('--no-tb', action='store_true',
                        help='不写 tensorboard 曲线')
    parser.add_argument('--checkpoint-every', type=int, default=200_000)
    parser.add_argument('--net-width', type=int, default=256,
                        help='每个隐藏层宽度（默认 256）')
    parser.add_argument('--net-depth', type=int, default=6,
                        help='隐藏层数量（默认 6）')
    parser.add_argument('--contact-offset-x', type=float, default=0.10,
                        help='狗刚体中心到胸部接触点的前向偏移（米）')
    parser.add_argument('--contact-offset-y', type=float, default=0.0,
                        help='狗刚体中心到胸部接触点的侧向偏移（米）')
    args = parser.parse_args(argv)
    if not 1.0 <= args.speed_scale <= 3.3:
        raise SystemExit('--speed-scale 必须在 [1.0, 3.3]（前进 ≤ 1.5 m/s）')
    if not 32 <= args.net_width <= 1024 or not 2 <= args.net_depth <= 12:
        raise SystemExit('--net-width 应在 32..1024，--net-depth 应在 2..12')
    action_scale = (BASE_SCALE[0] * args.speed_scale,
                    BASE_SCALE[1] * args.speed_scale,
                    BASE_SCALE[2])
    speed_limit = (0.45 * args.speed_scale if args.speed_limit is None
                   else args.speed_limit)

    gym, torch, nn, PPO, CheckpointCallback, _, _, DummyVecEnv = _deps()
    SingleAgentEnv = build_env_class()
    policy_class = make_policy_class()
    os.makedirs(args.out, exist_ok=True)
    agent = AGENT_OF[args.role]

    peers = {}
    teammate_path = args.peer_teammate or args.peer
    opponent_path = args.peer_opponents or args.peer
    if teammate_path or opponent_path:
        from .peer import PeerPolicy
        if teammate_path:
            peers['teammate'] = PeerPolicy(teammate_path, args.obs_dim, 'teammate')
        if opponent_path:
            peers['opponent'] = PeerPolicy(opponent_path, args.obs_dim, 'opponent')
        print('非受控狗使用历史权重：'
              + ', '.join(f'{k}={v.path}' for k, v in peers.items()))
    if args.ablate:
        print(f'消融信息（维度不变，只抹掉）：{args.ablate}')

    ball = {}
    if args.ball_restitution is not None:
        ball['ball_restitution'] = (args.ball_restitution, args.ball_restitution)
    if args.ball_drag is not None:
        ball['ball_drag'] = (args.ball_drag, args.ball_drag)

    def make_env():
        cfg = EnvConfig(**ball, speed_limit=speed_limit,
                        action_scale=action_scale,
                        randomize=not args.no_randomize,
                        obs_dim=args.obs_dim,
                        body_dropout=args.body_dropout,
                        dash=args.dash, dash_speed=args.dash_speed,
                        dash_ms=args.dash_ms,
                        dash_cooldown=args.dash_cooldown,
                        dash_goal_cone=args.dash_goal_cone,
                        reward_dead_ball=args.reward_dead_ball,
                        reward_half=args.reward_half,
                        reward_possess=args.reward_possess,
                        reward_press=args.reward_press,
                        reward_edge=args.reward_edge,
                        shaping_ball=args.shaping_ball,
                        shaping_stage=args.shaping_stage,
                        reward_aim=args.reward_aim,
                        reward_own=args.reward_own,
                        reward_front=args.reward_front,
                        reward_back=args.reward_back,
                        scripted_speed=args.scripted_speed,
                        contact_offset=(args.contact_offset_x, args.contact_offset_y),
                        ablate=parse_ablate(args.ablate))
        return SingleAgentEnv(agent=agent, role=args.role, cfg=cfg,
                              seed=args.seed, peers=peers)

    venv = DummyVecEnv([make_env for _ in range(max(1, args.envs))])
    model = PPO(
        policy_class, venv,
        n_steps=512, batch_size=2048, learning_rate=args.lr,
        gamma=0.995, gae_lambda=0.95, clip_range=0.2,
        ent_coef=args.ent_coef, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[args.net_width] * args.net_depth,
                           activation_fn=nn.Tanh,
                           log_std_init=-0.5),
        tensorboard_log=None if args.no_tb else tensorboard_dir(args.out),
        verbose=1, seed=args.seed)
    if args.init:
        load_initial_weights(model, args.init, args.obs_dim, action_scale)
    checkpoint = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every // max(1, args.envs)),
        save_path=os.path.join(args.out, 'ckpt'),
        name_prefix='ppo')
    model.learn(total_timesteps=args.steps, callback=checkpoint)
    path = os.path.join(args.out, 'final.zip')
    model.save(path)
    meta = dict(obs_dim=args.obs_dim, role=args.role,
                action_scale=[float(v) for v in action_scale],
                speed_limit=float(speed_limit),
                speed_scale=float(args.speed_scale),
                dash=bool(args.dash), dash_speed=float(args.dash_speed),
                dash_ms=int(args.dash_ms),
                dash_cooldown=float(args.dash_cooldown),
                dash_goal_cone=float(args.dash_goal_cone),
                seed=int(args.seed), steps=int(args.steps),
                reward_dead_ball=float(args.reward_dead_ball),
                reward_half=float(args.reward_half),
                reward_possess=float(args.reward_possess),
                reward_press=float(args.reward_press),
                reward_edge=float(args.reward_edge),
                reward_aim=float(args.reward_aim),
                shaping_ball=float(args.shaping_ball),
                shaping_stage=float(args.shaping_stage),
                reward_own=float(args.reward_own),
                reward_front=float(args.reward_front),
                reward_back=float(args.reward_back),
                lr=float(args.lr), ent_coef=float(args.ent_coef),
                ball_restitution=ball.get('ball_restitution'),
                ball_drag=ball.get('ball_drag'),
                scripted_speed=float(args.scripted_speed),
                body_dropout=float(args.body_dropout),
                randomize=not bool(args.no_randomize),
                peer_teammate=teammate_path, peer_opponents=opponent_path,
                init=args.init, net_width=int(args.net_width),
                net_depth=int(args.net_depth),
                contact_offset=[float(args.contact_offset_x),
                                float(args.contact_offset_y)])
    with open(os.path.join(args.out, 'train_config.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(f'训练完成：{path}')
    print(f'速度档：动作缩放={meta["action_scale"]} 限速={speed_limit:.2f} m/s '
          f'短冲={"开" if args.dash else "关"}')
    print(f'导出权重：python -m train.export_actor --model {path} '
          f'--out {args.out}/policy.npz --role {args.role} '
          f'--obs-dim {args.obs_dim} --speed-scale {args.speed_scale}')


if __name__ == '__main__':
    main()
