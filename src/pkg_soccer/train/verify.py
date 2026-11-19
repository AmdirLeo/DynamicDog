"""自检：观测契约、环境冒烟、导出权重前向。

用法（在包含 pkg_soccer/ 与 train/ 的工作区根目录下）：
    python -m train.verify            # 全部检查
    python -m train.verify --env      # 只跑环境冒烟
    python -m train.verify --contract # 只对比 RLController 观测（29 与 41 维）
    python -m train.verify --forward  # 只检查 npz 前向

契约检查是训练与部署之间唯一可靠的"锁"：它用真实的
`pkg_soccer.rl_control.RLController.observation` 对比 `train.obs.make_obs`，
不一致就不要上狗。41 维（2v2）会用一份临时合成的权重来验证，
不依赖是否已经训好新策略。
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from .obs import (ACTION_SCALE, OBS_DIM_TEAM, SERVICE_POINTS, FieldSpec,
                  make_obs)
from .team_env import EnvConfig, TeamEnv
from . import scripted


def check_forward(models=None):
    """部署端 Actor 的前向：形状、有限值、输出严格落在 (-1, 1)。"""
    import pkg_soccer.rl_control as rl

    root = Path(__file__).resolve().parent.parent / 'pkg_soccer'
    names = models or ['policy_stage2_actor.npz',
                       'policy_duel_attacker_actor.npz',
                       'policy_duel_defender_actor.npz',
                       'policy_team_attacker_actor.npz']
    rng = np.random.default_rng(0)
    for name in names:
        path = root / name
        actor = rl.Actor(path)
        obs = rng.uniform(-5, 5, size=actor.obs_dim).astype(np.float32)
        out = actor(obs)
        assert out.shape == (3,), (name, out.shape)
        assert np.isfinite(out).all(), name
        assert np.all(np.abs(out) < 1.0), (name, out)
        print(f'  OK forward {name}: obs_dim={actor.obs_dim} '
              f'role={actor.role} layout={actor.layout} '
              f'action={np.round(out, 3)}')
    print('forward 检查通过')


def synthetic_weights(path, obs_dim, role='attacker', seed=0,
                      action_scale=ACTION_SCALE):
    """写一份随机权重，只为验证 41 维契约；不可用于上机。"""
    rng = np.random.default_rng(seed)
    np.savez(
        path,
        actor_0_weight=rng.normal(0, 0.05, (128, obs_dim)).astype(np.float32),
        actor_0_bias=np.zeros(128, np.float32),
        actor_2_weight=rng.normal(0, 0.05, (128, 128)).astype(np.float32),
        actor_2_bias=np.zeros(128, np.float32),
        actor_4_weight=rng.normal(0, 0.05, (3, 128)).astype(np.float32),
        actor_4_bias=np.zeros(3, np.float32),
        role=np.str_(role), action_scale=np.asarray(action_scale, np.float32),
        obs_dim=np.int64(obs_dim),
        obs_layout=np.str_('2v2' if obs_dim == OBS_DIM_TEAM else '1v1'))
    return str(path)


def _controller(policy='attacker', speed_limit=0.50, model_path='', **kwargs):
    from pkg_soccer.control import Config
    from pkg_soccer.rl_control import RLController

    cfg = Config(goal_x=4.585, goal_y=0.0, goal_width=1.5,
                 field_min_x=-5.0, field_max_x=5.0,
                 field_min_y=-2.75, field_max_y=2.75,
                 field_center_x=0.0, field_center_y=0.0, field_yaw=0.0)
    return RLController(cfg, model_path=model_path, speed_limit=speed_limit,
                        policy=policy, **kwargs)


def _feed(controller, kind, x, y, yaw, start, steps=4):
    for k in range(steps):
        t = start + 0.05 * k
        controller.update(kind, x, y, yaw, t, 'world', t)
    return start + 0.05 * (steps - 1)


def _slot(controller, kind):
    pose = controller.bodies[kind]
    velocity = controller.tracks[kind].velocity
    return (pose.x, pose.y, pose.yaw, (velocity[0], velocity[1]))


def check_contract():
    """RLController.observation 与 train.obs.make_obs 逐位一致（29 与 41 维）。"""
    field = FieldSpec()
    dog = (0.5, 0.2, 0.3)
    ball = (1.5, -0.4)
    near = (2.0, 0.6, 0.9)      # 更近
    far = (-2.0, -1.8, -0.4)    # 更远（41 维的"次近对手"）
    mate = (0.2, 1.4, 1.2)      # 队友
    start = 10.0

    # ---- 29 维：1v1，只有最近对手 -------------------------------------
    # 显式指定 29 维权重：包里现在还有 41 维的 policy_team_*，构造器会优先选它。
    duel = (Path(__file__).resolve().parent.parent / 'pkg_soccer'
            / 'policy_duel_attacker_actor.npz')
    controller = _controller('attacker', model_path=str(duel))
    for kind, pose in (('dog', dog), ('ball', (ball[0], ball[1], 0.0)),
                       ('opponent1', near), ('opponent2', far)):
        now = _feed(controller, kind, pose[0], pose[1], pose[2], start)
    controller.previous_action = np.array([0.1, -0.2, 0.3], np.float32)
    controller.contact_seen = True
    kind, opponent = controller.nearest_opponent(now)
    assert kind == 'opponent1', f'最近对手应为 opponent1，实得 {kind}'
    reference = controller.observation(now)
    dog_pose, ball_pose = controller.dog, controller.ball
    obs = make_obs(
        dog_pose.x, dog_pose.y, dog_pose.yaw,
        controller.tracks['dog'].velocity[:2], controller.tracks['dog'].velocity[2],
        ball_pose.x, ball_pose.y, controller.tracks['ball'].velocity[:2],
        controller.previous_action, controller.contact_seen,
        (now - dog_pose.received, now - ball_pose.received),
        field=field, obs_dim=29,
        opponent=_slot(controller, kind),
        opponent_age=now - opponent.received)
    assert obs.shape == reference.shape == (29,)
    difference = float(np.max(np.abs(obs - reference)))
    print(f'  OK contract 1v1(29) nearest={kind} max|Δ|={difference:.3e}')
    assert difference < 1e-5, '观测契约不一致：训练与部署公式不同源'

    # ---- 41 维：2v2，队友 + 最近对手 + 次近对手 ------------------------
    with tempfile.TemporaryDirectory() as folder:
        weights = synthetic_weights(Path(folder) / 'synthetic41.npz', OBS_DIM_TEAM)
        controller = _controller('attacker', model_path=weights)
        assert controller.actor.obs_dim == OBS_DIM_TEAM
        for kind, pose in (('dog', dog), ('ball', (ball[0], ball[1], 0.0)),
                           ('teammate', mate), ('opponent1', near),
                           ('opponent2', far)):
            now = _feed(controller, kind, pose[0], pose[1], pose[2], start)
        controller.previous_action = np.array([0.1, -0.2, 0.3], np.float32)
        controller.contact_seen = True
        ranked = controller.nearest_opponents(now)
        assert [k for k, _ in ranked] == ['opponent1', 'opponent2'], ranked
        reference = controller.observation(now)
        dog_pose, ball_pose = controller.dog, controller.ball
        obs = make_obs(
            dog_pose.x, dog_pose.y, dog_pose.yaw,
            controller.tracks['dog'].velocity[:2],
            controller.tracks['dog'].velocity[2],
            ball_pose.x, ball_pose.y, controller.tracks['ball'].velocity[:2],
            controller.previous_action, controller.contact_seen,
            (now - dog_pose.received, now - ball_pose.received),
            field=field, obs_dim=OBS_DIM_TEAM,
            opponent=_slot(controller, 'opponent1'),
            opponent_age=now - controller.bodies['opponent1'].received,
            teammate=_slot(controller, 'teammate'),
            teammate_age=now - controller.bodies['teammate'].received,
            opponent2=_slot(controller, 'opponent2'),
            opponent2_age=now - controller.bodies['opponent2'].received)
        assert obs.shape == reference.shape == (OBS_DIM_TEAM,)
        difference = float(np.max(np.abs(obs - reference)))
        print(f'  OK contract 2v2(41) ranked={[k for k, _ in ranked]} '
              f'max|Δ|={difference:.3e}')
        assert difference < 1e-5, '41 维观测契约不一致'

        # 队友/次近对手缺失 → 该 6 维必须全 0（部署端不阻塞控制）
        controller = _controller('attacker', model_path=weights)
        for kind, pose in (('dog', dog), ('ball', (ball[0], ball[1], 0.0)),
                           ('opponent1', near)):
            now = _feed(controller, kind, pose[0], pose[1], pose[2], start)
        filled = controller.observation(now)
        assert np.all(filled[29:35] == 0.0), filled[29:35]
        assert np.all(filled[35:41] == 0.0), filled[35:41]
        print('  OK contract 41 维缺槽位填 0（sin/cos=(0,0) 可辨识）')
    print('contract 检查通过')


def check_speed():
    """速度档与短冲：npz 自带缩放、上一动作槽位、冲刺规则两侧同源。"""
    import pkg_soccer.rl_control as rl

    # 1) 动作缩放随权重走（换档只换 npz，不改代码）
    with tempfile.TemporaryDirectory() as folder:
        fast = synthetic_weights(Path(folder) / 'fast41.npz', OBS_DIM_TEAM,
                                 action_scale=(0.90, 0.40, 0.70))
        actor = rl.Actor(fast)
        assert np.allclose(actor.scale, [0.90, 0.40, 0.70], atol=1e-6), actor.scale
        print(f'  OK 速度档：npz 自带的 action_scale={tuple(float(v) for v in actor.scale)}')

        # 2) 上一动作槽位（9-11）必须是真实指令，不能恒 0（训练端曾漏写）
        env = TeamEnv(EnvConfig(randomize=False), seed=0)
        env.reset()
        actions = np.tile(np.array([[0.9, 0.3, 0.4]], np.float32), (4, 1))
        for _ in range(6):
            obs, _, _, _ = env.step(actions)
        assert np.allclose(obs[0][9:12], [0.9, 0.3, 0.4], atol=1e-4), obs[0][9:12]
        print(f'  OK 上一动作槽位 9-11 = {np.round(obs[0][9:12], 3)}（非 0 且等于指令）')

        # 3) 冲刺触发规则：同一几何下训练端与部署端给出同一机身系指令
        env = TeamEnv(EnvConfig(randomize=False, dash=True), seed=0)
        env.reset()
        env.pos[0] = [0.0, 0.0]
        env.yaw[0] = 0.0
        env.ball = np.array([0.5, 0.0])
        dash_action = np.zeros((4, 3), np.float32)
        dash_action[0, 0] = 1.0
        env.step(dash_action)
        assert env.dash_count == 1, env.dash_count
        env_cmd = env.prev_action[0] * env.cfg.action_scale
        assert np.allclose(env_cmd, [1.5, 0.0, 0.0], atol=1e-4), env_cmd

        controller = _controller('attacker', speed_limit=1.0, model_path=fast,
                                 dash=True)
        for kind, pose in (('dog', (0.0, 0.0, 0.0)),
                           ('ball', (0.5, 0.0, 0.0)),
                           ('opponent1', (3.0, 2.0, 0.0))):
            now = _feed(controller, kind, pose[0], pose[1], pose[2], 10.0)
        deploy_cmd = controller._dash(np.array([1.0, 0.0, 0.0], np.float32),
                                      (0.45, 0.0, 0.0), now)
        assert np.allclose(deploy_cmd, env_cmd, atol=1e-6), (deploy_cmd, env_cmd)
        during = controller._dash(np.array([1.0, 0.0, 0.0], np.float32),
                                  (0.45, 0.0, 0.0), now + 0.10)
        assert np.allclose(during, env_cmd, atol=1e-6), '冲刺期内应保持冲刺指令'
        after = controller._dash(np.array([1.0, 0.0, 0.0], np.float32),
                                 (0.45, 0.0, 0.0), now + 0.70)
        assert np.allclose(after, [0.45, 0.0, 0.0]), '冷却期内不应再次冲刺'
        print(f'  OK 短冲规则两侧一致：指令={tuple(round(float(v), 3) for v in env_cmd)}，'
              '冷却期内不重复触发')


def check_env(steps=2400, seed=0):
    """环境冒烟：随机/脚本动作无异常，事件判定正确，单边进攻能进球。"""
    rng = np.random.default_rng(2)
    for obs_dim in (OBS_DIM_TEAM, 29):
        env = TeamEnv(EnvConfig(obs_dim=obs_dim), seed=seed)
        obs = env.reset()
        assert obs.shape == (4, obs_dim), (obs.shape, obs_dim)
        events = {}
        for k in range(steps):
            if k % 3 == 0:
                actions = np.stack([scripted.act(env, i) for i in range(4)])
            else:
                actions = rng.uniform(-1, 1, (4, 3)).astype(np.float32)
            obs, reward, done, info = env.step(actions)
            assert obs.shape == (4, obs_dim) and np.isfinite(obs).all()
            assert reward.shape == (4,) and np.isfinite(reward).all()
            for kind, _ in info['events']:
                events[kind] = events.get(kind, 0) + 1
            if done:
                env.reset()
        print(f'  OK env {obs_dim} 维混合动作 {steps} 步：事件={events}')

    # 41 维的槽位语义：最近对手不得超过次近对手，队友槽位必须是有效朝向
    env = TeamEnv(EnvConfig(randomize=False), seed=3)
    env.reset()
    obs = env.observations()
    for i in range(4):
        nearest, second = obs[i][23:29], obs[i][35:41]
        assert np.hypot(*nearest[4:6]) > 0.9          # sin/cos 单位向量
        assert np.hypot(*second[4:6]) > 0.9
        assert (np.hypot(*nearest[:2]) <= np.hypot(*second[:2]) + 1e-4), \
            '最近对手距离大于次近对手'
        assert np.hypot(*obs[i][29:31]) > 0.0         # 队友槽位有值
    print('  OK env 41 维槽位：最近/次近有序、队友槽位有效')

    # 事件判定：直接摆球验证，避免依赖启发式对手
    def probe(ball, velocity):
        env.pending = None
        env.reset()
        env.ball[:] = ball
        env.ball_v[:] = velocity
        _, _, _, info = env.step(np.zeros((4, 3), np.float32))
        return [kind for kind, _ in info['events']]

    assert 'goal' in probe((4.50, 0.0), (2.0, 0.0)), '门线内射门未判进球'
    assert 'goal' in probe((-4.50, 0.2), (-2.0, 0.0)), '反向进球未判'
    assert 'dead' in probe((0.0, 2.30), (0.0, 0.0)), '死区未判死球'
    assert 'out' in probe((4.95, 2.60), (2.0, 0.0)), '出界未判'
    print('  OK 事件判定：进球/死区/出界')

    # 单边进攻：0 队两只脚本狗、1 队不动，必须能进球
    # 注：单只脚本狗贴球后容易打转（脚本对手的已知弱点），这里两只都当
    # attacker 只是为了让"推球→过线→判进球"整条链路有确定的正例。
    env = TeamEnv(EnvConfig(randomize=False), seed=7)
    env.reset()
    scored = 0
    for _ in range(4000):
        actions = np.zeros((4, 3), np.float32)
        actions[0] = scripted.act(env, 0, 'attacker')
        actions[1] = scripted.act(env, 1, 'attacker')
        _, _, done, info = env.step(actions)
        scored += sum(1 for kind, _ in info['events'] if kind == 'goal')
        if done:
            env.reset()
        if scored:
            break
    assert scored, '单边脚本进攻 4000 步内没有进球'
    print(f'  OK 单边进攻：进球 {scored} 次')

    # 支援角色：不能把球推进自家门（守位 + 危险时才回追）
    env = TeamEnv(EnvConfig(randomize=False), seed=11)
    env.reset()
    for _ in range(600):
        actions = np.zeros((4, 3), np.float32)
        actions[0] = scripted.act(env, 0, 'attacker')
        actions[1] = scripted.act(env, 1, 'support')
        _, _, done, info = env.step(actions)
        if done:
            env.reset()
    assert env.score[1] == 0, '支援角色把球送进了自家门'
    print('  OK support 角色：600 步内未送乌龙')
    print('env 检查通过')


def check_match():
    """比赛循环：进球/出界 → 暂停等裁判，球回到发球点 → 自动继续（不重启进程）。"""
    from pkg_soccer.control import Config, Controller

    config = Config(goal_x=4.585, goal_y=0.0, goal_yaw=0.0, goal_width=1.5,
                    field_min_x=-5.0, field_max_x=5.0,
                    field_min_y=-2.75, field_max_y=2.75,
                    field_center_x=0.0, field_center_y=0.0, field_yaw=0.0)
    controller = Controller(config)
    points = controller.service_points()
    assert len(points) == 9, points
    # 部署端发球点必须与训练端 SERVICE_POINTS 完全一致（差一点就开始/恢复时机不同）
    expected = {(round(x, 6), round(y, 6)) for x, y in SERVICE_POINTS}
    actual = {(round(x, 6), round(y, 6)) for x, y in points}
    assert actual == expected, (sorted(actual), sorted(expected))
    assert controller.ball_at_service_point(2.5, 1.25), points
    assert controller.ball_at_service_point(-2.5, -1.25), points
    assert not controller.ball_at_service_point(3.6, 0.0), points
    assert config.max_run_time >= 480.0, config.max_run_time
    assert config.max_kicks >= 50, config.max_kicks
    print(f'  OK 发球点 {len(points)} 个（3x3 网格），单轮上限 '
          f'{config.max_run_time:.0f}s / {config.max_kicks} 次射门')

    def run(controller, dog, ball, t, seconds):
        """狗和球同步喂位姿；每个周期调用一次 command（等价 20 Hz 回调）。"""
        end = t + seconds
        while t < end - 1e-9:
            t += 0.05
            controller.update('dog', dog[0], dog[1], dog[2], t, 'world', t)
            controller.update('ball', ball[0], ball[1], 0.0, t, 'world', t)
            controller.command(t)
        return t

    # ---- 1) 进球 → 暂停，球放到发球点后继续 ----
    dog = (0.0, 0.0, 0.0)
    t = run(controller, dog, (2.0, 0.0), 0.0, 0.4)
    assert not controller.paused, controller.state
    t = run(controller, dog, (4.50, 0.0), t, 0.15)
    t = run(controller, dog, (4.75, 0.0), t, 0.15)   # 整球越过 4.585 门线
    assert controller.paused, controller.state
    assert controller.state == 'WAIT_PLAY', controller.state
    assert controller.goals == 1, controller.goals
    print(f'  OK 进球 → 暂停：{controller.reason}')
    t = run(controller, dog, (4.75, 0.0), t, 0.5)
    assert controller.paused and controller.state == 'WAIT_PLAY', controller.state
    t = run(controller, dog, (2.5, 1.25), t, 0.7)    # 裁判把球放到发球点
    assert not controller.paused, controller.reason
    assert controller.state != 'WAIT_PLAY', controller.state
    assert controller.resumes == 1 and controller.kicks == 0, controller.resumes
    assert not controller.scored and controller.goals == 1, controller.goals
    print(f'  OK 发球点确认 → 自动继续：{controller.reason}（state={controller.state}）')
    t = run(controller, dog, (2.5, 1.25), t, 0.3)
    assert controller.state in ('POSITION', 'ALIGN', 'APPROACH', 'WAIT_POSE'), \
        controller.state

    # ---- 2) 球出边线 → 暂停，放到另一个发球点也能继续 ----
    controller = Controller(config)
    t = run(controller, dog, (0.5, 0.0), 0.0, 0.4)
    t = run(controller, dog, (0.5, 3.2), t, 0.3)     # 越过边线（|y|>2.75）
    assert controller.paused and controller.state == 'WAIT_PLAY', controller.state
    assert '出界' in controller.reason, controller.reason
    assert controller.goals == 0, controller.goals
    print(f'  OK 球出界 → 暂停：{controller.reason}')
    t = run(controller, dog, (-2.5, -1.25), t, 0.7)
    assert not controller.paused and controller.resumes == 1, controller.resumes
    print(f'  OK 任意发球点 → 自动继续：{controller.reason}')

    # ---- 3) 门线后未进球（例如从门柱外侧滚过）也不能停机 ----
    controller = Controller(config)
    t = run(controller, dog, (2.0, 0.0), 0.0, 0.4)
    t = run(controller, dog, (3.5, 0.90), t, 0.2)
    t = run(controller, dog, (4.80, 1.00), t, 0.2)   # 门线后、门宽外
    assert controller.paused and not controller.finished, controller.state
    assert controller.goals == 0, controller.goals
    print(f'  OK 门线后未进球 → 暂停：{controller.reason}')

    # ---- 4) RL：球滚出边线 → 暂停；没有对手数据也能恢复 ----
    rl = _controller('attacker')
    t = 0.0
    for _ in range(12):
        t += 0.05
        for kind, pose in (('dog', (0.0, 0.0, 0.0)), ('ball', (1.0, 0.0, 0.0)),
                           ('opponent1', (2.0, 1.0, 0.0))):
            rl.update(kind, pose[0], pose[1], pose[2], t, 'world', t)
        rl.command(t)
    for y in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.4):   # 以 10 m/s 连续滚出边线
        t += 0.05
        rl.update('dog', 0.0, 0.0, 0.0, t, 'world', t)
        rl.update('ball', 1.0, y, 0.0, t, 'world', t)
        rl.update('opponent1', 2.0, 1.0, 0.0, t, 'world', t)
        rl.command(t)
    assert rl.paused and rl.state == 'WAIT_PLAY', (rl.state, rl.reason)
    print(f'  OK RL 球出界 → 暂停：{rl.reason}')
    for _ in range(16):
        t += 0.05
        rl.update('dog', 0.0, 0.0, 0.0, t, 'world', t)
        rl.update('ball', 2.5, -1.25, 0.0, t, 'world', t)
        rl.command(t)
    assert not rl.paused and rl.resumes == 1, (rl.state, rl.resumes)
    print(f'  OK RL 发球点确认 → 自动继续：{rl.reason}')
    print('match 检查通过')


def main(argv=None):
    parser = argparse.ArgumentParser(description='pkg_soccer 训练套件自检')
    parser.add_argument('--env', action='store_true')
    parser.add_argument('--contract', action='store_true')
    parser.add_argument('--forward', action='store_true')
    parser.add_argument('--speed', action='store_true')
    parser.add_argument('--match', action='store_true')
    args = parser.parse_args(argv)
    run_all = not (args.env or args.contract or args.forward or args.speed
                   or args.match)
    if run_all or args.forward:
        check_forward()
    if run_all or args.contract:
        check_contract()
    if run_all or args.speed:
        check_speed()
    if run_all or args.match:
        check_match()
    if run_all or args.env:
        check_env()


if __name__ == '__main__':
    main()
