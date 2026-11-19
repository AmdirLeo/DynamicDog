"""单腿踢球：把「位控姿态」串成一段开环踢腿动作（纯逻辑，不依赖 ROS）。

依据官方《运动控制二次开发接口》（MiRoboticsLab/blogs cyberdog_loco_cn.md）
与 `motion_id_map.toml`，以及真机踩坑记录：

- 走 ROS2 时用 `/<dog_name>/motion_result_cmd`（protocol/srv/MotionResultCmd），
  字段会映射到运控的 `robot_control_cmd_lcmt`；
- `motion_id = 212` 位控姿态-相对姿态（**推荐**，官方 contact/foot_pose 就是
  在 mode/gait 21/0 下描述的）；`211` 绝对姿态要求 pos_des.z 给机身高度
  （0.1~0.32 m），给 0 会执行超时（code 3028）；`213` 相对姿态-抬腿低刚度；
- **duration 必须 > 0**，用于开环序列：每条命令指定目标姿态，duration 控制到位速度；
- `contact` 低 4 位：1=支撑，0=抬起；默认 0x0F（四腿都支撑）。
  抬一条腿：`0b1110` 右前(FR)、`0b1101` 左前(FL)、`0b1011` 右后(RR)、`0b0111` 左后(RL)；
  位的顺序是 FR、FL、RR、RL；
- `foot_pose[0:3]`：抬起那条腿的足端坐标偏移（机身系 x 前、y 左、z 上，米）；
  `foot_pose[3:6]` 官方预留给视觉行走后腿；
- `pos_des`：质心位置偏移（米）；`rpy_des`：机身姿态（弧度）；
- `duration` 单位毫秒，最小执行单元 2 ms。

真机注意：
1. 先确认 **没在充电**（motion_status.switch_status != 14 CHARGING）；
2. 发 111 后要等真正站稳（motion_id=112 或 111 进度≥96 且 switch_status=0），
   再等 1~2 s 让运控稳定，否则第一条位控命令容易超时（3028）；
3. 第一条命令是 `enter`（切到位控姿态），之后才是移重心/抬腿/前摆。

这段代码只负责产生"第几条命令、什么时候发、发什么"，不碰 ROS；
`kick_test.py`、`kick_action.py` 和 `soccer.py` 共用同一套参数与序列。
"""

import math

LEG_MASKS = {
    'fr': 0b1110,   # 抬右前
    'fl': 0b1101,   # 抬左前
    'rr': 0b1011,   # 抬右后
    'rl': 0b0111,   # 抬左后
}
# 机身系 y 轴向左为正：左腿在 +y，右腿在 -y
LEG_Y_SIGN = {'fr': -1.0, 'fl': 1.0, 'rr': -1.0, 'rl': 1.0}
ALL_CONTACT = 0x0F


class KickConfig:
    """一次踢腿的参数；全部可在 ROS 命令行覆盖。"""

    def __init__(self, leg='fr', motion_id=212, cmd_source=0,
                 lift_height=0.10, strike_height=0.10,
                 forward_reach=0.18, back_offset=-0.06, side_offset=0.0,
                 body_shift=0.0, body_height=0.22, body_pitch=0.0,
                 transition_ms=600, enter_ms=400,
                 windup_ms=200, lift_ms=250, swing_ms=100,
                 retract_ms=200, recover_ms=300,
                 phase_scale=1.0):
        self.leg = str(leg).lower()
        if self.leg not in LEG_MASKS:
            raise ValueError('kick_leg 只能是 fr/fl/rr/rl')
        if motion_id not in (211, 212, 213):
            raise ValueError('kick_motion_id 只能是 211/212/213')
        self.motion_id = int(motion_id)
        self.cmd_source = int(cmd_source)
        self.lift_height = self._number('lift_height', lift_height, 0.02, 0.25)
        self.strike_height = self._number('strike_height', strike_height, 0.0, 0.25)
        self.forward_reach = self._number('forward_reach', forward_reach, 0.02, 0.35)
        self.back_offset = self._number('back_offset', back_offset, -0.20, 0.10)
        self.side_offset = self._number('side_offset', side_offset, -0.15, 0.15)
        self.body_shift = self._number('body_shift', body_shift, 0.0, 0.08)
        self.body_height = self._number('body_height', body_height, 0.10, 0.32)
        self.body_pitch = self._number('body_pitch', body_pitch, -0.20, 0.20)
        self.transition_ms = self._ms('transition_ms', transition_ms, 100, 1500)
        self.enter_ms = self._ms('enter_ms', enter_ms, 100, 1500)
        self.windup_ms = self._ms('windup_ms', windup_ms)
        self.lift_ms = self._ms('lift_ms', lift_ms)
        self.swing_ms = self._ms('swing_ms', swing_ms, 60, 1000)
        self.retract_ms = self._ms('retract_ms', retract_ms)
        self.recover_ms = self._ms('recover_ms', recover_ms)
        self.phase_scale = self._number('phase_scale', phase_scale, 0.5, 2.0)

    @staticmethod
    def _number(name, value, low, high):
        value = float(value)
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{name} 必须在 [{low}, {high}]，实得 {value}')
        return value

    @staticmethod
    def _ms(name, value, low=50, high=2000):
        value = int(value)
        if not low <= value <= high:
            raise ValueError(f'{name} 必须在 [{low}, {high}] ms，实得 {value}')
        return value

    @classmethod
    def from_params(cls, get):
        """从 ROS 参数读取（get(name, default) 返回参数值）。"""
        return cls(
            leg=get('kick_leg', 'fr'),
            motion_id=int(get('kick_motion_id', 212)),
            cmd_source=int(get('kick_cmd_source', 0)),
            lift_height=float(get('kick_lift_height', 0.10)),
            strike_height=float(get('kick_strike_height', 0.10)),
            forward_reach=float(get('kick_forward_reach', 0.18)),
            back_offset=float(get('kick_back_offset', -0.06)),
            side_offset=float(get('kick_side_offset', 0.0)),
            body_shift=float(get('kick_body_shift', 0.0)),
            body_height=float(get('kick_body_height', 0.22)),
            body_pitch=float(get('kick_body_pitch', 0.0)),
            transition_ms=int(get('kick_transition_ms', 600)),
            enter_ms=int(get('kick_enter_ms', 400)),
            windup_ms=int(get('kick_windup_ms', 200)),
            lift_ms=int(get('kick_lift_ms', 250)),
            swing_ms=int(get('kick_swing_ms', 100)),
            retract_ms=int(get('kick_retract_ms', 200)),
            recover_ms=int(get('kick_recover_ms', 300)),
            phase_scale=float(get('kick_phase_scale', 1.0)),
        )

    def scaled(self, duration_ms):
        """按 phase_scale 缩放动作时长；服务要求 duration > 0。"""
        return max(20, int(round(duration_ms * self.phase_scale)))

    def body_pose(self, x, y):
        """212 相对姿态：z=0 表示不改高度；211 绝对姿态：z 必须是机身高度。"""
        if self.motion_id == 211:
            return (x, y, self.body_height)
        return (x, y, 0.0)


class KickStep:
    """一条待发送的位控姿态命令。"""

    __slots__ = ('name', 'duration_ms', 'contact', 'foot_pose',
                 'pos_des', 'rpy_des')

    def __init__(self, name, duration_ms, contact, foot_pose,
                 pos_des=(0.0, 0.0, 0.0), rpy_des=(0.0, 0.0, 0.0)):
        self.name = name
        self.duration_ms = int(duration_ms)
        self.contact = int(contact)
        values = [float(v) for v in foot_pose]
        # 只提供抬起腿的前 3 个分量；后 3 位官方预留给视觉行走
        if len(values) == 3:
            values += [0.0, 0.0, 0.0]
        if len(values) != 6:
            raise ValueError('foot_pose 必须是 3 或 6 个分量')
        self.foot_pose = values
        self.pos_des = [float(v) for v in pos_des]
        self.rpy_des = [float(v) for v in rpy_des]

    def __repr__(self):
        return (f'KickStep({self.name}, {self.duration_ms}ms, '
                f'contact=0b{self.contact:04b}, foot={self.foot_pose}, '
                f'pos={self.pos_des})')


def build_sequence(cfg):
    """enter 切位控 → [windup 移重心(可选)] → preload 屈膝收腿 → extend 伸小腿击球 → retract → recover。

    默认 body_shift=0：机身姿态与质心偏移全程为 0，只有一条腿的足端在动，
    相当于"身体站住、小腿伸出踢球"。需要轮椅式重心转移时才把 kick_body_shift > 0。
    """
    mask = LEG_MASKS[cfg.leg]
    shift = -LEG_Y_SIGN[cfg.leg] * cfg.body_shift
    steps = [KickStep('enter', cfg.scaled(cfg.enter_ms), ALL_CONTACT,
                      (0.0, 0.0, 0.0), cfg.body_pose(0.0, 0.0))]
    if cfg.body_shift > 1e-6:
        steps.append(KickStep('windup', cfg.scaled(cfg.windup_ms), ALL_CONTACT,
                              (0.0, 0.0, 0.0), cfg.body_pose(0.0, shift)))
    steps += [
        KickStep('preload', cfg.scaled(cfg.lift_ms), mask,
                 (cfg.back_offset, cfg.side_offset, cfg.lift_height),
                 cfg.body_pose(0.0, 0.0)),
        KickStep('extend', cfg.scaled(cfg.swing_ms), mask,
                 (cfg.forward_reach, cfg.side_offset, cfg.strike_height),
                 cfg.body_pose(0.0, 0.0), (0.0, cfg.body_pitch, 0.0)),
        KickStep('retract', cfg.scaled(cfg.retract_ms), mask,
                 (cfg.back_offset * 0.6, cfg.side_offset,
                  cfg.lift_height * 0.8), cfg.body_pose(0.0, 0.0)),
        KickStep('recover', cfg.scaled(cfg.recover_ms), ALL_CONTACT,
                 (0.0, 0.0, 0.0), cfg.body_pose(0.0, -shift)),
    ]
    return steps


def build_request(service_type, cfg, step):
    """把 KickStep 填进 protocol/srv/MotionResultCmd 的请求。

    所有数组字段都显式赋值：ROS 侧数组默认是空的，直接留空可能让
    桥接层拒绝请求。contact 也必须显式给（默认 0 会被解释成"四腿都抬"）。
    """
    request = service_type.Request()
    request.motion_id = cfg.motion_id
    request.cmd_source = cfg.cmd_source
    request.duration = int(step.duration_ms)
    request.contact = int(step.contact)
    request.vel_des = [0.0, 0.0, 0.0]
    request.rpy_des = list(step.rpy_des)
    request.pos_des = list(step.pos_des)
    request.acc_des = [0.0] * 6
    request.ctrl_point = [0.0, 0.0, 0.0]
    request.foot_pose = list(step.foot_pose)
    request.step_height = [0.0, 0.0]
    request.value = 0
    request.gait_toml = ''
    request.toml_data = ''
    return request


class KickRunner:
    """按 duration 节奏把序列喂给调用方；时间由调用方提供（单调秒）。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.steps = build_sequence(cfg)
        self.index = 0
        self.next_at = None
        self.started = None
        self.reason = ''

    def start(self, now):
        self.started = now
        self.index = 0
        # 先停步态、再切位控，给运控留切换时间
        self.next_at = now + self.cfg.transition_ms / 1000.0
        self.reason = '单腿踢球中'

    def poll(self, now):
        """到点则返回下一条 KickStep，否则 None。"""
        if self.started is None or self.index >= len(self.steps):
            return None
        if now < self.next_at:
            return None
        step = self.steps[self.index]
        self.index += 1
        # 中间步提前 10% 发，让运控任务栈无缝衔接；最后一步等满整段时长
        factor = 0.9 if self.index < len(self.steps) else 1.0
        self.next_at = now + step.duration_ms / 1000.0 * factor
        return step

    @property
    def done(self):
        return self.started is not None and self.index >= len(self.steps)

    def ready_to_end(self, now):
        return self.done and now >= self.next_at

    def recover_step(self):
        """异常/退出时的兜底：四腿支撑、足端和质心归零。"""
        return KickStep('recover', self.cfg.scaled(self.cfg.recover_ms),
                        ALL_CONTACT, (0.0, 0.0, 0.0),
                        self.cfg.body_pose(0.0, 0.0))


def _self_test():
    """命令行自检：python -m pkg_soccer.leg_kick"""
    for leg in ('fr', 'fl', 'rr', 'rl'):
        cfg = KickConfig(leg=leg)
        steps = build_sequence(cfg)
        assert steps[0].contact == ALL_CONTACT
        assert steps[2].contact == LEG_MASKS[leg]
        assert steps[-1].contact == ALL_CONTACT
        assert all(s.duration_ms > 0 for s in steps)
        assert all(len(s.foot_pose) == 6 and len(s.pos_des) == 3 for s in steps)
        print(f'{leg}: ' + ' -> '.join(
            f'{s.name}({s.duration_ms}ms,0b{s.contact:04b})' for s in steps))
    cfg = KickConfig(leg='fr')
    runner = KickRunner(cfg)
    runner.start(0.0)
    sent = []
    for tick in range(0, 500):
        now = tick * 0.01
        step = runner.poll(now)
        if step is not None:
            sent.append((round(now, 2), step.name))
        if runner.ready_to_end(now):
            break
    print('时序：', sent)
    assert [name for _, name in sent] == ['enter', 'preload', 'extend',
                                          'retract', 'recover']
    # 默认身体不动：所有 pos_des 必须为 0
    assert all(abs(v) < 1e-9 for step in runner.steps for v in step.pos_des)
    print('单腿踢球序列自检通过')


if __name__ == '__main__':
    _self_test()
