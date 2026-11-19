#!/usr/bin/env python3
"""CyberDog 2 单腿踢球独立测试脚本（单文件，可单独拷到机器狗上运行）。

不依赖 pkg_soccer 包，只依赖机器狗自带的 rclpy 与 protocol：
    ros2 service /<dog_name>/motion_result_cmd   (protocol/srv/MotionResultCmd)
    ros2 topic   /<dog_name>/motion_status        (protocol/msg/MotionStatus)

用法（在机器狗 NX 应用板上）：
    scp kick_action.py mi@192.168.44.1:/home/mi/
    ssh mi@192.168.44.1
    source /opt/ros/galactic/setup.bash      # 用你们平时 ros2 run 的环境
    python3 kick_action.py --leg fr          # 默认 dry-run：只打印，不动
    python3 kick_action.py --leg fr --execute  # 真机执行一次

重要（真机踩坑记录）：
1. 狗在充电（motion_status.switch_status=14 CHARGING）时不要试动作，
   脚本会直接拒绝；请拔充电器/下桩，确认 switch_status=0。
2. 位控姿态用 motion_id=212（相对姿态，官方文档的 contact+foot_pose 就是
   在 mode/gait 21/0 下描述的）。211 是"绝对姿态"，pos_des 的 z 必须给
   机身高度（0.1~0.32 m），给 0 会让运控执行超时（code 3028）。
3. 发 111 站立后要等真正站稳（motion_id=112 或 111 进度≥96 且
   switch_status=0），再等 1~2 s 让运控稳定，否则位控会超时。
4. 退出（含 Ctrl+C）会补发"四腿支撑 + 归位"和 111 站立。

动作设计（身体尽量不动）：
    enter(切位控) → preload(屈膝收腿) → extend(小腿伸出击球) → retract → recover
    默认 kick_body_shift=0：每一条命令的 pos_des 都是 (0,0,0)，机身姿态不动，
    只有被抬起那条腿的足端在动；需要重心转移时才加 --shift。

常用参数：
    --leg fr|fl|rr|rl       踢腿
    --motion-id 212|211|213 位控模式（默认 212）
    --scale 1.5             放慢动作便于观察
    --repeat 2              连踢
    --lift/--reach/--strike 屈膝收腿高度/小腿前伸距离/击球高度
    --shift 0.03            默认 0（身体不动）；抬不起腿时再加
    --cmd-source 0|4        命令来源（默认 0=APP）
    --allow-charging        允许在充电状态尝试（不建议）
    --self-test             只做序列自检，不需要 ROS
"""

import argparse
import math
import signal
import sys
import time

# 注意：rclpy / protocol 是"按需导入"的，不在这里 import：
#   - `--self-test` 和默认 dry-run 只打印序列，不需要 ROS，在普通电脑上就能跑；
#   - 真正 `--execute` 执行时才在 main()（rclpy.init 所在处）和 make_node()
#     （KickNode 的 ROS 方法会用到 rclpy）里导入。
# 如果改成顶层 import，那么在没装 rclpy / 没 source ROS 环境的电脑上，
# 连 `--self-test` 和 `--help` 都会直接报 ImportError。

# ---------------------------------------------------------------- 动作定义
# contact 低 4 位：1=支撑、0=抬起；位序 FR、FL、RR、RL
LEG_MASKS = {
    'fr': 0b1110,   # 抬右前
    'fl': 0b1101,   # 抬左前
    'rr': 0b1011,   # 抬右后
    'rl': 0b0111,   # 抬左后
}
LEG_Y_SIGN = {'fr': -1.0, 'fl': 1.0, 'rr': -1.0, 'rl': 1.0}
ALL_CONTACT = 0x0F

# 运控错误码（官方 common.hpp MotionManagerCodeDescribe_，见 _tools/notes/touch_dance.md）
MOTION_CODES = {
    3003: '无法执行当前状态',
    3008: '不支持该指令',
    3010: '参数超范围',
    3011: '忙碌中',
    3013: '电量低',
    3028: '动作执行超时',
    3029: '动作执行失败',
    3040: '急停',
    3041: '周边有障碍物无法站立',
}
SWITCH_STATUS = {
    0: 'Done', 1: 'TRANSITIONING', 2: 'ESTOP', 3: 'EDAMP', 4: 'LIFTED',
    5: 'BAN_TRANS', 6: 'OVER_HEAT', 7: 'LOW_BAT', 8: 'ORI_ERR',
    9: 'FOOTPOS_ERR', 10: 'STAND_STUCK', 11: 'MOTOR_OVER_HEAT',
    12: 'MOTOR_OVER_CURR', 13: 'MOTOR_ERR', 14: 'CHARGING',
}


class KickConfig:
    """一次踢腿的全部参数。"""

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
            raise ValueError('--leg 只能是 fr/fl/rr/rl')
        if motion_id not in (211, 212, 213):
            raise ValueError('--motion-id 只能是 211/212/213')
        self.motion_id = int(motion_id)
        self.cmd_source = int(cmd_source)
        self.lift_height = float(lift_height)
        self.strike_height = float(strike_height)
        self.forward_reach = float(forward_reach)
        self.back_offset = float(back_offset)
        self.side_offset = float(side_offset)
        self.body_shift = float(body_shift)
        self.body_height = float(body_height)
        self.body_pitch = float(body_pitch)
        self.transition_ms = int(transition_ms)
        self.enter_ms = int(enter_ms)
        self.windup_ms = int(windup_ms)
        self.lift_ms = int(lift_ms)
        self.swing_ms = int(swing_ms)
        self.retract_ms = int(retract_ms)
        self.recover_ms = int(recover_ms)
        self.phase_scale = float(phase_scale)
        checks = (('lift_height', self.lift_height, 0.02, 0.25),
                  ('strike_height', self.strike_height, 0.0, 0.25),
                  ('forward_reach', self.forward_reach, 0.02, 0.35),
                  ('back_offset', self.back_offset, -0.20, 0.10),
                  ('side_offset', self.side_offset, -0.15, 0.15),
                  ('body_shift', self.body_shift, 0.0, 0.08),
                  ('body_height', self.body_height, 0.10, 0.32),
                  ('body_pitch', self.body_pitch, -0.20, 0.20),
                  ('phase_scale', self.phase_scale, 0.5, 2.0))
        for name, value, low, high in checks:
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{name} 必须在 [{low}, {high}]，实得 {value}')
        for name, value in (('transition_ms', self.transition_ms),
                            ('enter_ms', self.enter_ms),
                            ('windup_ms', self.windup_ms),
                            ('lift_ms', self.lift_ms),
                            ('swing_ms', self.swing_ms),
                            ('retract_ms', self.retract_ms),
                            ('recover_ms', self.recover_ms)):
            if not 20 <= value <= 2000:
                raise ValueError(f'{name} 必须在 [20, 2000] ms，实得 {value}')

    def scaled(self, duration_ms):
        return max(20, int(round(duration_ms * self.phase_scale)))

    def body_pose(self, x, y):
        """212 相对姿态：z=0 表示不改高度；211 绝对姿态：z 必须是机身高度。"""
        if self.motion_id == 211:
            return (x, y, self.body_height)
        return (x, y, 0.0)


class KickStep:
    __slots__ = ('name', 'duration_ms', 'contact', 'foot_pose',
                 'pos_des', 'rpy_des')

    def __init__(self, name, duration_ms, contact, foot_pose,
                 pos_des=(0.0, 0.0, 0.0), rpy_des=(0.0, 0.0, 0.0)):
        self.name = name
        self.duration_ms = int(duration_ms)
        self.contact = int(contact)
        values = [float(v) for v in foot_pose]
        if len(values) == 3:
            values += [0.0, 0.0, 0.0]
        if len(values) != 6:
            raise ValueError('foot_pose 必须是 3 或 6 个分量')
        self.foot_pose = values
        self.pos_des = [float(v) for v in pos_des]
        self.rpy_des = [float(v) for v in rpy_des]

    def __repr__(self):
        return (f'{self.name}: {self.duration_ms}ms contact=0b{self.contact:04b} '
                f'foot=({self.foot_pose[0]:+.3f},{self.foot_pose[1]:+.3f},'
                f'{self.foot_pose[2]:+.3f}) pos=({self.pos_des[0]:+.3f},'
                f'{self.pos_des[1]:+.3f},{self.pos_des[2]:+.3f})')


def build_sequence(cfg):
    """enter 切位控 → [windup 移重心(可选)] → preload 屈膝收腿 → extend 伸小腿击球 → retract → recover。

    默认 body_shift=0：机身姿态与质心偏移全程为 0，只有一条腿的足端在动，
    即"身体站住、小腿伸出踢球"。
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
    """全部 14 个字段显式赋值：数组留空/contact=0 可能被桥接层误解。"""
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
    """按 duration 节奏推进序列；时间由调用方提供。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.steps = build_sequence(cfg)
        self.index = 0
        self.next_at = 0.0

    def start(self, now):
        self.index = 0
        self.next_at = now + self.cfg.transition_ms / 1000.0

    def poll(self, now):
        if self.index >= len(self.steps) or now < self.next_at:
            return None
        step = self.steps[self.index]
        self.index += 1
        factor = 0.9 if self.index < len(self.steps) else 1.0
        self.next_at = now + step.duration_ms / 1000.0 * factor
        return step

    def ready_to_end(self, now):
        return self.index >= len(self.steps) and now >= self.next_at

    def recover_step(self):
        return KickStep('recover', self.cfg.scaled(self.cfg.recover_ms),
                        ALL_CONTACT, (0.0, 0.0, 0.0), self.cfg.body_pose(0.0, 0.0))


def print_plan(cfg):
    steps = build_sequence(cfg)
    total = cfg.transition_ms + sum(s.duration_ms for s in steps)
    print('=== 单腿踢球序列 ===')
    print(f'腿={cfg.leg}  motion_id={cfg.motion_id}  cmd_source={cfg.cmd_source}  '
          f'过渡={cfg.transition_ms}ms  总时长≈{total / 1000.0:.2f}s')
    print(f'{"阶段":<8}{"时长":>6}  {"contact":>9}  foot_pose(x,y,z)          pos_des(x,y,z)')
    for step in steps:
        fp = step.foot_pose
        pd = step.pos_des
        print(f'{step.name:<8}{step.duration_ms:>5}ms  0b{step.contact:04b}   '
              f'({fp[0]:+.3f},{fp[1]:+.3f},{fp[2]:+.3f})   '
              f'({pd[0]:+.3f},{pd[1]:+.3f},{pd[2]:+.3f})')
    print('contact 位序：FR FL RR RL，1=支撑 0=抬起；'
          f'0b{LEG_MASKS[cfg.leg]:04b} 表示只抬 {cfg.leg}')
    print('身体：pos_des 全 0（默认身体不动，只动足端）；'
          '需要移重心时加 --shift 0.02~0.03。')
    print(f'motion_id=212 是相对姿态（推荐）；211 为绝对姿态，pos_des.z 用 '
          f'{cfg.body_height:.2f} m 机身高度。')
    print()


# ---------------------------------------------------------------- ROS 部分
def make_node(args, cfg):
    # 延迟导入：只有真机 --execute 路径会走到这里
    import rclpy
    from protocol.msg import MotionStatus
    from protocol.srv import MotionResultCmd
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    prefix = ('/' + args.dog_name.strip('/')) if args.dog_name else ''
    service_name = args.service or f'{prefix}/motion_result_cmd'
    status_topic = args.status_topic or f'{prefix}/motion_status'

    class KickNode(Node):
        def __init__(self):
            super().__init__('leg_kick_action')
            self.service_type = MotionResultCmd
            self.status = None
            self.qos = QoSProfile(depth=1,
                                  reliability=ReliabilityPolicy.BEST_EFFORT)
            self.client = self.create_client(MotionResultCmd, service_name)
            self.create_subscription(MotionStatus, status_topic,
                                     self.on_status, self.qos)
            self.get_logger().info(f'服务={service_name}  状态={status_topic}')

        def on_status(self, msg):
            self.status = msg

        def spin_for(self, seconds):
            end = time.monotonic() + seconds
            while rclpy.ok() and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.02)

        def wait_until(self, predicate, timeout, what):
            end = time.monotonic() + timeout
            while rclpy.ok() and time.monotonic() < end:
                if predicate():
                    return True
                rclpy.spin_once(self, timeout_sec=0.02)
            self.get_logger().error(f'等待{what}超时（{timeout:.0f}s）')
            return False

        def call(self, request, timeout=3.0):
            future = self.client.call_async(request)
            end = time.monotonic() + timeout
            while rclpy.ok() and not future.done() and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.01)
            if not future.done():
                raise TimeoutError('motion_result_cmd 服务调用超时')
            return future.result()

        def standing(self):
            status = self.status
            return (status is not None and
                    (status.motion_id == 112 or
                     (status.motion_id == 111 and status.order_process_bar >= 96)))

        def settled(self):
            status = self.status
            return (status is not None and status.switch_status == 0 and
                    (status.motion_id == 112 or
                     (status.motion_id == 111 and status.order_process_bar >= 96)))

        def healthy(self):
            status = self.status
            if status is None:
                return False
            motor_ok = (len(status.motor_error) > 0 and
                        all((int(v) & 0x3FFFFFFF) == 0
                            for v in status.motor_error))
            return (status.ori_error == 0 and status.footpos_error == 0 and
                    status.switch_status == 0 and motor_ok)

        def describe(self):
            status = self.status
            if status is None:
                return '无状态'
            switch = SWITCH_STATUS.get(status.switch_status,
                                       str(status.switch_status))
            return (f'motion_id={status.motion_id} switch={status.switch_status}'
                    f'({switch}) progress={status.order_process_bar} '
                    f'ori={status.ori_error} foot={status.footpos_error}')

        def stand(self):
            request = self.service_type.Request()
            request.motion_id = 111
            request.cmd_source = 0
            request.duration = 0
            request.contact = ALL_CONTACT
            for name, size in (('vel_des', 3), ('rpy_des', 3), ('pos_des', 3),
                               ('acc_des', 6), ('ctrl_point', 3),
                               ('foot_pose', 6), ('step_height', 2)):
                setattr(request, name, [0.0] * size)
            response = self.call(request, 3.0)
            self.get_logger().info(
                f'站立请求：result={response.result} code={response.code} '
                f'({MOTION_CODES.get(response.code, "未知")})')
            if not response.result:
                raise RuntimeError('站立请求被拒绝，先人工确认狗的状态')
            self.wait_until(self.settled, args.stand_timeout, '站稳')
            # 给运控留出稳定时间，避免位控命令赶在切换中途
            self.spin_for(args.stand_settle)

        def send_step(self, step, last):
            request = build_request(self.service_type, cfg, step)
            self.get_logger().info(f'发送 {step.name}：{step!r}  当前 {self.describe()}')
            began = time.monotonic()
            response = self.call(request, 3.0)
            elapsed = time.monotonic() - began
            code_text = MOTION_CODES.get(response.code, '未知')
            self.get_logger().info(
                f'{step.name}: result={response.result} code={response.code}'
                f'({code_text}) ({elapsed * 1000:.0f}ms)  当前 {self.describe()}')
            if not response.result:
                raise RuntimeError(f'{step.name} 被运控拒绝：code={response.code} '
                                   f'（{code_text}）')
            hold = step.duration_ms / 1000.0 * (1.0 if last else 0.9) - elapsed
            if hold > 0:
                self.spin_for(hold)

        def kick_once(self):
            runner = KickRunner(cfg)
            runner.start(time.monotonic())
            while rclpy.ok():
                now = time.monotonic()
                step = runner.poll(now)
                if step is not None:
                    self.send_step(step, runner.index >= len(runner.steps))
                if runner.ready_to_end(now):
                    break
                rclpy.spin_once(self, timeout_sec=0.01)

        def recover(self, final_stand):
            if args.dry_run:
                return
            try:
                request = build_request(self.service_type, cfg,
                                        KickRunner(cfg).recover_step())
                response = self.call(request, 1.5)
                self.get_logger().info(
                    f'恢复四腿支撑：result={response.result} code={response.code}')
                self.spin_for(0.35)
                if final_stand:
                    self.stand()
            except Exception as exc:
                self.get_logger().error(f'恢复动作失败：{exc}')

    return KickNode()


# ---------------------------------------------------------------- 入口
def self_test():
    for leg in ('fr', 'fl', 'rr', 'rl'):
        cfg = KickConfig(leg=leg)
        steps = build_sequence(cfg)
        assert [s.name for s in steps] == ['enter', 'preload', 'extend',
                                           'retract', 'recover']
        assert steps[0].contact == ALL_CONTACT
        assert steps[2].contact == LEG_MASKS[leg]
        assert steps[-1].contact == ALL_CONTACT
        assert all(s.duration_ms > 0 and len(s.foot_pose) == 6 for s in steps)
        # 默认身体不动：pos_des 全为 0
        assert all(abs(v) < 1e-9 for s in steps for v in s.pos_des)
        print(f'{leg}: ' + ' -> '.join(
            f'{s.name}({s.duration_ms}ms,0b{s.contact:04b})' for s in steps))
    shifted = build_sequence(KickConfig(leg='fr', body_shift=0.03))
    assert [s.name for s in shifted] == ['enter', 'windup', 'preload',
                                         'extend', 'retract', 'recover']
    print('自检通过（无需 ROS）')


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='CyberDog 2 单腿踢球独立测试（默认 dry-run）')
    parser.add_argument('--leg', default='fr', choices=('fr', 'fl', 'rr', 'rl'))
    parser.add_argument('--motion-id', type=int, default=212,
                        choices=(211, 212, 213),
                        help='212 相对姿态（推荐）/211 绝对姿态/213 低刚度')
    parser.add_argument('--cmd-source', type=int, default=0,
                        choices=(0, 1, 2, 3, 4))
    parser.add_argument('--execute', action='store_true',
                        help='真正发运动指令；不加则只打印序列')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--scale', type=float, default=1.0,
                        help='动作时长整体缩放，>1 更慢')
    parser.add_argument('--dog-name',
                        default='mi_desktop_48_b0_2d_7b_03_fd')
    parser.add_argument('--service', default='')
    parser.add_argument('--status-topic', default='')
    parser.add_argument('--lift', type=float, default=0.10,
                        help='抬腿高度（屈膝收腿时足端抬高）')
    parser.add_argument('--strike', type=float, default=0.10,
                        help='伸腿击球时足端高度（球心约 0.12）')
    parser.add_argument('--reach', type=float, default=0.18,
                        help='小腿前伸距离（踢不到球先加这个）')
    parser.add_argument('--back', type=float, default=-0.06,
                        help='收腿时足端后收量（负=向后）')
    parser.add_argument('--side', type=float, default=0.0)
    parser.add_argument('--shift', type=float, default=0.0,
                        help='质心侧向偏移；默认 0=身体不动，>0 才做移重心')
    parser.add_argument('--body-height', type=float, default=0.22,
                        help='仅 motion_id=211 绝对姿态时使用')
    parser.add_argument('--pitch', type=float, default=0.0)
    parser.add_argument('--transition-ms', type=int, default=600)
    parser.add_argument('--enter-ms', type=int, default=400)
    parser.add_argument('--pre-delay', type=float, default=1.0)
    parser.add_argument('--stand-timeout', type=float, default=25.0)
    parser.add_argument('--stand-settle', type=float, default=2.0,
                        help='站稳后再等多久才开始踢（秒）')
    parser.add_argument('--no-stand', action='store_true',
                        help='不自动请求 111 站立')
    parser.add_argument('--no-final-stand', action='store_true',
                        help='踢完不回到 111 站立')
    parser.add_argument('--allow-charging', action='store_true',
                        help='充电状态(switch=14)也尝试（不建议）')
    parser.add_argument('--self-test', action='store_true',
                        help='只做序列自检，不需要 ROS')
    return parser.parse_args(argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--ros-args' in argv:
        split = argv.index('--ros-args')
        own, ros_args = argv[:split], argv[split:]
    else:
        own, ros_args = argv, []
    args = parse_args(own)

    cfg = KickConfig(
        leg=args.leg, motion_id=args.motion_id, cmd_source=args.cmd_source,
        lift_height=args.lift, strike_height=args.strike,
        forward_reach=args.reach, back_offset=args.back,
        side_offset=args.side, body_shift=args.shift,
        body_height=args.body_height, body_pitch=args.pitch,
        transition_ms=args.transition_ms, enter_ms=args.enter_ms,
        phase_scale=args.scale)
    args.dry_run = not args.execute

    if args.self_test:
        self_test()
        return 0
    print_plan(cfg)
    if args.dry_run:
        print('当前是 dry-run：不会发送任何运动指令（也不需要 ROS）。'
              '\n确认序列后加 --execute 真机执行。\n')
        return 0

    # 真机执行路径才需要 ROS；dry-run / --self-test 在上面已经 return
    import rclpy
    stopping = {'value': False}

    def on_signal(*_):
        stopping['value'] = True
        print('\n收到 Ctrl+C，准备退出并恢复站姿…')

    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, on_signal)

    rclpy.init(args=ros_args or None)
    node = None
    try:
        node = make_node(args, cfg)
        if not node.client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('找不到运动服务；检查 dog_name/是否已 source 环境')
        if not node.wait_until(lambda: node.status is not None, 5.0, '运动状态'):
            raise RuntimeError('没有收到 motion_status')
        node.get_logger().info('状态：' + node.describe())
        if node.status.switch_status == 14 and not args.allow_charging:
            raise RuntimeError(
                '机器狗正在充电（switch_status=14 CHARGING），运动接口会受限，'
                '请拔掉充电器/下桩后重试；坚持尝试可加 --allow-charging')
        if not node.healthy():
            node.get_logger().warning(
                '当前状态不算健康（倾角/足端/电机/切换状态），'
                '建议先人工确认；仍将继续尝试')
        if not args.no_stand and not node.standing():
            node.get_logger().info('未处于站立，发送 111 恢复站立…')
            node.stand()
        else:
            node.get_logger().info('已处于站立，等待稳定…')
            node.spin_for(args.stand_settle)
        node.spin_for(args.pre_delay)
        for index in range(max(1, args.repeat)):
            if stopping['value'] or not rclpy.ok():
                break
            node.get_logger().info(
                f'--- 第 {index + 1}/{args.repeat} 次踢腿（{cfg.leg}） ---')
            node.kick_once()
            node.spin_for(0.4)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'执行失败：{exc}')
        if node is not None:
            node.get_logger().error(str(exc))
    finally:
        if node is not None:
            try:
                node.recover(not args.no_final_stand)
            except Exception as exc:
                print(f'恢复失败：{exc}')
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0


if __name__ == '__main__':
    sys.exit(main())
