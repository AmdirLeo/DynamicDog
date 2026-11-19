#!/usr/bin/env python3
"""用快走步态（motion_id=305）的摆动腿踢球——单文件真机测试脚本。

原理：
    305 是快速对角小跑，摆动腿在腾空相向前摆。脚本在检测到目标腿刚离地
    （motion_status.contact 对应位 1→0）时，发一段短促的"前进速度 + 抬腿
    高度"脉冲（step_height 前 0.06 m），让摆动腿在球的位置扫过去，把球踢出
    去；脉冲结束立即回落，避免一直用身体推球。

不依赖 pkg_soccer，只依赖机器狗自带 rclpy + protocol：
    publish  /<dog_name>/motion_servo_cmd   (protocol/msg/MotionServoCmd)
    subscribe/<dog_name>/motion_status      (protocol/msg/MotionStatus)
    service  /<dog_name>/motion_result_cmd  (protocol/srv/MotionResultCmd，111 站立)

用法（NX 板上）：
    python3 gait_kick_action.py                    # dry-run：只打印脉冲计划
    python3 gait_kick_action.py --execute          # 真机：走几步后踢一次
    python3 gait_kick_action.py --execute --sync-leg fr --speed 0.75 --burst-ms 200
    python3 gait_kick_action.py --execute --repeat 3 --gap-ms 500
    python3 gait_kick_action.py --push --execute   # 全力推球：1.5 m/s × 450ms，无腿部动作

注意：这个动作会让狗**持续向前走 0.2~0.4 m**，第一次务必清空前方、
      用 --speed 0.4 --burst-ms 120 低速试，并确认 sync-leg 方向对。

参数：
    --speed 0.75        脉冲时的前进速度（m/s）
    --burst-ms 200      脉冲持续时间
    --step-front/back   脉冲时的前后抬腿高度（默认 0.06/0.05）
    --sync-leg fr       用哪条腿的离地相位触发；off=不检测相位，直接踢
    --warmup-ms 250     先以 --warmup-speed 走一小段，进入 305 步态
    --phase-timeout-ms  等不到相位时的兜底超时
    --repeat / --gap-ms 连踢次数与间隔
"""

import argparse
import math
import signal
import sys
import time

LEG_BITS = {'fr': 0, 'fl': 1, 'rr': 2, 'rl': 3}
ALL_CONTACT = 0x0F
MOTION_ID_WALK_FAST = 305
DT = 0.05


class GaitKickConfig:
    def __init__(self, burst_speed=0.75, burst_ms=200,
                 step_front=0.06, step_back=0.05, step_normal=0.05,
                 phase_timeout_ms=350, sync_leg='fr',
                 warmup_ms=250, warmup_speed=0.15, gap_ms=0):
        self.burst_speed = float(burst_speed)
        self.warmup_speed = float(warmup_speed)
        self.step_front = float(step_front)
        self.step_back = float(step_back)
        self.step_normal = float(step_normal)
        self.burst_ms = int(burst_ms)
        self.phase_timeout_ms = int(phase_timeout_ms)
        self.warmup_ms = int(warmup_ms)
        self.gap_ms = int(gap_ms)
        leg = str(sync_leg).lower()
        self.sync_leg = leg if leg in LEG_BITS else None
        checks = (('burst_speed', self.burst_speed, 0.05, 1.5),
                  ('warmup_speed', self.warmup_speed, 0.0, 0.6),
                  ('step_front', self.step_front, 0.0, 0.08),
                  ('step_back', self.step_back, 0.0, 0.08),
                  ('step_normal', self.step_normal, 0.0, 0.08))
        for name, value, low, high in checks:
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{name} 必须在 [{low}, {high}]，实得 {value}')
        for name, value, low, high in (
                ('burst_ms', self.burst_ms, 40, 600),
                ('phase_timeout_ms', self.phase_timeout_ms, 50, 1200),
                ('warmup_ms', self.warmup_ms, 0, 1000),
                ('gap_ms', self.gap_ms, 0, 2000)):
            if not low <= value <= high:
                raise ValueError(f'{name} 必须在 [{low}, {high}] ms，实得 {value}')


class SwingDetector:
    """从 contact 低 4 位检测目标腿离地（1→0）。"""

    def __init__(self, leg):
        self.bit = LEG_BITS.get(leg) if leg else None
        self.last = None
        self.edge_at = None
        self.period = None

    def update(self, contact, now):
        if contact is None or self.bit is None:
            return False
        bit = (int(contact) >> self.bit) & 1
        edge = self.last == 1 and bit == 0
        self.last = bit
        if edge:
            self.edge_at = now
        return edge


class GaitKickPlan:
    """一次踢球的状态机（与 pkg_soccer/gait_kick.py 保持一致）。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.detector = SwingDetector(cfg.sync_leg) if cfg.sync_leg else None
        self.state = 'idle'
        self.started = None
        self.wait_started = None
        self.until = None
        self.base_speed = 0.0
        self.detected_phase = None

    def start(self, now, velocity):
        vx, vy = float(velocity[0]), float(velocity[1])
        self.base_speed = math.hypot(vx, vy)
        self.started = now
        self.detected_phase = None
        if self.cfg.warmup_ms > 0:
            self.state = 'warmup'
            self.until = now + self.cfg.warmup_ms / 1000.0
        else:
            self.state = 'waiting'
            self.wait_started = now

    def update(self, now, contact):
        if self.state == 'idle':
            return None
        if self.state == 'warmup':
            if now >= self.until:
                self.state = 'waiting'
                self.wait_started = now
            return ('warmup', self.cfg.warmup_speed,
                    (self.cfg.step_normal, self.cfg.step_normal))
        if self.state == 'waiting':
            if self.detector is None:
                self.state = 'burst'
                self.until = now + self.cfg.burst_ms / 1000.0
                return ('burst', self.cfg.burst_speed,
                        (self.cfg.step_front, self.cfg.step_back))
            edge = self.detector.update(contact, now) if self.detector else False
            if edge:
                self.detected_phase = now - self.started
            if edge or now - self.wait_started >= self.cfg.phase_timeout_ms / 1000.0:
                self.state = 'burst'
                self.until = now + self.cfg.burst_ms / 1000.0
            return ('waiting', self.base_speed,
                    (self.cfg.step_normal, self.cfg.step_normal))
        if self.state == 'burst':
            if now >= self.until:
                self.state = 'gap' if self.cfg.gap_ms > 0 else 'done'
                self.until = now + self.cfg.gap_ms / 1000.0
            return ('burst', self.cfg.burst_speed,
                    (self.cfg.step_front, self.cfg.step_back))
        if self.state == 'gap':
            if now >= self.until:
                self.state = 'done'
            return ('gap', self.cfg.warmup_speed,
                    (self.cfg.step_normal, self.cfg.step_normal))
        return None

    @property
    def done(self):
        return self.state == 'done'


# ---------------------------------------------------------------- 计划打印
def print_plan(cfg, repeat):
    print('=== 快走(305)踢球计划 ===')
    print(f'速度脉冲={cfg.burst_speed:.2f} m/s × {cfg.burst_ms} ms；'
          f'抬腿高度 前{cfg.step_front:.2f}/后{cfg.step_back:.2f} m')
    if cfg.sync_leg:
        wait_text = f'等摆动腿离地（最长 {cfg.phase_timeout_ms} ms）'
        wait_distance = 0.2 * cfg.phase_timeout_ms / 1000.0
    else:
        wait_text = '不检测相位，预热后直接脉冲'
        wait_distance = 0.0
    print(f'相位同步={cfg.sync_leg or "关闭"}；预热 {cfg.warmup_ms} ms @ '
          f'{cfg.warmup_speed:.2f} m/s')
    if repeat > 1:
        print(f'连踢 {repeat} 次，间隔 {cfg.gap_ms} ms')
    distance = (cfg.warmup_speed * cfg.warmup_ms / 1000.0 +
                wait_distance + cfg.burst_speed * cfg.burst_ms / 1000.0)
    print('流程：')
    print(f'  1) 起走 305：{cfg.warmup_ms} ms 低速前进，进入快走步态')
    print(f'  2) {wait_text}')
    print(f'  3) 脉冲：{cfg.burst_ms} ms 内前进速度 {cfg.burst_speed:.2f} m/s，'
          f'步高 {cfg.step_front:.2f}/{cfg.step_back:.2f} m')
    print(f'  → 预计前进约 {distance:.2f} m（请留出空间）')
    print()
    return distance


# ---------------------------------------------------------------- ROS 部分
def make_node(args, cfg):
    import rclpy
    from protocol.msg import MotionServoCmd, MotionStatus
    from protocol.srv import MotionResultCmd
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    prefix = ('/' + args.dog_name.strip('/')) if args.dog_name else ''
    cmd_topic = args.cmd_topic or f'{prefix}/motion_servo_cmd'
    status_topic = args.status_topic or f'{prefix}/motion_status'

    class GaitKickNode(Node):
        def __init__(self):
            super().__init__('gait_kick_action')
            self.status = None
            self.qos = QoSProfile(depth=1,
                                  reliability=ReliabilityPolicy.BEST_EFFORT)
            self.publisher = self.create_publisher(MotionServoCmd, cmd_topic, 1)
            self.create_subscription(MotionStatus, status_topic,
                                     self.on_status, self.qos)
            self.service_type = MotionResultCmd
            self.client = self.create_client(
                MotionResultCmd, args.service or f'{prefix}/motion_result_cmd')
            self.get_logger().info(
                f'指令={cmd_topic}  状态={status_topic}  速度={cfg.burst_speed:.2f}')

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
            self.get_logger().error(f'等待{what}超时')
            return False

        def describe(self):
            s = self.status
            if s is None:
                return '无状态'
            return (f'motion_id={s.motion_id} switch={s.switch_status} '
                    f'contact=0b{int(s.contact) & 0x0F:04b} '
                    f'progress={s.order_process_bar}')

        def publish(self, velocity, step_height, end=False):
            from protocol.msg import MotionServoCmd
            msg = MotionServoCmd()
            msg.motion_id = MOTION_ID_WALK_FAST
            msg.cmd_type = MotionServoCmd.SERVO_END if end else MotionServoCmd.SERVO_DATA
            msg.cmd_source = 0
            msg.value = 2
            msg.vel_des = [float(velocity[0]), float(velocity[1]),
                           float(velocity[2])]
            msg.step_height = [float(step_height[0]), float(step_height[1])]
            self.publisher.publish(msg)

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
            future = self.client.call_async(request)
            end = time.monotonic() + 3.0
            while rclpy.ok() and not future.done() and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.01)
            if not future.done():
                raise TimeoutError('站立服务超时')
            response = future.result()
            self.get_logger().info(f'111 站立：result={response.result} '
                                   f'code={response.code}')

        def stop_servo(self):
            """退出/结束：连发 END 帧，确保步态伺服停止。"""
            for _ in range(3):
                self.publish((0.0, 0.0, 0.0), (cfg.step_normal, cfg.step_normal),
                             end=True)
                self.spin_for(0.03)

        def run_burst(self, plan):
            """执行一次：预热 → 等相位 → 脉冲，返回是否成功。"""
            plan.start(time.monotonic(), (cfg.warmup_speed, 0.0, 0.0))
            last_phase = None
            next_tick = time.monotonic()
            while rclpy.ok():
                now = time.monotonic()
                contact = None if self.status is None else self.status.contact
                step = plan.update(now, contact)
                if step is None:
                    break
                phase, speed, heights = step
                if phase != last_phase:
                    self.get_logger().info(
                        f'阶段 {phase} @ t={now - plan.started:.2f}s；'
                        f'{self.describe()}')
                    last_phase = phase
                self.publish((speed, 0.0, 0.0), heights)
                next_tick += DT
                delay = next_tick - time.monotonic()
                if delay > 0:
                    rclpy.spin_once(self, timeout_sec=delay)
                    # spin_once 可能被回调提前唤醒，补足剩余时间保证 20 Hz
                    rest = next_tick - time.monotonic()
                    if rest > 0:
                        time.sleep(rest)
                else:
                    next_tick = time.monotonic()
            self.get_logger().info(
                f'本次踢球结束：检测到相位={plan.detected_phase}；'
                f'{self.describe()}')
            return True

    return GaitKickNode()


# ---------------------------------------------------------------- 入口
def self_test():
    cfg = GaitKickConfig(warmup_ms=0)
    plan = GaitKickPlan(cfg)
    plan.start(0.0, (0.2, 0.0, 0.0))
    contacts = [0x0F] * 3 + [0x0E] * 4 + [0x0F] * 3
    phases = []
    for tick in range(60):
        step = plan.update(tick * 0.02, contacts[min(tick, len(contacts) - 1)])
        if step is None:
            break
        phases.append(step[0])
    print('阶段序列：', phases)
    assert 'burst' in phases
    print('自检通过（无需 ROS）')


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='用快走(305)摆动腿踢球（默认 dry-run）')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dog-name', default='mi_desktop_48_b0_2d_7b_03_fd')
    parser.add_argument('--cmd-topic', default='')
    parser.add_argument('--status-topic', default='')
    parser.add_argument('--service', default='')
    parser.add_argument('--speed', type=float, default=None,
                        help='脉冲前进速度 m/s；默认 0.75，--push 时 1.5')
    parser.add_argument('--burst-ms', type=int, default=None,
                        help='脉冲时长 ms；默认 200，--push 时 450')
    parser.add_argument('--step-front', type=float, default=None,
                        help='前腿抬腿高度 m；默认 0.06，--push 时 0.05')
    parser.add_argument('--step-back', type=float, default=None,
                        help='后腿抬腿高度 m；默认 0.05')
    parser.add_argument('--step-normal', type=float, default=None)
    parser.add_argument('--sync-leg', default=None,
                        choices=('fr', 'fl', 'rr', 'rl', 'off'),
                        help='相位同步腿；默认 fr，--push 时 off')
    parser.add_argument('--phase-timeout-ms', type=int, default=350)
    parser.add_argument('--warmup-ms', type=int, default=None,
                        help='预热时长 ms；默认 250')
    parser.add_argument('--warmup-speed', type=float, default=0.15)
    parser.add_argument('--push', action='store_true',
                        help='一键全力推球：1.5 m/s × 450 ms，不检测相位、步高不变')
    parser.add_argument('--gap-ms', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--pre-delay', type=float, default=1.0)
    parser.add_argument('--no-stand', action='store_true')
    parser.add_argument('--no-final-stand', action='store_true')
    parser.add_argument('--allow-charging', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--ros-args' in argv:
        split = argv.index('--ros-args')
        own, ros_args = argv[:split], argv[split:]
    else:
        own, ros_args = argv, []
    args = parse_args(own)
    # --push 只改“未显式指定”的默认值，用户手写的参数仍然优先
    if args.speed is None:
        args.speed = 1.5 if args.push else 0.75
    if args.burst_ms is None:
        args.burst_ms = 450 if args.push else 200
    if args.step_front is None:
        args.step_front = 0.05 if args.push else 0.06
    if args.step_back is None:
        args.step_back = 0.05
    if args.step_normal is None:
        args.step_normal = 0.05
    if args.sync_leg is None:
        args.sync_leg = 'off' if args.push else 'fr'
    if args.warmup_ms is None:
        args.warmup_ms = 250
    sync_leg = None if args.sync_leg == 'off' else args.sync_leg
    cfg = GaitKickConfig(
        burst_speed=args.speed, burst_ms=args.burst_ms,
        step_front=args.step_front, step_back=args.step_back,
        step_normal=args.step_normal, phase_timeout_ms=args.phase_timeout_ms,
        sync_leg=sync_leg, warmup_ms=args.warmup_ms,
        warmup_speed=args.warmup_speed, gap_ms=args.gap_ms)
    args.dry_run = not args.execute

    if args.self_test:
        self_test()
        return 0
    distance = print_plan(cfg, max(1, args.repeat))
    if args.dry_run:
        print('当前是 dry-run：不会发送任何运动指令（也不需要 ROS）。')
        print(f'确认后加 --execute 真机执行（狗会向前走约 {distance:.2f} m）。\n')
        return 0

    import rclpy
    stopping = {'value': False}

    def on_signal(*_):
        stopping['value'] = True
        print('\n收到 Ctrl+C，停止步态并站立…')

    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, on_signal)

    rclpy.init(args=ros_args or None)
    node = None
    try:
        node = make_node(args, cfg)
        if not node.client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('找不到站立服务；检查 dog_name/环境')
        if not node.wait_until(lambda: node.status is not None, 5.0, '运动状态'):
            raise RuntimeError('没有收到 motion_status')
        node.get_logger().info('状态：' + node.describe())
        if node.status.switch_status == 14 and not args.allow_charging:
            raise RuntimeError('机器狗正在充电（switch_status=14），请下桩后重试')
        if not args.no_stand:
            node.stand()
        node.spin_for(args.pre_delay)
        for index in range(max(1, args.repeat)):
            if stopping['value'] or not rclpy.ok():
                break
            node.get_logger().info(f'--- 第 {index + 1}/{args.repeat} 次 ---')
            node.run_burst(GaitKickPlan(cfg))
            node.spin_for(0.2)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'执行失败：{exc}')
        if node is not None:
            node.get_logger().error(str(exc))
    finally:
        if node is not None:
            try:
                node.stop_servo()
                if not args.no_final_stand:
                    node.spin_for(0.2)
                    node.stand()
            except Exception as exc:
                print(f'收尾失败：{exc}')
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0


if __name__ == '__main__':
    sys.exit(main())
