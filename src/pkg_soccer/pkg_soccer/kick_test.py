"""单腿踢球测试节点：不依赖动捕/足球，站着踢一次或连续踢。

先 dry-run 打印序列，确认参数后再真发：

    # 只打印，不发运动指令（默认）
    ros2 run pkg_soccer leg_kick --ros-args -p kick_leg:=fr

    # 真机：先确保狗能自己站稳，或 auto_stand:=true 由本节点请求站立
    ros2 run pkg_soccer leg_kick --ros-args -p kick_leg:=fr -p dry_run:=false

建议流程：
1. 把狗放在软垫/悬空（腿能自由摆动），`dry_run:=true` 看时序；
2. `dry_run:=false`、`kick_phase_scale:=1.5`（慢速）先看动作；
3. 正常速度、垫一个球在右前方，微调 `kick_forward_reach / kick_lift_height / kick_strike_height`；
4. 到场地用 `kick_mode:=leg` 接入 `soccer` 节点（见 soccer.py 文档）。

安全：动作中随时 Ctrl+C；节点退出前会尽力发一条"四腿支撑归零"的恢复命令。
"""

import signal
import threading
import time

import rclpy
from protocol.msg import MotionStatus
from protocol.srv import MotionResultCmd
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from .leg_kick import KickConfig, KickRunner, build_request


class KickTest(Node):
    def __init__(self):
        super().__init__('leg_kick_test')
        self.dog_name = self.declare_parameter(
            'dog_name', 'mi_desktop_48_b0_2d_7b_03_fd').value.strip('/')
        self.dry_run = bool(self.param('dry_run', True))
        self.auto_stand = bool(self.param('auto_stand', True))
        self.repeat = int(self.param('repeat', 1))
        self.pre_delay = float(self.param('pre_delay', 1.0))
        self.stand_timeout = float(self.param('stand_timeout', 25.0))
        self.stand_settle = float(self.param('stand_settle', 2.0))
        self.allow_charging = bool(self.param('allow_charging', False))
        self.cfg = KickConfig.from_params(self.param)
        self.runner = KickRunner(self.cfg)
        self.status = None
        self.status_count = 0
        self.qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.client = self.create_client(MotionResultCmd,
                                         self.endpoint('motion_result_cmd'))
        self.create_subscription(MotionStatus, self.endpoint('motion_status'),
                                 self.on_status, self.qos)
        self.get_logger().info(
            f'单腿踢球测试：leg={self.cfg.leg} motion_id={self.cfg.motion_id} '
            f'dry_run={self.dry_run} auto_stand={self.auto_stand} '
            f'repeat={self.repeat} phase_scale={self.cfg.phase_scale}')
        for step in self.runner.steps:
            self.get_logger().info('  ' + repr(step))

    def param(self, name, default):
        return self.declare_parameter(name, default).value

    def endpoint(self, name):
        return f'/{self.dog_name}/{name}' if self.dog_name else f'/{name}'

    def on_status(self, msg):
        self.status = msg
        self.status_count += 1

    def spin_for(self, duration):
        end = time.monotonic() + duration
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

    def call(self, request, timeout=2.0):
        future = self.client.call_async(request)
        end = time.monotonic() + timeout
        while rclpy.ok() and not future.done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)
        if not future.done():
            raise TimeoutError('motion_result_cmd 服务超时')
        return future.result()

    def _standing(self):
        status = self.status
        return (status is not None and
                (status.motion_id == 112 or
                 (status.motion_id == 111 and status.order_process_bar >= 96)))

    def _settled(self):
        status = self.status
        return (status is not None and status.switch_status == 0 and
                (status.motion_id == 112 or
                 (status.motion_id == 111 and status.order_process_bar >= 96)))

    def _healthy(self):
        status = self.status
        return (status is not None and status.ori_error == 0 and
                status.footpos_error == 0 and status.switch_status == 0)

    def stand(self):
        request = self.service_type.Request()
        request.motion_id = 111
        request.cmd_source = 0
        request.duration = 0
        request.contact = 0x0F      # 四腿支撑，避免默认 0 被当成"全部抬起"
        for name, size in (('vel_des', 3), ('rpy_des', 3), ('pos_des', 3),
                           ('acc_des', 6), ('ctrl_point', 3),
                           ('foot_pose', 6), ('step_height', 2)):
            setattr(request, name, [0.0] * size)
        response = self.call(request, timeout=3.0)
        self.get_logger().info(f'站立请求返回：result={response.result} '
                               f'motion_id={response.motion_id} code={response.code}')
        if not response.result:
            raise RuntimeError('站立请求失败，先人工确认狗的状态')
        self.wait_until(self._settled, self.stand_timeout, '站稳')
        # 站稳后再等一会儿，避免位控命令赶在运控切换/稳定过程中
        self.spin_for(self.stand_settle)

    def send_step(self, step):
        if self.dry_run:
            self.get_logger().info(f'[dry-run] {step!r}')
            return
        response = self.call(build_request(self.service_type, self.cfg, step))
        self.get_logger().info(f'{step.name}: result={response.result} '
                               f'motion_id={response.motion_id} code={response.code}')
        if not response.result:
            raise RuntimeError(f'{step.name} 被运控拒绝（code={response.code}）')

    def recover(self):
        """异常/退出兜底：四腿支撑、足端与质心归零。"""
        if self.dry_run or not rclpy.ok():
            return
        try:
            request = build_request(self.service_type, self.cfg,
                                    self.runner.recover_step())
            response = self.call(request, timeout=1.5)
            self.get_logger().info(f'恢复命令：result={response.result} '
                                   f'code={response.code}')
            self.spin_for(0.35)
        except Exception as exc:  # 退出路径尽力而为
            self.get_logger().error(f'恢复命令失败：{exc}')

    def run(self):
        self.service_type = MotionResultCmd
        if not self.client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('找不到 ' + self.endpoint('motion_result_cmd'))
        self.wait_until(lambda: self.status is not None, 5.0, '运动状态')
        if (self.status is not None and self.status.switch_status == 14
                and not self.allow_charging):
            raise RuntimeError(
                '机器狗正在充电（switch_status=14 CHARGING），运动接口受限；'
                '请拔掉充电器/下桩后重试，坚持尝试可加 -p allow_charging:=true')
        if not self._healthy():
            self.get_logger().warning(
                '运动状态不是正常站立，先检查狗；auto_stand 会尝试起身')
        if self.auto_stand and not self._standing():
            self.stand()
        self.spin_for(self.pre_delay)
        for index in range(max(1, self.repeat)):
            if not rclpy.ok():
                break
            self.get_logger().info(f'--- 第 {index + 1}/{self.repeat} 次踢腿 ---')
            self.runner.start(time.monotonic())
            while rclpy.ok():
                now = time.monotonic()
                step = self.runner.poll(now)
                if step is not None:
                    self.send_step(step)
                if self.runner.ready_to_end(now):
                    break
                rclpy.spin_once(self, timeout_sec=0.01)
            self.spin_for(0.3)
        self.recover()


def main(args=None):
    rclpy.init(args=args)
    node = None
    stopping = threading.Event()
    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, lambda *_: stopping.set())
        node = KickTest()
        while rclpy.ok() and not stopping.is_set():
            node.run()
            break
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'leg_kick 失败：{exc}')
    finally:
        if node is not None:
            try:
                node.recover()
            except Exception:
                pass
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    main()
