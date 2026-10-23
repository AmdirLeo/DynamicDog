"""订阅 ultrasonic_payload，用 MotionServoCmd 实现低速反应式避障。"""

import json
import math
import time
from dataclasses import fields

from protocol.msg import MotionServoCmd
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import String

from pkg_1016.ros_helpers import latest_sensor_qos, run_node, startup_parameter
from pkg_1016.topic_logic import AvoidanceConfig, AvoidanceController


class UltrasonicAvoidance(Node):
    """默认只观测；enable_motion:=true 才创建运动发布者。"""

    def __init__(self):
        """使用相对话题名，机器狗命名空间由启动命令指定。"""
        super().__init__('ultrasonic_avoidance')
        self.enable_motion = startup_parameter(self, 'enable_motion', False)
        self.motion_id = startup_parameter(self, 'motion_id', 303)
        self.motion_value = startup_parameter(self, 'motion_value', 2)
        self.cmd_source = startup_parameter(self, 'cmd_source', 0)
        self.step_height = startup_parameter(self, 'step_height', 0.05)
        defaults = AvoidanceConfig()
        values = {
            f.name: startup_parameter(self, f.name, getattr(defaults, f.name))
            for f in fields(defaults)
        }
        self.controller = AvoidanceController(AvoidanceConfig(**values))
        if (not math.isfinite(self.step_height)
                or not 0 < self.step_height <= 0.10):
            raise ValueError('需要 0 < step_height <= 0.10 m')
        if self.cmd_source not in (0, 1, 2):
            raise ValueError('cmd_source 只能是 APP=0、AUDIO=1、VIS=2')
        self.motion_pub = None
        if self.enable_motion:
            self.motion_pub = self.create_publisher(
                MotionServoCmd, 'motion_servo_cmd', 10)
        self.status_pub = self.create_publisher(
            String, 'homework/avoidance_status', 10)
        self.subscription = self.create_subscription(
            Range, 'ultrasonic_payload', self.on_range, latest_sensor_qos())
        self.last_state = None
        self.last_report = float('-inf')
        self.frame_id = ''
        self.range_valid = False
        # 使用稳态时钟，即使 ROS 仿真时钟暂停也继续检查接收超时。
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(
            0.1, self.on_timer, clock=self.steady_clock)
        mode = '实机运动已启用' if self.enable_motion else '只观测，不发布运动命令'
        self.get_logger().info(
            f'{mode}；namespace={self.get_namespace()}；控制周期=0.1s')

    def on_range(self, msg):
        """回调只更新数据，决策和命令发布由固定周期定时器完成。"""
        self.frame_id = msg.header.frame_id
        # 本节点只用于超声波连续测距，不接受红外或二值测距消息。
        distance = (msg.range if msg.radiation_type == Range.ULTRASOUND
                    else math.nan)
        self.range_valid = self.controller.update_range(
            distance, msg.min_range, msg.max_range, time.monotonic())

    def publish_motion(self, vx, wz, end=False):
        """沿用课程示例的步态字段，使用接口定义的 DATA/END 常量。"""
        if self.motion_pub is None:
            return
        msg = MotionServoCmd()
        msg.motion_id = self.motion_id
        msg.cmd_type = (MotionServoCmd.SERVO_END if end
                        else MotionServoCmd.SERVO_DATA)
        msg.cmd_source = self.cmd_source
        msg.value = self.motion_value
        msg.vel_des = [float(vx), 0.0, float(wz)]
        msg.step_height = [self.step_height, self.step_height]
        self.motion_pub.publish(msg)

    def on_timer(self):
        """以 10 Hz 发送命令，以 2 Hz 或状态变化时报告决策。"""
        now = time.monotonic()
        vx, wz = self.controller.command(now)
        self.publish_motion(vx, wz)
        state = self.controller.state
        changed = state != self.last_state
        if changed or now - self.last_report >= 0.5:
            last = self.controller.last_received
            status = {
                'state': state,
                'enable_motion': self.enable_motion,
                'frame_id': self.frame_id,
                'distance_m': self.controller.distance,
                'last_sample_valid': self.range_valid,
                'receive_age_s': (
                    None if last is None else round(now - last, 3)),
                'clear_count': self.controller.clear_count,
                'desired_vx_m_s': vx,
                'desired_wz_rad_s': wz,
                'motion_subscribers': (
                    self.motion_pub.get_subscription_count()
                    if self.motion_pub is not None else None),
            }
            text = json.dumps(status, ensure_ascii=False, allow_nan=False)
            msg = String()
            msg.data = text
            self.status_pub.publish(msg)
            self.get_logger().info(text)
            self.last_state = state
            self.last_report = now

    def stop(self):
        """正常退出时发送零速度和结束帧；不能保证断电/强杀时送达。"""
        self.timer.cancel()
        if self.motion_pub is None:
            return
        for _ in range(3):
            self.publish_motion(0.0, 0.0)
            time.sleep(0.05)
        self.publish_motion(0.0, 0.0, end=True)
        time.sleep(0.10)


def main(args=None):
    """ROS 2 命令行入口。"""
    run_node(UltrasonicAvoidance, args)
