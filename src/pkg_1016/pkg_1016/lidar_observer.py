"""订阅激光雷达 scan，输出四个方向的最近障碍距离和近距离提示。"""

import json
import math
import time

from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from pkg_1016.ros_helpers import latest_sensor_qos, run_node, startup_parameter
from pkg_1016.topic_logic import analyse_scan


class LidarObserver(Node):
    """只读传感器功能，不向机器狗发送运动命令。"""

    def __init__(self):
        """发布 JSON 状态，缺测用 null 表示，不把缺测当作空旷。"""
        super().__init__('lidar_observer')
        self.warning_distance = startup_parameter(
            self, 'warning_distance', 0.40)
        self.scan_timeout = startup_parameter(self, 'scan_timeout', 1.0)
        if any(not math.isfinite(v) or v <= 0 for v in (
                self.warning_distance, self.scan_timeout)):
            raise ValueError('warning_distance 和 scan_timeout 必须是有限正数')
        self.publisher = self.create_publisher(
            String, 'homework/lidar_obstacles', 10)
        self.subscription = self.create_subscription(
            LaserScan, 'scan', self.on_scan, latest_sensor_qos())
        self.last_received = None
        self.frame_id = ''
        self.result = None
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(
            0.5, self.on_timer, clock=self.steady_clock)
        self.get_logger().info('激光扇区方向相对于 scan.header.frame_id；输出频率 2 Hz')

    def on_scan(self, msg):
        """按 angle_min + i * angle_increment 还原每条射线角度。"""
        self.result = analyse_scan(
            msg.ranges, msg.angle_min, msg.angle_increment,
            msg.range_min, msg.range_max, self.warning_distance)
        self.frame_id = msg.header.frame_id
        self.last_received = time.monotonic()

    def on_timer(self):
        """数据超时时清空旧距离，防止把历史观测当成当前结果。"""
        now = time.monotonic()
        age = None if self.last_received is None else now - self.last_received
        if age is None or age > self.scan_timeout:
            report = analyse_scan([], 0.0, 1.0, 0.0, 1.0)
            report['status'] = 'WAIT_SCAN' if age is None else 'STALE'
        else:
            report = dict(self.result)
        report.update({
            'frame_id': self.frame_id,
            'receive_age_s': None if age is None else round(age, 3),
            'warning_distance_m': self.warning_distance,
        })
        text = json.dumps(report, ensure_ascii=False, allow_nan=False)
        msg = String()
        msg.data = text
        self.publisher.publish(msg)
        if any(s['near'] is True for s in report['sectors'].values()):
            self.get_logger().warning(text)
        else:
            self.get_logger().info(text)

    def stop(self):
        """停止只读节点的报告定时器。"""
        self.timer.cancel()


def main(args=None):
    """ROS 2 命令行入口。"""
    run_node(LidarObserver, args)
