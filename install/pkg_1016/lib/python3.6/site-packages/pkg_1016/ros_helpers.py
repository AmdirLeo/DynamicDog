"""两个作业节点共用的参数、QoS 与退出处理。"""

import signal
import threading

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy


def startup_parameter(node, name, default):
    """声明仅在启动时配置的参数，避免运行时更新但算法未更新。"""
    descriptor = ParameterDescriptor(read_only=True)
    return node.declare_parameter(name, default, descriptor).value


def latest_sensor_qos():
    """兼容 best-effort/reliable 传感器发布者，只保留最新一帧。"""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST, depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def run_node(factory, args=None):
    """先处理节点退出帧，再关闭 ROS context。"""
    rclpy.init(args=args)
    stop_requested = threading.Event()
    previous = {}
    node = None

    def request_stop(signum, frame):
        stop_requested.set()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, request_stop)
        node = factory()
        while rclpy.ok() and not stop_requested.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None and rclpy.ok():
                node.stop()
        finally:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            for signum, handler in previous.items():
                signal.signal(signum, handler)
