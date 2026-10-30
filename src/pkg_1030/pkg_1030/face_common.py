"""人脸演示共用的服务等待、参数和退出处理。"""

import math
import signal
import time

import rclpy
from rclpy.node import Node


class DemoNode(Node):
    """在主流程中等待服务，让订阅回调只负责接收结果。"""

    def __init__(self, name):
        super().__init__(name)
        self.stop_requested = False
        self.dog_name = self.declare_parameter(
            'dog_name', 'mi_desktop_48_b0_2d_7b_03_fd'
        ).value.strip('/')
        self.service_timeout = self.positive_parameter('service_timeout', 120.0)

    def endpoint(self, name):
        """生成机器狗的接口路径，也允许 dog_name 为空。"""
        prefix = f'/{self.dog_name}' if self.dog_name else ''
        return f'{prefix}/{name}'

    def positive_parameter(self, name, default):
        """读取正数时间参数，拒绝无穷值和非正数。"""
        value = float(self.declare_parameter(name, default).value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} 必须是有限的正数')
        return value

    def wait_until(self, predicate, timeout, cleanup=False):
        """使用真实时间计时，并持续处理 ROS 回调。"""
        deadline = time.monotonic() + timeout
        while not predicate():
            if not rclpy.ok() or (self.stop_requested and not cleanup):
                raise InterruptedError('演示已中断')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('等待超时')
            rclpy.spin_once(self, timeout_sec=min(0.1, remaining))
        if self.stop_requested and not cleanup:
            raise InterruptedError('演示已中断')

    def wait_service(self, client):
        """最多等待十秒发现服务；实际响应可等待 service_timeout 秒。"""
        self.get_logger().info(f'等待服务：{client.srv_name}')
        self.wait_until(client.service_is_ready, 10.0)

    def call(self, client, request, cleanup=False):
        """异步发送请求，再在主流程中处理回调直至收到响应。"""
        if not rclpy.ok() or (self.stop_requested and not cleanup):
            raise InterruptedError('演示已中断')
        future = client.call_async(request)
        try:
            self.wait_until(
                future.done, 5.0 if cleanup else self.service_timeout, cleanup)
            if future.cancelled():
                raise RuntimeError('服务请求已取消')
            response = future.result()
            if response is None:
                raise RuntimeError('服务没有返回响应')
            return response
        finally:
            if not future.done():
                # 只取消本地等待；服务端任务须由节点的 cleanup 另行处理。
                future.cancel()

    def cleanup(self):
        """由子类实现取消任务或归还灯光控制。"""


def run_demo(factory, args=None):
    """Ctrl+C 时先清理服务端状态，再关闭 ROS，保留清理所需的通信。"""
    rclpy.init(args=args)
    node = None
    previous = {}
    exit_code = 0

    def request_stop(signum, frame):
        if node is None:
            raise KeyboardInterrupt
        node.stop_requested = True

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, request_stop)
        node = factory()
        node.run()
    except (InterruptedError, KeyboardInterrupt):
        exit_code = 130
        if node is not None:
            node.get_logger().info('已中断，正在清理。')
    except Exception as exc:
        exit_code = 1
        if node is not None:
            node.get_logger().error(str(exc))
        else:
            print(f'启动失败：{exc}')
    finally:
        if node is not None:
            try:
                node.cleanup()
            except Exception as exc:
                node.get_logger().error(f'清理未确认成功：{exc}')
                exit_code = exit_code or 1
            finally:
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if exit_code:
        raise SystemExit(exit_code)
