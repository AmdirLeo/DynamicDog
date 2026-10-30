"""课件第 32 页：通过服务请求，让机器狗执行一次太空步。"""

import rclpy
from rclpy.node import Node
from protocol.srv import MotionResultCmd


class BasicMove(Node):
    """调用机器狗已有的运动服务。"""

    def __init__(self):
        super().__init__('service_motion')
        # 使用现有项目中的机器狗名称，可用 --ros-args -p dog_name:=名称 修改。
        self.dog_name = self.declare_parameter(
            'dog_name', 'mi_desktop_48_b0_2d_7b_03_fd'
        ).value.strip('/')
        self.service_name = f'/{self.dog_name}/motion_result_cmd'
        self.client = self.create_client(MotionResultCmd, self.service_name)

    def send_request(self, motion_id):
        """异步发送请求，返回用于取得服务响应的 Future。"""
        request = MotionResultCmd.Request()
        request.motion_id = motion_id
        return self.client.call_async(request)


def main(args=None):
    """等待服务上线，发送一次跳舞请求，收到响应后退出。"""
    rclpy.init(args=args)
    node = None
    try:
        node = BasicMove()

        # 服务端在机器狗上运行；客户端每秒检查一次是否可以连接。
        while rclpy.ok():
            if node.client.wait_for_service(timeout_sec=1.0):
                break
            node.get_logger().info(f'等待服务：{node.service_name}')
        if not rclpy.ok():
            return

        future = node.send_request(152)  # 课件中的动作编号：152 = 太空步。
        node.get_logger().info('已发送太空步请求（motion_id=152），等待响应…')

        # call_async 不会阻塞；spin 处理通信，直到 Future 收到响应。
        rclpy.spin_until_future_complete(node, future)
        if not future.done() or future.cancelled():
            node.get_logger().warning('等待已结束，尚未收到服务响应。')
            return

        try:
            response = future.result()
        except Exception as exc:
            node.get_logger().error(f'服务调用失败：{exc}')
            return

        # 打印实际响应，供课堂观察结果；收到响应并不代表动作一定成功。
        node.get_logger().info(f'服务响应：{response}')
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
