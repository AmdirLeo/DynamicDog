"""通过 FaceEntry 服务录入人脸，通过 FaceEntryResult 话题等待结果。"""

from protocol.msg import FaceEntryResult
from protocol.srv import FaceEntry

from .face_common import DemoNode, run_demo


class FaceEnroll(DemoNode):
    """独立录入演示，支持查询名单及按名字删除旧记录。"""

    def __init__(self):
        super().__init__('face_enroll')
        self.username = self.declare_parameter('username', '').value.strip()
        self.action = self.declare_parameter('action', 'enroll').value
        self.enroll_timeout = self.positive_parameter('enroll_timeout', 120.0)
        if self.action not in ('enroll', 'list', 'delete'):
            raise ValueError('action 只能是 enroll、list 或 delete')
        if self.action != 'list' and not self.username:
            raise ValueError('请用 --ros-args -p username:=leo 指定名字')
        self.client = self.create_client(
            FaceEntry, self.endpoint('cyberdog_face_entry_srv'))
        self.result = None
        self.listening = False
        self.pending = False
        self.subscription = self.create_subscription(
            FaceEntryResult, self.endpoint('face_entry_msg'),
            self.on_result, 10)

    def request(self, command):
        """为每次调用新建请求，演示账号默认不设置为主人。"""
        request = FaceEntry.Request()
        request.command = command
        request.username = self.username
        request.ishost = False
        return request

    def on_result(self, message):
        """只保留当前录入的首个终态结果，不在回调中调用服务。"""
        if not self.listening or self.result is not None:
            return
        # 官方服务端的失败消息可能不带 username；成功消息必须匹配。
        if message.result == 0 and message.username != self.username:
            return
        if message.username and message.username != self.username:
            return
        self.result = message

    def run(self):
        """发起录入后等待话题；服务响应成功仅表示请求被接受。"""
        self.wait_service(self.client)
        commands = {'enroll': 0, 'list': 5, 'delete': 4}
        if self.action == 'enroll':
            self.get_logger().info(
                f'开始录入 {self.username}：请坐在摄像头前，按语音提示调整姿势。')
            self.listening = True
            self.pending = True
        response = self.call(self.client, self.request(commands[self.action]))
        if response.result != 0:
            self.pending = False
            raise RuntimeError(f'{self.action} 请求失败，result={response.result}')
        if self.action == 'list':
            self.get_logger().info(f'已录入名单：{response.allfaces}')
            return
        if self.action == 'delete':
            self.get_logger().info(f'已删除人脸：{self.username}')
            return

        self.get_logger().info('录入请求已接受，等待采集与保存结果…')
        self.wait_until(lambda: self.result is not None, self.enroll_timeout)
        self.listening = False
        self.pending = False
        if self.result.result == 5921:
            raise RuntimeError('这张脸已经录入；请查询名单后再决定是否删除旧记录。')
        if self.result.result != 0:
            raise RuntimeError(f'录入未完成，result={self.result.result}')
        # 官方 cyberdog_face 服务端在发布成功话题前已自动 CONFIRM_LAST_FACE。
        # 不能重复发 command=2，否则可能确认错误的“上一次”采集。
        self.get_logger().info(f'录入成功：{self.result.username}。现在可以单独演示识别。')

    def cleanup(self):
        """未取得终态结果时尝试取消本次录入。"""
        self.listening = False
        if self.pending:
            response = self.call(self.client, self.request(1), cleanup=True)
            if response.result != 0:
                raise RuntimeError(f'取消录入失败，result={response.result}')
            self.pending = False
            self.get_logger().info('已取消录入。')


def main(args=None):
    """ROS 2 命令行入口。"""
    run_demo(FaceEnroll, args)


if __name__ == '__main__':
    main()
