"""通过服务启动人脸识别，匹配成功话题后亮绿灯三秒。"""

import time
import uuid

from protocol.msg import FaceRecognitionResult
from protocol.srv import FaceRec, LedExecute

from .face_common import DemoNode, run_demo


class FaceRecognizeLed(DemoNode):
    """识别与 LED 联动演示；不发送机器狗运动指令。"""

    def __init__(self):
        super().__init__('face_recognize_led')
        self.username = self.declare_parameter('username', '').value.strip()
        self.timeout = self.declare_parameter('timeout', 60).value
        if not isinstance(self.timeout, int) or not 30 <= self.timeout <= 300:
            raise ValueError('timeout 必须是 30～300 秒的整数')
        self.led_seconds = self.positive_parameter('led_seconds', 3.0)
        self.request_id = uuid.uuid4().hex
        self.face_client = self.create_client(
            FaceRec, self.endpoint('cyberdog_face_recognition_srv'))
        self.led_client = self.create_client(
            LedExecute, self.endpoint('led_execute'))
        self.result = None
        self.listening = False
        self.recognition_pending = False
        self.led_pending = False
        self.subscription = self.create_subscription(
            FaceRecognitionResult, self.endpoint('face_rec_msg'),
            self.on_result, 10)

    def face_request(self):
        """空名字表示识别任意已录入人脸，指定名字则只识别该人。"""
        request = FaceRec.Request()
        request.command = 1 if self.username else 0
        request.username = self.username
        request.id = self.request_id
        request.timeout = self.timeout
        return request

    def led_request(self, occupation):
        """使用官方 VP 客户端优先级，控制头灯并支持归还控制权。"""
        request = LedExecute.Request()
        if not hasattr(request, 'occupation'):
            raise RuntimeError(
                '本机 LedExecute 缺少 occupation 字段，无法可靠释放灯光控制；'
                '请先用 ros2 interface show protocol/srv/LedExecute 核对版本。')
        request.occupation = occupation
        request.client = 'vp'
        request.target = 1       # HEAD_LED：头灯。
        request.mode = 2         # USER_DEFINED：自定义 RGB。
        request.effect = 1       # RGB_ON：常亮。
        request.r_value = 0
        request.g_value = 255
        request.b_value = 0
        return request

    def on_result(self, message):
        """按请求 ID 过滤旧消息及其他客户端结果。"""
        if not self.listening or self.result is not None:
            return
        if message.id != self.request_id:
            return
        if message.result == 0:
            if not message.username or message.username in ('stranger', 'jushi'):
                return
            if self.username and message.username != self.username:
                return
        self.result = message

    def run(self):
        """先完成识别，再亮灯，最后归还控制权。"""
        light_on = self.led_request(True)
        self.wait_service(self.face_client)
        self.wait_service(self.led_client)
        self.get_logger().info('请正对摄像头；正在启动人脸识别…')
        self.listening = True
        self.recognition_pending = True
        response = self.call(self.face_client, self.face_request())
        if response.result != 0:
            self.recognition_pending = False
            raise RuntimeError(f'识别请求失败，result={response.result}')
        self.get_logger().info(f'识别已启动，最多等待 {self.timeout} 秒。')
        self.wait_until(lambda: self.result is not None, self.timeout + 5.0)
        self.listening = False
        if self.result.result != 0:
            raise RuntimeError(f'未识别成功，result={self.result.result}')
        # 官方服务端发送成功结果时已移除本次请求，不再全局取消其他识别任务。
        self.recognition_pending = False
        self.get_logger().info(f'识别成功：{self.result.username}')

        self.led_pending = True  # 超时也可能已亮灯，退出时仍尝试释放。
        response = self.call(self.led_client, light_on)
        if response.code != 0:
            raise RuntimeError(f'绿灯未确认点亮，code={response.code}')
        self.get_logger().info(f'绿灯已亮，保持 {self.led_seconds:g} 秒。')
        until = time.monotonic() + self.led_seconds
        self.wait_until(lambda: time.monotonic() >= until, self.led_seconds + 1)
        self.release_led()

    def release_led(self):
        """释放 VP 对头灯的占用，恢复由系统优先级队列决定的灯效。"""
        if not self.led_pending:
            return
        response = self.call(self.led_client, self.led_request(False), cleanup=True)
        if response.code != 0:
            raise RuntimeError(f'释放头灯失败，code={response.code}')
        self.led_pending = False
        self.get_logger().info('已归还头灯控制权。')

    def cleanup(self):
        """中断或失败时释放灯光；识别请求由服务端到期自动清理。"""
        self.listening = False
        self.release_led()
        if self.recognition_pending:
            # 官方 command=2 会清空所有识别请求，不能只取消本客户端。
            self.get_logger().warning(
                '本地已停止等待；服务端识别请求将在 timeout 到期后清理。'
                '若请求响应超时，服务端是否启动尚未确认。')


def main(args=None):
    """ROS 2 命令行入口。"""
    run_demo(FaceRecognizeLed, args)


if __name__ == '__main__':
    main()
