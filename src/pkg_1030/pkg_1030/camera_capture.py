"""通过相机服务开启图像流、拍照，并清理本程序开启的图像发布。"""

import posixpath
import time

from protocol.srv import CameraService

from .face_common import DemoNode, run_demo


class CameraCapture(DemoNode):
    """单次或连续拍照；照片由机器狗服务端保存，不发送运动指令。"""

    def __init__(self):
        super().__init__('camera_capture')
        self.count = self.integer_parameter('count', 1)
        self.width = self.integer_parameter('width', 640, 65535)
        self.height = self.integer_parameter('height', 480, 65535)
        self.fps = self.integer_parameter('fps', 5, 65535)
        self.warmup_seconds = self.positive_parameter('warmup_seconds', 2.0)
        self.interval_seconds = self.positive_parameter('interval_seconds', 2.0)
        self.keep_stream = self.declare_parameter('keep_stream', False).value
        if not isinstance(self.keep_stream, bool):
            raise ValueError('keep_stream 必须是 true 或 false')
        self.camera_client = self.create_client(
            CameraService, self.endpoint('camera_service'))
        self.stream_started_here = False
        self.start_unconfirmed = False

    def integer_parameter(self, name, default, maximum=None):
        """检查计数和 uint16 图像流参数，拒绝布尔值及越界数值。"""
        value = self.declare_parameter(name, default).value
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} 必须是正整数')
        if maximum is not None and value > maximum:
            raise ValueError(f'{name} 不能超过 {maximum}')
        return value

    @staticmethod
    def request(command):
        """创建相机请求；拍照使用服务端默认分辨率。"""
        request = CameraService.Request()
        request.command = command
        request.args = ''
        return request

    @staticmethod
    def response_text(response):
        """同时显示所有响应字段；成功与否以 result 为准。"""
        return (
            f'result={response.result}, msg={response.msg!r}, '
            f'code={response.code}')

    def pause(self, seconds):
        """等待出图或下一次拍照，同时响应 Ctrl+C。"""
        until = time.monotonic() + seconds
        self.wait_until(lambda: time.monotonic() >= until, seconds + 1.0)

    def start_stream(self):
        """尝试开启图像发布，仅在明确成功时取得清理责任。"""
        request = self.request(CameraService.Request.START_IMAGE_PUBLISH)
        request.width = self.width
        request.height = self.height
        request.fps = self.fps
        # 请求超时或中断不代表服务端没有执行，不能贸然关闭共享图像流。
        self.start_unconfirmed = True
        response = self.call(self.camera_client, request)
        self.start_unconfirmed = False
        if response.result == CameraService.Response.RESULT_SUCCESS:
            self.stream_started_here = True
            self.get_logger().info(
                '已开启图像发布：' + self.response_text(response))
        elif response.result == CameraService.Response.RESULT_INVALID_STATE:
            # 官方 startCameraStream 在对应流已开启时返回 INVALID_STATE。
            # 这里只尝试复用；最终仍必须由拍照响应确认功能正常。
            self.get_logger().warning(
                '开启图像发布返回 INVALID_STATE，可能已经开启；'
                '将尝试拍照，退出时不关闭这路已有图像发布。'
                + self.response_text(response))
        else:
            raise RuntimeError('开启图像发布失败：' + self.response_text(response))

    def take_picture(self, index):
        """检查服务响应并打印照片信息，不把 code=0 当作拍照成功。"""
        response = self.call(
            self.camera_client, self.request(CameraService.Request.TAKE_PICTURE))
        detail = self.response_text(response)
        if response.result != CameraService.Response.RESULT_SUCCESS:
            raise RuntimeError(f'第 {index} 张拍照失败：{detail}')

        # 官方 msg 格式为“文件名,字节数”；这里只核验服务端元数据。
        # 文件在机器狗上，客户端可能在另一台电脑，不能检查本机同名路径。
        try:
            filename, size_text = response.msg.rsplit(',', 1)
            size = int(size_text)
            if (not filename or filename in ('.', '..') or
                    posixpath.basename(filename) != filename or size <= 0):
                raise ValueError
        except (ValueError, AttributeError):
            raise RuntimeError(
                '服务报告成功，但照片文件名或大小异常，需核对实机文件：' + detail)

        self.get_logger().info(f'第 {index}/{self.count} 张拍照成功：{detail}')
        self.get_logger().info(
            '机器狗端默认保存路径：' + posixpath.join('/home/mi/Camera', filename))

    def run(self):
        """先开启图像流，等待出图，再按指定数量逐张拍照。"""
        self.wait_service(self.camera_client)
        self.start_stream()
        self.get_logger().info(f'等待相机出图 {self.warmup_seconds:g} 秒。')
        self.pause(self.warmup_seconds)
        for index in range(1, self.count + 1):
            if index > 1:
                self.pause(self.interval_seconds)
            self.take_picture(index)
        self.get_logger().info(
            f'已完成 {self.count} 次拍照；请在机器狗端核对照片内容。')

    def cleanup(self):
        """仅关闭本程序确认开启的流；不删除照片或取消其他视觉任务。"""
        if self.start_unconfirmed:
            self.get_logger().warning(
                '开启图像发布的结果未确认，服务端可能已执行。'
                '本程序不自动关闭状态未知的共享图像流；'
                '请确认没有其他程序使用后，再按需调用 command=10。')
        if not self.stream_started_here:
            return
        if self.keep_stream:
            self.stream_started_here = False
            self.get_logger().info(
                '按 keep_stream=true 保留图像发布；不用时可调用 command=10 关闭。')
            return
        response = self.call(
            self.camera_client,
            self.request(CameraService.Request.STOP_IMAGE_PUBLISH), cleanup=True)
        if response.result != CameraService.Response.RESULT_SUCCESS:
            raise RuntimeError('关闭图像发布未确认成功：' + self.response_text(response))
        self.stream_started_here = False
        self.get_logger().info('已关闭本程序开启的图像发布，已保存的照片不受影响。')


def main(args=None):
    """ROS 2 命令行入口。"""
    run_demo(CameraCapture, args)


if __name__ == '__main__':
    main()
