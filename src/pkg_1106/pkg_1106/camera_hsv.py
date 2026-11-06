"""通过六个滑动条实时标定相机图像的 HSV 颜色范围。"""

import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class CameraHSV(Node):
    """显示原图、二值掩码和筛选结果，支持暂停来流时继续调节。"""

    def __init__(self):
        super().__init__('camera_hsv')
        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_hsv = None
        self.exit_requested = False
        best_effort = self.declare_parameter('best_effort', False).value
        self.sub_cam = self.create_subscription(
            Image,
            '/image_rgb',
            self.cam_callback,
            QoSProfile(
                depth=10,
                reliability=(
                    ReliabilityPolicy.BEST_EFFORT if best_effort
                    else ReliabilityPolicy.RELIABLE
                ),
                durability=DurabilityPolicy.VOLATILE,
            ),
        )

        cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Trackbars', 480, 300)
        for channel, maximum in (('H', 179), ('S', 255), ('V', 255)):
            cv2.createTrackbar(
                channel + ' min', 'Trackbars', 0, maximum, lambda value: None)
            cv2.createTrackbar(
                channel + ' max', 'Trackbars', maximum, maximum,
                lambda value: None)
        for window in ('Original', 'Mask', 'Result'):
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        # 即使没有新图像，也处理滑动条和窗口事件。
        self.gui_timer = self.create_timer(1.0 / 30.0, self.update_gui)
        self.get_logger().info(
            '等待 /image_rgb；拖动滑动条调节 HSV，各通道 min 应不大于 max。'
            '在图像窗口按 s 输出阈值，按 q / Esc 或在终端按 Ctrl+C 退出。')

    def cam_callback(self, my_img):
        """缓存最后一帧及其 HSV 图像，转换失败时保留上一帧。"""
        try:
            frame = self.bridge.imgmsg_to_cv2(my_img, 'bgr8')
        except CvBridgeError as exc:
            self.get_logger().error(f'图像转换失败：{exc}')
            return
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.latest_frame = frame
        self.latest_hsv = hsv

    def update_gui(self):
        """读取阈值并刷新掩码；按 s 将当前阈值输出到终端。"""
        # 不用 WND_PROP_VISIBLE 判断退出：窗口尚未显示或后端不支持时，
        # 它可能返回 0 / -1，导致首次 waitKey 处理事件之前就退出。
        # 通过 q、Esc 或终端 Ctrl+C 显式退出。
        lower = tuple(
            cv2.getTrackbarPos(channel + ' min', 'Trackbars')
            for channel in ('H', 'S', 'V'))
        upper = tuple(
            cv2.getTrackbarPos(channel + ' max', 'Trackbars')
            for channel in ('H', 'S', 'V'))

        if self.latest_frame is not None:
            mask = cv2.inRange(self.latest_hsv, lower, upper)
            result = cv2.bitwise_and(
                self.latest_frame, self.latest_frame, mask=mask)
            cv2.imshow('Original', self.latest_frame)
            cv2.imshow('Mask', mask)
            cv2.imshow('Result', result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            self.get_logger().info(f'HSV lower={lower}, upper={upper}')
        elif key in (ord('q'), 27):
            self.exit_requested = True


def main(args=None):
    """运行图形界面，退出时释放 ROS 节点及所有窗口。"""
    rclpy.init(args=args)
    node = None
    try:
        node = CameraHSV()
        while rclpy.ok() and not node.exit_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
