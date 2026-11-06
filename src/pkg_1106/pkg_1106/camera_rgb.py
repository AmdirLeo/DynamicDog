"""订阅 /image_rgb，在 MobaXterm 的 X11 窗口中显示相机图像。"""

import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class CameraRGB(Node):
    """将 ROS 图像转换为 OpenCV BGR 图像并显示。"""

    def __init__(self):
        super().__init__('camera_rgb')
        self.bridge = CvBridge()
        best_effort = self.declare_parameter('best_effort', False).value
        reliability = (
            ReliabilityPolicy.BEST_EFFORT if best_effort
            else ReliabilityPolicy.RELIABLE
        )
        self.sub_cam = self.create_subscription(
            Image,
            '/image_rgb',
            self.cam_callback,
            QoSProfile(
                depth=10,
                reliability=reliability,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self.get_logger().info('等待 /image_rgb 图像；按 Ctrl+C 退出。')

    def cam_callback(self, my_img):
        """转换成功后显示图像，转换失败则跳过当前帧。"""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(my_img, 'bgr8')
        except CvBridgeError as exc:
            self.get_logger().error(f'图像转换失败：{exc}')
            return
        cv2.imshow('Image window', cv_img)
        cv2.waitKey(3)


def main(args=None):
    """启动节点，并在退出时释放节点和图像窗口。"""
    rclpy.init(args=args)
    node = None
    try:
        node = CameraRGB()
        rclpy.spin(node)
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
