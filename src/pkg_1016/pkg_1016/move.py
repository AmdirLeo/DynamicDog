import rclpy                          # ROS2 Python interface
from rclpy.node import Node           # ROS2 Node class
from protocol.msg import MotionServoCmd   # Motion command message type

class BasicMove(Node):
    def __init__(self, name):
        super().__init__(name)
        self.dog_name = "mi_desktop_48_b0_2d_7b_03_fd"
        # Create publisher for motion commands
        self.pub = self.create_publisher(MotionServoCmd, f'/{self.dog_name}/motion_servo_cmd', 10)
        # Create timer to call timer_callback every 0.1s (10Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        msg = MotionServoCmd()
        # Configure message for marking time (treading in place)
        msg.motion_id = 303        # ID 303: Specific motion for stepping
        msg.cmd_type = 1           # Command type 1: Execute
        msg.value = 2              # Mode value
        msg.vel_des = [0.0, 0.0, 0.0]  # Zero velocity = stay in place
        msg.step_height = [0.05, 0.05] # 5cm step height

        self.get_logger().info("Action: Marking time...")
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = BasicMove("move_the_dog")
    rclpy.spin(node)           # Keep the node running
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()