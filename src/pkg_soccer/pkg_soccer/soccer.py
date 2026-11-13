"""ROS 2 动捕足球节点。

前置条件（No4 课件）：FZMotion 导出选 Z-up；创建狗刚体时狗头朝 +X；
启动 vrpn_listener，确认两刚体发布 geometry_msgs/PoseStamped，单位为米。
先用遥控让狗站稳，停止其他运动控制节点；本节点不自动站立。

编译、只观察：
    colcon build --packages-select pkg_soccer
    source install/setup.bash
    ros2 run pkg_soccer soccer

现场标定后启用（球门净宽和场地宽度为示例，请实际测量）：
    ros2 run pkg_soccer soccer --ros-args -p enable_motion:=true \
      -p attack_goal:=goal_right -p goal_width:=1.0 -p field_width:=3.0

默认使用两个球门刚体定位，球门须正对；进攻方向为本方门中心到对方门中心。
goal_left/right_offset 是各自刚体坐标系内原点到门线中心的 [x,y] 米偏移。
门宽用卷尺测两门柱内侧净距离；场地宽度为两侧边线间距，场地居中于两门。
门中心偏移标定：将 ball 刚体的原点放到该球门门线中心，再以
    ros2 run pkg_soccer soccer --ros-args -p calibrate_goal:=goal_left
启动自动采样；可以离开终端去摆球，稳定约 1 秒后程序会自动打印结果并退出。
换到另一球门重复，运行时传入两个偏移。若 ball 刚体原点就在球心，
只需将球放在门线中点；高度不影响平面标定。
球门/场地原点和角度由动捕自动获得，field_end_margin 是门线后允许活动的距离。
use_mocap_goals:=false 时可手工指定 goal_x/y/yaw 和 field_* 坐标参数。
狗刚体名称和运动接口 dog_name 是两件事，分别配置。
参数只在启动时读取；位置安装偏置、球半径、kick_distance 须现场标定。
"""

import math
import signal
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from protocol.msg import MotionServoCmd, MotionStatus
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .control import Config, Controller, Pose, wrap_angle, yaw_from_quaternion


class Soccer(Node):
    """20 Hz 控制；订阅回调仅更新观测，独立单调时钟检查数据有效期。"""

    def __init__(self):
        super().__init__('soccer')
        values = {name: self.param(name, default)
                  for name, default in Config()._asdict().items()}
        self.controller = Controller(Config(**values))
        self.enable_motion = self.param('enable_motion', False)
        if type(self.enable_motion) is not bool:
            raise ValueError('enable_motion 必须为 true/false')
        self.dog_name = self.param('dog_name', 'mi_desktop_48_b0_2d_7b_03_fd').strip('/')
        self.dog_topic = self.param('dog_pose_topic', '/vrpn/xll_dog/pose')
        self.ball_topic = self.param('ball_pose_topic', '/vrpn/ball/pose')
        self.world_frame = self.param('world_frame', 'world')
        self.position_scale = float(self.param('position_scale', 1.0))
        self.yaw_offset = float(self.param('dog_yaw_offset', 0.0))
        self.posture_timeout = float(self.param('posture_timeout', 1.0))
        self.use_mocap_goals = self.param('use_mocap_goals', True)
        self.attack_goal = self.param('attack_goal', 'goal_right')
        self.calibrate_goal = self.param('calibrate_goal', '')
        self.field_width = float(self.param('field_width', 3.0))
        self.field_end_margin = float(self.param('field_end_margin', 0.6))
        if (type(self.use_mocap_goals) is not bool or
                self.attack_goal not in ('goal_left', 'goal_right')):
            raise ValueError('attack_goal 为 goal_left/goal_right；use_mocap_goals 为布尔值')
        if self.calibrate_goal not in ('', 'goal_left', 'goal_right'):
            raise ValueError('calibrate_goal 为 goal_left/goal_right 或空字符串')
        if self.calibrate_goal and (self.enable_motion or not self.use_mocap_goals):
            raise ValueError('门中心标定需要动捕球门且 enable_motion=false')
        if not (math.isfinite(self.field_width) and self.field_width > values['goal_width'] and
                math.isfinite(self.field_end_margin) and self.field_end_margin > 0):
            raise ValueError('field_width 必须大于 goal_width，field_end_margin 必须为正数')
        self.goals = {}
        self.goal_offsets = {}
        self.goal_subs = []
        self.goal_sequence = 0
        self.goal_confirm = 0
        self.goal_pair = (0, 0)
        self.goal_reference = None
        self.goal_note = '等待两个球门的动捕位姿'
        self.goal_raw = {}
        self.calibration_values = []
        self.calibration_complete = False
        self.calibration_reported = False
        if (not self.world_frame or not self.dog_topic or not self.ball_topic or
                self.dog_topic == self.ball_topic):
            raise ValueError('坐标系/话题不能为空，狗和球的话题必须不同')
        if not all(math.isfinite(v) for v in
                   (self.position_scale, self.yaw_offset, self.posture_timeout)):
            raise ValueError('位置比例、偏航修正和姿态超时必须有限')
        if self.position_scale <= 0 or self.posture_timeout <= 0:
            raise ValueError('position_scale 和 posture_timeout 必须为正数')
        self.status_at = None
        self.status_count = 0
        self.status_ready = False
        self.armed_once = False
        self.servo_active = False
        self.last_report = -math.inf
        self.last_state = None
        self.last_error_report = -math.inf
        self.pub = None
        self.qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)
        self.dog_sub = self.create_subscription(
            PoseStamped, self.dog_topic, lambda msg: self.on_pose('dog', msg), self.qos)
        self.ball_sub = self.create_subscription(
            PoseStamped, self.ball_topic, lambda msg: self.on_pose('ball', msg), self.qos)
        if self.use_mocap_goals:
            for name in ('goal_left', 'goal_right'):
                offset = self.param(name + '_offset', [0.0, 0.0])
                if len(offset) != 2 or not all(math.isfinite(v) for v in offset):
                    raise ValueError(name + '_offset 应为 [x,y]，单位米')
                self.goal_offsets[name] = offset
                topic = self.param(name + '_topic', '/vrpn/' + name + '/pose')
                self.goal_subs.append(self.create_subscription(
                    PoseStamped, topic, lambda msg, name=name: self.on_goal(name, msg), self.qos))
        self.status_sub = None
        if self.enable_motion:
            self.pub = self.create_publisher(MotionServoCmd, self.endpoint('motion_servo_cmd'), 1)
            self.status_sub = self.create_subscription(
                MotionStatus, self.endpoint('motion_status'), self.on_status, self.qos)
        self.timer = self.create_timer(
            0.05, self.on_timer, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(
            f'动捕足球智能体：dog={self.dog_topic}, ball={self.ball_topic}, '
            f'frame={self.world_frame}；进攻球门={self.attack_goal}；'
            f'球门动捕={self.use_mocap_goals}；运动输出={self.enable_motion}')
        self.get_logger().info(
            '先标定球门/场地和触球距离。运动模式需手动站稳并收到正常 motion_status。'
            if self.enable_motion else
            '当前仅观察：不会创建运动发布器。确认坐标后以 enable_motion:=true 重启。')

    def param(self, name, default):
        """启动参数只读，避免运行时变更却未实际应用。"""
        return self.declare_parameter(
            name, default, ParameterDescriptor(read_only=True)).value

    def endpoint(self, name):
        """兼容有/无机器狗命名空间的运动接口。"""
        return f'/{self.dog_name}/{name}' if self.dog_name else f'/{name}'

    def on_pose(self, kind, msg):
        """校验坐标系和有限位置；狗偏航按课件四元数公式得到。"""
        now = time.monotonic()
        try:
            if msg.header.frame_id != self.world_frame:
                raise ValueError(f'{kind} frame={msg.header.frame_id!r}，'
                                 f'期望 {self.world_frame!r}')
            p, q = msg.pose.position, msg.pose.orientation
            if not all(math.isfinite(v) for v in (p.x, p.y, p.z)):
                raise ValueError('动捕位置包含无效数值')
            yaw = (wrap_angle(yaw_from_quaternion(q.x, q.y, q.z, q.w) + self.yaw_offset)
                   if kind == 'dog' else 0.0)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.controller.update(kind, p.x*self.position_scale, p.y*self.position_scale,
                                   yaw, now, msg.header.frame_id, stamp)
        except (ValueError, TypeError, AttributeError) as exc:
            setattr(self.controller, kind, None)
            self.goal_note = str(exc)
            self.controller.wait_for_pose(str(exc))
            if now - self.last_error_report >= 1.0:
                self.get_logger().warning(str(exc))
                self.last_error_report = now

    def on_goal(self, name, msg):
        """保存球门刚体原点，同时应用正式运行时的门中心偏移。"""
        now = time.monotonic()
        try:
            if msg.header.frame_id != self.world_frame:
                raise ValueError(name + ' 坐标系与 world_frame 不符')
            p, q = msg.pose.position, msg.pose.orientation
            yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
            raw_x = p.x*self.position_scale
            raw_y = p.y*self.position_scale
            ox, oy = self.goal_offsets[name]
            x = raw_x + ox*math.cos(yaw) - oy*math.sin(yaw)
            y = raw_y + ox*math.sin(yaw) + oy*math.cos(yaw)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
            if not all(math.isfinite(v) for v in (x, y, stamp)) or stamp < 0:
                raise ValueError(name + ' 位姿无效')
            previous = self.goals.get(name)
            if previous is not None:
                if now - previous.received > self.controller.cfg.pose_timeout:
                    self.goal_confirm = 0
                    self.controller.wait_for_pose('球门动捕曾断流，重新确认')
                if stamp > 0 and previous.stamp > 0 and stamp <= previous.stamp:
                    if stamp < previous.stamp:
                        self.controller.stop('球门动捕时间戳倒退，请重启')
                    return
            self.goal_sequence += 1
            self.goal_raw[name] = Pose(raw_x, raw_y, yaw, now,
                                       self.world_frame, stamp, self.goal_sequence)
            self.goals[name] = Pose(x, y, yaw, now, self.world_frame, stamp,
                                    self.goal_sequence)
        except (AttributeError, TypeError, ValueError) as exc:
            self.goals.pop(name, None)
            self.goal_raw.pop(name, None)
            self.goal_confirm = 0
            self.goal_note = str(exc)
            self.controller.wait_for_pose(str(exc))

    def calibrate_goal_offset(self, now):
        """自动采样选定门和球的相对位姿，稳定一秒后打印并请求退出。"""
        if not self.calibrate_goal or self.calibration_complete:
            return False
        goal = self.goal_raw.get(self.calibrate_goal)
        ball = self.controller.ball
        if goal is None or ball is None:
            if goal is None and self.goal_note in ('', '等待两个球门的动捕位姿'):
                self.goal_note = '未收到目标门话题 ' + self.calibrate_goal
            elif ball is None and self.goal_note in ('', '等待两个球门的动捕位姿'):
                self.goal_note = '未收到 /vrpn/ball/pose'
            self.calibration_values.clear()
            return True
        if now - goal.received > self.controller.cfg.pose_timeout:
            self.goal_note = f'{self.calibrate_goal} 位姿已超过 {self.controller.cfg.pose_timeout}s'
            self.calibration_values.clear()
            return True
        if now - ball.received > self.controller.cfg.pose_timeout:
            self.goal_note = f'ball 位姿已超过 {self.controller.cfg.pose_timeout}s'
            self.calibration_values.clear()
            return True
        if goal.frame != ball.frame:
            self.goal_note = f'坐标系不一致：门={goal.frame!r}，球={ball.frame!r}'
            self.calibration_values.clear()
            return True
        if abs(goal.received - ball.received) > self.controller.cfg.sync_tolerance:
            self.goal_note = '门和球的接收时间不同步'
            self.calibration_values.clear()
            return True
        dx, dy = ball.x - goal.x, ball.y - goal.y
        ox = dx*math.cos(goal.yaw) + dy*math.sin(goal.yaw)
        oy = -dx*math.sin(goal.yaw) + dy*math.cos(goal.yaw)
        self.calibration_values.append((ox, oy))
        if len(self.calibration_values) > 20:
            self.calibration_values.pop(0)
        if len(self.calibration_values) < 20:
            self.goal_note = '已收到门和球，正在检查静止稳定性'
            return True
        mean_x = sum(v[0] for v in self.calibration_values) / 20
        mean_y = sum(v[1] for v in self.calibration_values) / 20
        spread = max(math.hypot(x - mean_x, y - mean_y)
                     for x, y in self.calibration_values)
        if spread > 0.02:
            self.goal_note = f'球或门仍在移动，最近窗口最大偏差 {spread:.3f}m'
            return True
        if not self.calibration_reported:
            self.get_logger().info(
                f'自动标定完成：{self.calibrate_goal}_offset='
                f'[{mean_x:.3f}, {mean_y:.3f}] 米；'
                '请记录该值，程序即将退出。')
            self.calibration_reported = True
        self.calibration_complete = True
        return True

    def configure_goals(self, now):
        """从两门中心标定球场；运行中移动球门必须重启并重新标定。"""
        if not self.use_mocap_goals:
            return True
        c = self.controller
        left, right = self.goals.get('goal_left'), self.goals.get('goal_right')
        if left is None or right is None:
            self.goal_confirm = 0
            return False
        poses = (left, right)
        if (any(not 0 <= now - p.received <= c.cfg.pose_timeout for p in poses) or
                abs(left.received - right.received) > c.cfg.sync_tolerance or
                not ((left.stamp == right.stamp == 0) or
                     (left.stamp > 0 and right.stamp > 0 and
                      abs(left.stamp - right.stamp) <= c.cfg.sync_tolerance))):
            self.goal_confirm = 0
            self.goal_note = '球门动捕断流或不同步'
            return False
        pair = (left.sequence, right.sequence)
        if pair[0] != self.goal_pair[0] and pair[1] != self.goal_pair[1]:
            self.goal_confirm += 1
            self.goal_pair = pair
        if self.goal_confirm < 3:
            self.goal_note = '等待两个球门连续三组有效位姿'
            return False
        target, own = (right, left) if self.attack_goal == 'goal_right' else (left, right)
        length = math.hypot(target.x - own.x, target.y - own.y)
        if length < 2*c.cfg.behind_distance:
            self.goal_note = '两门距离过近，检查单位、刚体名或门中心偏移'
            return False
        if self.goal_reference is not None:
            if any(math.hypot(p.x - ref.x, p.y - ref.y) > 0.10
                   for p, ref in zip(poses, self.goal_reference)):
                c.stop('球门位置变化超过 10cm，请重新标定并重启')
                return False
            return True
        yaw = math.atan2(target.y - own.y, target.x - own.x)
        config = c.cfg._replace(
            goal_x=target.x, goal_y=target.y, goal_yaw=yaw,
            field_center_x=(left.x + right.x)/2,
            field_center_y=(left.y + right.y)/2, field_yaw=yaw,
            field_min_x=-length/2 - self.field_end_margin,
            field_max_x=length/2 + self.field_end_margin,
            field_min_y=-self.field_width/2, field_max_y=self.field_width/2)
        config.validate()
        c.cfg = config
        c.wait_for_pose('球门已标定，确认狗与球的定位')
        self.goal_reference = poses
        self.get_logger().info(
            f'球场标定完成：goal_left=({left.x:.3f},{left.y:.3f})，'
            f'goal_right=({right.x:.3f},{right.y:.3f}) 米；进攻 {self.attack_goal}，'
            f'朝向={yaw:.3f}rad；门间距={length:.3f}m；'
            f'门宽={config.goal_width:.3f}m；场宽={self.field_width:.3f}m')
        return True

    def on_status(self, msg):
        """启动连续三帧站稳；运行中故障/人工改变姿态后禁止自动恢复。"""
        now = time.monotonic()
        if self.status_at is not None and now - self.status_at > self.posture_timeout:
            self.status_count = 0
            if self.armed_once:
                self.controller.stop('运动状态断流，请检查机器狗后重启')
        self.status_at = now
        try:
            healthy = (msg.ori_error == 0 and msg.footpos_error == 0 and
                       len(msg.motor_error) > 0 and
                       all((int(v) & 0x3fffffff) == 0 for v in msg.motor_error))
            standing = (msg.motion_id == 112 or
                        msg.motion_id == 111 and 96 <= msg.order_process_bar <= 100 or
                        msg.motion_id == 303)
            if not healthy or msg.switch_status not in (0, 1):
                self.controller.stop('运动状态报告故障或禁止切换')
            if self.armed_once and not standing:
                self.controller.stop('运行中姿态被改变，交还控制')
            self.status_ready = healthy and standing and msg.switch_status in (0, 1)
            if self.status_ready and msg.switch_status == 0:
                self.status_count += 1
            elif not self.armed_once:
                self.status_count = 0
        except (AttributeError, TypeError, ValueError):
            self.status_ready = False
            self.controller.stop('MotionStatus 字段不兼容')

    def publish(self, velocity, end=False):
        """课件的 303 步态支持 [vx, vy, wz]，结束帧使用协议常量。"""
        if self.pub is None:
            return
        msg = MotionServoCmd()
        msg.motion_id = 303
        msg.cmd_type = MotionServoCmd.SERVO_END if end else MotionServoCmd.SERVO_DATA
        msg.cmd_source = 0
        msg.value = 2
        msg.vel_des = [float(v) for v in velocity]
        msg.step_height = [0.05, 0.05]
        self.pub.publish(msg)
        self.servo_active = not end

    def end_motion(self):
        """仅在本节点发过控制帧时发结束帧，避免干预其他程序。"""
        if self.servo_active:
            self.publish(Controller.ZERO, end=True)

    def report(self, now, state, reason, velocity):
        """状态变化立即报告，否则每秒报告位置和计划速度。"""
        if state != self.last_state or now - self.last_report >= 1.0:
            d, b = self.controller.dog, self.controller.ball
            positions = (f'dog=({d.x:.2f},{d.y:.2f},{d.yaw:.2f}rad) '
                         f'ball=({b.x:.2f},{b.y:.2f})' if d and b else '等待定位')
            self.get_logger().info(
                f'{state}: {reason}；{positions}；射门 {self.controller.kicks}/'
                f'{self.controller.cfg.max_kicks}；速度=({velocity[0]:.2f},'
                f'{velocity[1]:.2f},{velocity[2]:.2f})')
            self.last_state, self.last_report = state, now

    def on_timer(self):
        """先通过姿态门控，再计算本周期指令，停止时发送 SERVO_END。"""
        now = time.monotonic()
        c = self.controller
        if self.calibrate_goal:
            # 标定模式只需要目标门和球，不要求狗位姿或另一扇门在线。
            self.calibrate_goal_offset(now)
            if self.calibration_complete:
                return
            if now - self.last_report >= 1.0:
                self.get_logger().info(
                    f'自动标定 {self.calibrate_goal}：{self.goal_note or "等待数据"}；'
                    '请把球心放到门线中心并保持约 1 秒。'
                    f'当前采样 {len(self.calibration_values)}/20。')
                self.last_report = now
            return
        if self.enable_motion and not c.finished:
            fresh = self.status_at is not None and now - self.status_at <= self.posture_timeout
            if self.armed_once and not fresh:
                c.stop('运动状态超时，请检查后重启')
            if not c.finished and not (fresh and self.status_ready and
                                       (self.armed_once or self.status_count >= 3)):
                self.end_motion()
                self.report(now, 'WAIT_STAND', '等待手动站稳及连续正常状态', Controller.ZERO)
                return
            if not c.finished and not self.armed_once:
                self.armed_once = True
                c.wait_for_pose('站稳后重新确认动捕数据')
        if not c.finished and not self.configure_goals(now):
            c.wait_for_pose(self.goal_note)
            self.end_motion()
            self.report(now, c.state if c.finished else 'WAIT_GOALS',
                        c.reason if c.finished else self.goal_note, Controller.ZERO)
            return
        # 观察模式只显示规划，不让静止的实机被虚拟计时推进到一次射门。
        if not self.enable_motion:
            if c.poses_valid(now):
                d, b = c.dog, c.ball
                yaw = math.atan2(c.cfg.goal_y - b.y, c.cfg.goal_x - b.x)
                reason = (f'球距={math.hypot(b.x-d.x, b.y-d.y):.2f}m '
                          f'射门方向={yaw:.2f}rad（仅观察）')
                self.report(now, 'OBSERVE', reason, Controller.ZERO)
            else:
                self.report(now, 'WAIT_POSE', '等待同步动捕，检查话题/坐标系', Controller.ZERO)
            return
        velocity = c.command(now)
        if c.finished or c.state == 'WAIT_POSE':
            self.end_motion()
        else:
            self.publish(velocity)
        self.report(now, c.state, c.reason, velocity)

    def stop(self):
        """退出时取消定时器并重复结束帧，保证 ROS context 此时仍有效。"""
        self.timer.cancel()
        active = self.servo_active
        self.controller.stop('用户退出')
        if active:
            for _ in range(3):
                self.publish(Controller.ZERO, end=True)
                time.sleep(0.03)


def main(args=None):
    """Ctrl+C/SIGTERM 先发结束帧，再销毁节点和 ROS context。"""
    rclpy.init(args=args)
    node = None
    stopping = threading.Event()
    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, lambda *_: stopping.set())
        node = Soccer()
        while rclpy.ok() and not stopping.is_set() and not (
                node is not None and node.calibration_complete):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    main()
