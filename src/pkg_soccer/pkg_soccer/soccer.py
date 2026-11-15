"""ROS 2 动捕足球节点。

前置条件（No4 课件）：FZMotion 导出选 Z-up；狗刚体局部 +X 应与狗头一致；
启动 vrpn_listener，确认两刚体发布 geometry_msgs/PoseStamped，单位为米。
启用运动后，启动时坐/趴会请求一次站立并等待站稳；停止其他运动控制节点。
初始朝向可以任意；刚体朝向与狗头不一致时用 dog_yaw_offset（弧度）修正。

编译、只观察：
    colcon build --packages-select pkg_soccer
    source install/setup.bash
    ros2 run pkg_soccer soccer

现场标定后启用（球门净宽和场地宽度为示例，请实际测量）：
    ros2 run pkg_soccer soccer --ros-args -p enable_motion:=true \
      -p attack_goal:=goal_right -p goal_width:=1.53 -p field_width:=6.0 \
      -p walk_motion_id:=305

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
from .stand_guard import StandGuard


class Soccer(Node):
    """20 Hz 控制；订阅回调仅更新观测，独立单调时钟检查数据有效期。"""

    def __init__(self, attacker_profile=False):
        super().__init__('soccer')
        self.attacker_profile = attacker_profile
        self.control_mode = self.param('control_mode', 'rl' if attacker_profile else 'rules')
        if self.control_mode not in ('rules', 'rl'):
            raise ValueError('control_mode 必须为 rules 或 rl')
        self.rl_policy = self.param('rl_policy', 'attacker' if attacker_profile else 'stage2')
        if self.rl_policy not in ('stage2', 'attacker', 'defender'):
            raise ValueError('rl_policy 必须为 stage2/attacker/defender')
        if attacker_profile and (self.control_mode != 'rl' or self.rl_policy != 'attacker'):
            raise ValueError('attacker 入口固定使用 ATTACK 混合控制；其他模式请使用 soccer 入口')
        duel = self.control_mode == 'rl' and self.rl_policy != 'stage2'
        defaults = Config()
        if duel:
            defaults = defaults._replace(goal_x=4.585, goal_width=1.5,
                                         field_min_x=-5., field_max_x=5.,
                                         field_min_y=-2.75, field_max_y=2.75)
        values = {name: self.param(name, default)
                  for name, default in defaults._asdict().items()}
        self.controller = Controller(Config(**values))
        if self.control_mode == 'rl':
            from .rl_control import RLController
            self.controller = RLController(
                Config(**values), self.param('rl_model_path', ''),
                float(self.fixed_param('rl_speed_limit', .45) if attacker_profile else
                      self.param('rl_speed_limit', .45)), self.rl_policy)
        self.enable_motion = self.param('enable_motion', False)
        if type(self.enable_motion) is not bool:
            raise ValueError('enable_motion 必须为 true/false')
        self.dog_name = self.param('dog_name', 'mi_desktop_48_b0_2d_7b_03_fd').strip('/')
        self.dog_topic = self.param('dog_pose_topic', '/vrpn/xll_dog/pose')
        self.ball_topic = self.fixed_param('ball_pose_topic', '/vrpn/ball/pose')
        self.opponent_topic = self.param('opponent_pose_topic', '').strip()
        self.opponent_yaw_offset = float(self.param('opponent_yaw_offset', 0.0))
        if duel and (not self.opponent_topic or
                     self.opponent_topic.strip('/') in
                     (self.dog_topic.strip('/'), self.ball_topic.strip('/'))):
            raise ValueError('攻防 RL 需要 opponent_pose_topic，且不能与本狗/球话题相同')
        if not math.isfinite(self.opponent_yaw_offset):
            raise ValueError('opponent_yaw_offset 必须为有限弧度值')
        # Lock the actual first VRPN frame for this run; do not guess its name.
        self.world_frame = self.fixed_param('world_frame', '')
        self.position_scale = float(self.fixed_param('position_scale', 1.0))
        calibrated_yaw = math.pi if attacker_profile and self.dog_topic == '/vrpn/xll_dog/pose' else 0.
        self.yaw_offset = float(self.param('dog_yaw_offset', calibrated_yaw))
        self.dog_quaternion = None
        self.posture_timeout = float(self.param('posture_timeout', 1.0))
        self.stand_timeout = float(self.param('stand_timeout', 25.0))
        self.walk_motion_id = int(self.param('walk_motion_id', 305))
        if self.walk_motion_id not in (303, 305):
            raise ValueError('walk_motion_id 只支持 303（慢速）或 305（快速）')
        self.stand_guard = StandGuard(self.walk_motion_id, self.posture_timeout,
                                      self.stand_timeout)
        self.stand_client = None
        self.stand_future = None
        self.use_mocap_goals = self.param('use_mocap_goals', True)
        self.attack_goal = self.param('attack_goal', 'goal_right')
        self.calibrate_goal = self.param('calibrate_goal', '')
        # 攻防模型使用复赛外场 10 x 5.5 m；单狗保留原默认参数。
        self.field_width = float(self.param('field_width', 5.5 if duel else 6.0))
        self.field_end_margin = float(self.param('field_end_margin', .415 if duel else .6))
        self.field_length = float(self.param('field_length', 10.0))
        if not math.isfinite(self.field_length) or self.field_length <= 0:
            raise ValueError('field_length 必须为有限正数')
        self.goal_pose_timeout = float(self.param('goal_pose_timeout', 2.0))
        self.calibration_pose_timeout = float(
            self.param('calibration_pose_timeout', 2.0))
        self.calibration_window = float(
            self.param('calibration_window', 8.0))
        self.calibration_min_samples = int(
            self.param('calibration_min_samples', 12))
        self.calibration_spread = float(
            self.param('calibration_spread', 0.05))
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
        if not math.isfinite(self.goal_pose_timeout) or self.goal_pose_timeout <= 0:
            raise ValueError('goal_pose_timeout 必须为有限正数')
        if (not math.isfinite(self.calibration_pose_timeout) or
                not math.isfinite(self.calibration_window) or
                self.calibration_pose_timeout <= 0 or self.calibration_window <
                self.calibration_pose_timeout):
            raise ValueError('calibration_pose_timeout/window 参数无效')
        if (self.calibration_min_samples < 3 or
                not math.isfinite(self.calibration_spread) or
                self.calibration_spread <= 0):
            raise ValueError('calibration_min_samples/spread 参数无效')
        self.goals = {}
        self.goal_offsets = {}
        self.goal_subs = []
        self.goal_sequence = 0
        self.goal_confirm = 0
        self.goal_pair = (0, 0)
        self.goal_reference = None
        self.goal_note = '等待两个球门的动捕位姿'
        self.detected_frame = self.world_frame or None
        self.goal_raw = {}
        self.calibration_values = []
        self.calibration_pair = (0, 0)
        self.calibration_complete = False
        self.calibration_reported = False
        if (not self.dog_topic or not self.ball_topic or
                self.dog_topic == self.ball_topic):
            raise ValueError('话题不能为空，狗和球的话题必须不同')
        if not all(math.isfinite(v) for v in
                   (self.position_scale, self.yaw_offset, self.posture_timeout,
                    self.goal_pose_timeout)):
            raise ValueError('位置比例、偏航和姿态超时必须有限')
        if self.position_scale <= 0 or self.posture_timeout <= 0:
            raise ValueError('position_scale 和 posture_timeout 必须为正数')
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
        # 门中心标定不需要狗位姿；不订阅 dog 可避免狗断流干扰 offset 采样。
        self.dog_sub = None
        if not self.calibrate_goal:
            self.dog_sub = self.create_subscription(
                PoseStamped, self.dog_topic, lambda msg: self.on_pose('dog', msg), self.qos)
        self.ball_sub = self.create_subscription(
            PoseStamped, self.ball_topic, lambda msg: self.on_pose('ball', msg), self.qos)
        self.opponent_sub = None
        if duel and not self.calibrate_goal:
            self.opponent_sub = self.create_subscription(
                PoseStamped, self.opponent_topic,
                lambda msg: self.on_pose('opponent', msg), self.qos)
        if self.use_mocap_goals:
            # 用户实测的刚体原点到门线中心偏移，单位米。
            measured_offsets = {
                'goal_left': [0.125, -0.030],
                'goal_right': [0.034, 0.017],
            }
            for name in ('goal_left', 'goal_right'):
                offset = self.param(name + '_offset', measured_offsets[name])
                if len(offset) != 2 or not all(math.isfinite(v) for v in offset):
                    raise ValueError(name + '_offset 应为 [x,y]，单位米')
                self.goal_offsets[name] = offset
                topic = self.fixed_param(name + '_topic', '/vrpn/' + name + '/pose')
                self.goal_subs.append(self.create_subscription(
                    PoseStamped, topic, lambda msg, name=name: self.on_goal(name, msg), self.qos))
        self.status_sub = None
        if self.enable_motion:
            self.pub = self.create_publisher(MotionServoCmd, self.endpoint('motion_servo_cmd'), 1)
            self.status_sub = self.create_subscription(
                MotionStatus, self.endpoint('motion_status'), self.on_status, self.qos)
            from protocol.srv import MotionResultCmd
            self.stand_service_type = MotionResultCmd
            self.stand_client = self.create_client(
                MotionResultCmd, self.endpoint('motion_result_cmd'))
        self.timer = self.create_timer(
            0.05, self.on_timer, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(
            f'动捕足球智能体：dog={self.dog_topic}, ball={self.ball_topic}, '
            f'frame={self.world_frame}；进攻球门={self.attack_goal}；'
            f'球门动捕={self.use_mocap_goals}；运动输出={self.enable_motion}')
        if self.calibrate_goal:
            self.get_logger().info(
                f'当前为门中心标定模式：calibrate_goal={self.calibrate_goal!r}；'
                '稳定采样后输出 offset 并退出。')
        elif self.enable_motion:
            self.get_logger().info(
                '当前为运动模式：启动时自动站立已内置；'
                '等待正常 motion_status，站稳后才允许踢球。')
        else:
            self.get_logger().info(
                '当前为 OBSERVE 观察模式：calibrate_goal 为空，不计算或输出 offset。'
                '已有偏移会应用于门中心定位；确认后以 enable_motion:=true 重启。')
        if self.use_mocap_goals:
            self.get_logger().info(f'采用球门偏移（米）：{self.goal_offsets}')
        self.get_logger().info(
            f'速度上限：行走={self.controller.cfg.approach_speed:.2f}m/s，'
            f'触球={self.controller.cfg.kick_speed:.2f}m/s；'
            f'步态 motion_id={self.walk_motion_id}；'
            f'狗朝向修正 dog_yaw_offset={self.yaw_offset:.3f}rad')
        if self.control_mode == 'rl':
            self.get_logger().info(
                f'RL {self.rl_policy} 已加载（{self.controller.actor.obs_dim} 维）；'
                f'对手话题={self.opponent_topic if duel else "不使用"}；'
                f'平移限速={self.controller.speed_limit:.2f}m/s；'
                'max_kicks 不适用于连续 RL 控制，进球/超时/故障时结束；'
                f'模式={"实机控制" if self.enable_motion else "只推理、不发运动指令"}')

    def param(self, name, default):
        """启动参数只读，避免运行时变更却未实际应用。"""
        return self.declare_parameter(
            name, default, ParameterDescriptor(read_only=True)).value

    def fixed_param(self, name, value):
        """Attacker shortcut fixes lab settings; legacy soccer remains configurable."""
        actual = self.param(name, value)
        if self.attacker_profile and actual != value:
            raise ValueError(f'attacker 入口固定 {name}={value!r}；请移除该覆盖参数')
        return actual

    def endpoint(self, name):
        """兼容有/无机器狗命名空间的运动接口。"""
        return f'/{self.dog_name}/{name}' if self.dog_name else f'/{name}'

    def message_frame(self, msg):
        """接纳 VRPN 的实际 frame_id；空 frame 统一为本节点内部名称。"""
        frame = (msg.header.frame_id or '').strip() or '__vrpn_default_frame__'
        if self.world_frame and frame != self.world_frame:
            raise ValueError(f'frame={frame!r}，期望 {self.world_frame!r}')
        if self.detected_frame is None:
            self.detected_frame = frame
        elif frame != self.detected_frame:
            raise ValueError(f'刚体坐标系不一致：收到 {frame!r}，已有 {self.detected_frame!r}')
        return frame

    def on_pose(self, kind, msg):
        """校验坐标系和有限位置；狗偏航按课件四元数公式得到。"""
        now = time.monotonic()
        try:
            frame = self.message_frame(msg)
            p, q = msg.pose.position, msg.pose.orientation
            if not all(math.isfinite(v) for v in (p.x, p.y, p.z)):
                raise ValueError('动捕位置包含无效数值')
            yaw = (wrap_angle(yaw_from_quaternion(q.x, q.y, q.z, q.w) + self.yaw_offset)
                   if kind == 'dog' else
                   (wrap_angle(yaw_from_quaternion(q.x, q.y, q.z, q.w) +
                               self.opponent_yaw_offset) if kind == 'opponent' else 0.0))
            if kind == 'dog':
                self.dog_quaternion = (q.x, q.y, q.z, q.w)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.controller.update(kind, p.x*self.position_scale, p.y*self.position_scale,
                                   yaw, now, frame, stamp)
        except (ValueError, TypeError, AttributeError) as exc:
            setattr(self.controller, kind, None)
            if self.control_mode == 'rl' and kind in self.controller.tracks:
                self.controller.tracks[kind].invalidate('invalid_pose')
            self.goal_note = str(exc)
            self.controller.wait_for_pose(str(exc))
            if now - self.last_error_report >= 1.0:
                self.get_logger().warning(str(exc))
                self.last_error_report = now

    def on_goal(self, name, msg):
        """保存球门刚体原点，同时应用正式运行时的门中心偏移。"""
        now = time.monotonic()
        try:
            frame = self.message_frame(msg)
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
                    # 以实际回调时间判断新鲜度；不把复用的 VRPN stamp 当成重复帧。
                    stamp = previous.stamp
            self.goal_sequence += 1
            self.goal_raw[name] = Pose(raw_x, raw_y, yaw, now,
                                       frame, stamp, self.goal_sequence)
            self.goals[name] = Pose(x, y, yaw, now, frame, stamp,
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
        if now - goal.received > self.calibration_pose_timeout:
            self.goal_note = f'{self.calibrate_goal} 位姿已超过 {self.calibration_pose_timeout}s'
            self.calibration_values.clear()
            return True
        if now - ball.received > self.calibration_pose_timeout:
            self.goal_note = f'ball 位姿已超过 {self.calibration_pose_timeout}s'
            self.calibration_values.clear()
            return True
        if goal.frame != ball.frame:
            self.goal_note = f'坐标系不一致：门={goal.frame!r}，球={ball.frame!r}'
            self.calibration_values.clear()
            return True
        # 标定对象是静止的门和球，不要求两个回调在同一时刻到达；
        # 只要各自没有超过 calibration_pose_timeout，就可放入滑动窗口。
        dx, dy = ball.x - goal.x, ball.y - goal.y
        ox = dx*math.cos(goal.yaw) + dy*math.sin(goal.yaw)
        oy = -dx*math.sin(goal.yaw) + dy*math.cos(goal.yaw)
        pair = (goal.sequence, ball.sequence)
        if pair != self.calibration_pair:
            self.calibration_values.append((now, ox, oy))
            self.calibration_pair = pair
        self.calibration_values = [
            sample for sample in self.calibration_values
            if now - sample[0] <= self.calibration_window]
        if len(self.calibration_values) < self.calibration_min_samples:
            self.goal_note = '已收到门和球，正在检查静止稳定性'
            return True
        mean_x = sum(v[1] for v in self.calibration_values) / len(self.calibration_values)
        mean_y = sum(v[2] for v in self.calibration_values) / len(self.calibration_values)
        spread = max(math.hypot(x - mean_x, y - mean_y)
                     for _, x, y in self.calibration_values)
        if spread > self.calibration_spread:
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
            missing = []
            if left is None:
                missing.append('goal_left')
            if right is None:
                missing.append('goal_right')
            self.goal_note = '未收到：' + ', '.join(missing)
            return False
        poses = (left, right)
        if any(not 0 <= now - p.received <= self.goal_pose_timeout for p in poses):
            self.goal_confirm = 0
            ages = ', '.join(f'{name}={now - pose.received:.2f}s'
                             for name, pose in zip(('left', 'right'), poses))
            self.goal_note = (f'球门数据过期（{ages}，允许 '
                              f'{self.goal_pose_timeout:.1f}s）')
            return False
        if left.frame != right.frame:
            self.goal_confirm = 0
            self.goal_note = f'两门坐标系不一致：{left.frame!r} / {right.frame!r}'
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
        if self.control_mode == 'rl' and c.duel:
            # Outer length is independent of measured goal-line separation.
            config = config._replace(field_min_x=-self.field_length/2,
                                     field_max_x=self.field_length/2)
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
        """按运动反馈确认坐/趴/站姿；运行后姿态改变则停止，不再次起身。"""
        guard = self.stand_guard
        already_failed = guard.state == 'POSTURE_FAILED'
        try:
            healthy = (msg.ori_error == 0 and msg.footpos_error == 0 and
                       len(msg.motor_error) > 0 and
                       all((int(v) & 0x3fffffff) == 0 for v in msg.motor_error) and
                       0 <= msg.order_process_bar <= 100)
            guard.update(msg.motion_id, msg.switch_status, msg.order_process_bar,
                         healthy, time.monotonic())
            if guard.state == 'POSTURE_FAILED' and not already_failed:
                # Preserve the triggering frame, even if later reports recover.
                guard.reason += (
                    f'；motion_id={msg.motion_id}, switch_status={msg.switch_status}, '
                    f'ori_error={msg.ori_error}, footpos_error={msg.footpos_error}, '
                    f'order_process_bar={msg.order_process_bar}, '
                    f'motor_error={list(msg.motor_error)}')
        except (AttributeError, TypeError, ValueError):
            guard.fail('MotionStatus 字段不兼容')
        if guard.state == 'POSTURE_FAILED':
            self.controller.stop(guard.reason)
            self.cancel_stand_wait()

    def cancel_stand_wait(self):
        """取消本地服务等待；不会声称已经撤销服务端的起身动作。"""
        if self.stand_future is not None:
            if not self.stand_future.done():
                self.stand_future.cancel()
            self.stand_future = None

    def posture_allows_motion(self, now):
        """只请求一次 111 站立；成功且连续反馈站稳后才发送快速步态。"""
        guard = self.stand_guard
        # 先检查超时，晚到的服务结果不能解除故障锁定。
        guard.command(now)
        if guard.state == 'POSTURE_FAILED':
            self.controller.stop(guard.reason)
            self.cancel_stand_wait()
            return False
        if self.stand_future is not None and self.stand_future.done():
            future = self.stand_future
            self.stand_future = None
            try:
                response = future.result()
                guard.result(response.result, response.motion_id, response.code)
            except Exception as exc:
                guard.fail(f'站立服务异常：{exc}')
        action = guard.command(now)
        if action == 'REQUEST':
            if not self.stand_client.service_is_ready():
                guard.reason = '等待站立服务：' + self.endpoint('motion_result_cmd')
            else:
                guard.requested(now)
                try:
                    request = self.stand_service_type.Request()
                    request.motion_id = 111
                    request.cmd_source = 0
                    request.duration = 0
                    for name, size in (
                            ('vel_des', 3), ('rpy_des', 3), ('pos_des', 3),
                            ('acc_des', 6), ('ctrl_point', 3),
                            ('foot_pose', 6), ('step_height', 2)):
                        setattr(request, name, [0.0] * size)
                    self.stand_future = self.stand_client.call_async(request)
                    self.get_logger().info('已请求站立 motion_id=111，等待服务成功和站稳反馈。')
                except Exception as exc:
                    guard.fail(f'发送站立请求失败：{exc}')
        if guard.state == 'POSTURE_FAILED':
            self.controller.stop(guard.reason)
            self.cancel_stand_wait()
        return action == 'WALK'

    def publish(self, velocity, end=False):
        """发送可配置的 303/305 步态，结束帧使用协议常量。"""
        if self.pub is None:
            return
        msg = MotionServoCmd()
        msg.motion_id = self.walk_motion_id
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
            if self.control_mode == 'rl' and self.controller.duel:
                o = self.controller.opponent
                positions += (f' opponent=({o.x:.2f},{o.y:.2f},{o.yaw:.2f}rad)'
                              f' age={now-o.received:.2f}s' if o else ' opponent=未收到')
            target = self.controller.target_yaw
            heading = (f'；目标朝向={target:.2f}rad，'
                       f'偏差={math.degrees(wrap_angle(target-d.yaw)):.1f}度'
                       if d is not None and target is not None else '')
            orientation = (f'；dog原始四元数(x,y,z,w)='
                           f'{tuple(round(v, 4) for v in self.dog_quaternion)}'
                           if self.dog_quaternion is not None else '')
            progress = ('；PPO 连续控球' if self.control_mode == 'rl' else
                        f'；射门 {self.controller.kicks}/{self.controller.cfg.max_kicks}')
            self.get_logger().info(
                f'{state}: {reason}；{positions}{progress}；速度=({velocity[0]:.2f},'
                f'{velocity[1]:.2f},{velocity[2]:.2f}){heading}{orientation}')
            self.last_state, self.last_report = state, now

    def pose_wait_reason(self, now):
        """把定位等待拆成可操作的原因，避免笼统显示“断流/不同步”。"""
        d, b = self.controller.dog, self.controller.ball
        if d is None:
            return '未收到 dog 位姿'
        if b is None:
            return '未收到 ball 位姿'
        if d.frame != b.frame:
            return f'dog/ball 坐标系不同：{d.frame!r}/{b.frame!r}'
        ages = f'dog={now-d.received:.2f}s, ball={now-b.received:.2f}s'
        if now - d.received > self.controller.cfg.pose_timeout:
            return f'dog 位姿过期（{ages}）'
        if now - b.received > self.controller.cfg.pose_timeout:
            return f'ball 位姿过期（{ages}）'
        return f'位姿已收到但尚未满足控制条件（{ages}）'

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
                    f'当前采样 {len(self.calibration_values)}/'
                    f'{self.calibration_min_samples}。')
                self.last_report = now
            return
        if self.enable_motion and not c.finished:
            if not self.posture_allows_motion(now):
                self.end_motion()
                guard = self.stand_guard
                self.report(now, guard.state, guard.reason, Controller.ZERO)
                return
            if not self.armed_once:
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
            if self.control_mode == 'rl':
                velocity = c.command(now)
                self.report(now, (c.state + '_SHADOW') if c.state in (
                    'RL', 'ATTACK_RL', 'ATTACK_REPOSITION', 'ATTACK_BLOCKED') else c.state,
                            c.reason + '（只推理，不发运动指令）', velocity)
                return
            if c.poses_valid(now):
                d, b = c.dog, c.ball
                yaw = math.atan2(c.cfg.goal_y - b.y, c.cfg.goal_x - b.x)
                reason = (f'球距={math.hypot(b.x-d.x, b.y-d.y):.2f}m '
                          f'射门方向={yaw:.2f}rad（仅观察）')
                self.report(now, 'OBSERVE', reason, Controller.ZERO)
            else:
                self.report(now, 'WAIT_POSE', self.pose_wait_reason(now), Controller.ZERO)
            return
        velocity = c.command(now)
        if c.finished or c.state in ('WAIT_POSE', 'WAIT_REFEREE'):
            self.end_motion()
        else:
            self.publish(velocity)
        self.report(now, c.state, c.reason, velocity)

    def stop(self):
        """退出时取消定时器并重复结束帧，保证 ROS context 此时仍有效。"""
        self.timer.cancel()
        active = self.servo_active
        self.controller.stop('用户退出')
        self.cancel_stand_wait()
        if active:
            for _ in range(3):
                self.publish(Controller.ZERO, end=True)
                time.sleep(0.03)


def main(args=None, attacker_profile=False):
    """Ctrl+C/SIGTERM 先发结束帧，再销毁节点和 ROS context。"""
    rclpy.init(args=args)
    node = None
    stopping = threading.Event()
    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, lambda *_: stopping.set())
        node = Soccer(attacker_profile=attacker_profile)
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


def attacker_main(args=None):
    """Short competition entry: fixed lab configuration, ATTACK supervisor."""
    main(args, attacker_profile=True)


if __name__ == '__main__':
    main()
