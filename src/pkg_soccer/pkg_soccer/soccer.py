"""ROS 2 动捕足球节点（复赛 2v2 主程序）。

最少参数（只保留狗名、四条狗的位置话题、进攻方向）：
    ros2 run pkg_soccer soccer --ros-args \
      -p dog_name:=<本狗运动接口命名空间> \
      -p dog_pose_topic:=/vrpn/dog1/pose \
      -p teammate_pose_topic:=/vrpn/dog2/pose \
      -p opponent1_pose_topic:=/vrpn/opp1/pose \
      -p opponent2_pose_topic:=/vrpn/opp2/pose \
      -p attack_goal:=goal_right

1v1（去掉 teammate/opponent2 两个话题即可，队友与次近对手槽位填 0）：
    ... -p opponent1_pose_topic:=/vrpn/opp1/pose -p attack_goal:=goal_right
单狗（没有对手刚体）：-p rl_policy:=stage2，或 -p control_mode:=rules。

默认值（通常都不用传）：
    control_mode=rl、rl_policy=attacker、enable_motion=true、
    use_mocap_goals=false（场地与球门按尺寸直接算，不需要门刚体）、
    ball_pose_topic=/vrpn/ball/pose、field_length=10、field_width=5.5、
    goal_width=1.5、goal_line_inset=0.415、max_run_time=600、max_kicks=200、
    approach_speed=0.70、kick_speed=0.90、kick_max_speed=1.5、kick_max_ms=450。
    只观察不动：-p enable_motion:=false。

比赛循环（按复赛规则 3.4 / 4.2 / 4.4）：
  - 进球（整球越过门线且在门宽内）→ 记一分并暂停，等裁判把球放回发球点；
  - 球出界、停在门线后、进死区、双方僵持 8 s → 同样暂停等裁判；
  - 球被放到任意发球点（3x3 网格，含开球点）静止 0.4 s、狗也静止后，
    自动 rearm 继续比赛，不需要重启程序，也不消耗 max_run_time 预算；
  - 只有姿态故障、狗离开场地、超时（600 s）和人工 Ctrl+C 才真正结束。

启用运动后，启动时坐/趴会请求一次站立并等待站稳；停止其他运动控制节点。
前置条件（No4 课件）：FZMotion 导出选 Z-up；狗刚体局部 +X 应与狗头一致；
启动 vrpn_listener，确认各刚体发布 geometry_msgs/PoseStamped，单位为米。
初始朝向可以任意；刚体朝向与狗头不一致时用 dog_yaw_offset（弧度）修正。

需要门刚体定位时（可选，默认关闭）：
    -p use_mocap_goals:=true -p goal_left_topic:=/vrpn/goal_left/pose \
    -p goal_right_topic:=/vrpn/goal_right/pose
当时以两门刚体自动建场；门宽用卷尺测两门柱内侧净距离；
goal_left/right_offset 是各自刚体坐标系内原点到门线中心的 [x,y] 米偏移；
门中心偏移标定：把 ball 刚体原点放到该门门线中心，再以
    ros2 run pkg_soccer soccer --ros-args -p calibrate_goal:=goal_left \
      -p use_mocap_goals:=true -p enable_motion:=false
启动自动采样；稳定约 1 秒后打印结果并退出，换另一门重复。
field_end_margin 只在门刚体模式下生效（门线后允许活动的距离）。

踢球方式：只有"全力推球"（push）。控制器进入 KICK 时锁定方向，直接用
    kick_max_speed（默认 1.5 m/s，协议 vel_des 上限）× kick_max_ms（默认 450 ms）
    冲上去把球推出去；不切位控、不抬腿、步高不变。速度靠下面两个参数调：
    -p approach_speed:=0.70   # 平时走位速度
    -p kick_speed:=0.90       # 触球前的推进速度
    （跑步机/台架想空跑不推球时加 -p kick_dry_run:=true，只打印不发动指令）
    kick_distance 是球心到狗刚体原点；位置安装偏置、球半径须现场标定。
    参数只在启动时读取。
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

    def __init__(self):
        super().__init__('soccer')
        self.control_mode = self.param('control_mode', 'rl')
        if self.control_mode not in ('rules', 'rl'):
            raise ValueError('control_mode 必须为 rules 或 rl')
        self.rl_policy = self.param('rl_policy', 'attacker')
        if self.rl_policy not in ('stage2', 'attacker', 'defender'):
            raise ValueError('rl_policy 必须为 stage2/attacker/defender')
        duel = self.control_mode == 'rl' and self.rl_policy != 'stage2'
        defaults = Config()
        if duel:
            defaults = defaults._replace(goal_x=4.585, goal_width=1.5,
                                         field_min_x=-5., field_max_x=5.,
                                         field_min_y=-2.75, field_max_y=2.75)
        # 场地与进攻方向：默认不用门刚体，直接按尺寸算，少传参数。
        self.use_mocap_goals = self.param('use_mocap_goals', False)
        self.attack_goal = self.param('attack_goal', 'goal_right')
        self.field_length = float(self.param('field_length', 10.0))
        self.field_width = float(self.param('field_width', 5.5))
        self.field_end_margin = float(self.param('field_end_margin', .415))
        self.goal_line_inset = float(self.param('goal_line_inset', .415))
        if (type(self.use_mocap_goals) is not bool or
                self.attack_goal not in ('goal_left', 'goal_right')):
            raise ValueError('attack_goal 为 goal_left/goal_right；use_mocap_goals 为布尔值')
        if not math.isfinite(self.field_length) or self.field_length <= 0:
            raise ValueError('field_length 必须为有限正数')
        if not math.isfinite(self.field_width) or self.field_width <= 0:
            raise ValueError('field_width 必须为有限正数')
        if not (math.isfinite(self.goal_line_inset) and
                0 < self.goal_line_inset < self.field_length / 2):
            raise ValueError('goal_line_inset 必须在 (0, field_length/2) 内')
        values = {name: self.param(name, default)
                  for name, default in defaults._asdict().items()}
        if not self.use_mocap_goals:
            values = self.static_field(values)
        self.controller = Controller(Config(**values))
        if self.control_mode == 'rl':
            from .rl_control import RLController
            self.controller = RLController(
                Config(**values), self.param('rl_model_path', ''),
                float(self.param('rl_speed_limit', 0.0)), self.rl_policy,
                dash=bool(self.param('rl_dash', False)),
                dash_speed=float(self.param('rl_dash_speed', 1.5)),
                dash_ms=int(self.param('rl_dash_ms', 450)),
                dash_cooldown=float(self.param('rl_dash_cooldown', 1.2)),
                # 默认 0.9 rad（约 52°）：短冲只在"朝对方球门"的锥内触发。
                # 之前默认 0 = 不限方向，是"把球捅向边线/自家门"的直接来源
                # （train/README.md §17：sp_e 有 40% 的推球是朝自家门方向）。
                dash_goal_cone=float(self.param('rl_dash_goal_cone', 0.9)))
        self.enable_motion = self.param('enable_motion', True)
        if type(self.enable_motion) is not bool:
            raise ValueError('enable_motion 必须为 true/false')
        self.dog_name = self.param('dog_name', 'mi_desktop_48_b0_2d_7b_03_fd').strip('/')
        self.dog_topic = self.param('dog_pose_topic', '/vrpn/xll_dog/pose')
        self.ball_topic = self.param('ball_pose_topic', '/vrpn/ball/pose')
        # 2v2：队友 + 两个对手；旧的 opponent_pose_topic 继续兼容 1v1。
        self.teammate_topic = self.param('teammate_pose_topic', '').strip()
        self.opponent1_topic = self.param('opponent1_pose_topic', '').strip()
        self.opponent2_topic = self.param('opponent2_pose_topic', '').strip()
        self.opponent_topic = self.param('opponent_pose_topic', '').strip()
        self.opponent_yaw_offset = float(self.param('opponent_yaw_offset', 0.0))
        self.teammate_yaw_offset = float(self.param('teammate_yaw_offset', 0.0))
        if not math.isfinite(self.opponent_yaw_offset):
            raise ValueError('opponent_yaw_offset 必须为有限弧度值')
        if not math.isfinite(self.teammate_yaw_offset):
            raise ValueError('teammate_yaw_offset 必须为有限弧度值')
        self.topics = {
            'dog': self.dog_topic,
            'ball': self.ball_topic,
            'teammate': self.teammate_topic,
            'opponent': self.opponent_topic,
            'opponent1': self.opponent1_topic,
            'opponent2': self.opponent2_topic,
        }
        named = [t.strip('/') for t in self.topics.values() if t]
        if len(named) != len(set(named)):
            raise ValueError('狗/球/队友/对手话题不能重复：' + ', '.join(named))
        self.opponent_topics = [t for t in (self.opponent_topic,
                                            self.opponent1_topic,
                                            self.opponent2_topic) if t]
        if duel and not self.opponent_topics:
            raise ValueError('攻防 RL 至少需要一个对手话题：opponent_pose_topic '
                             '或 opponent1/opponent2_pose_topic')
        # 为空时接受 VRPN 实际发布的 frame_id，并要求所有刚体一致。
        # 需要强制检查时可显式传入 -p world_frame:=world。
        self.world_frame = self.param('world_frame', '')
        self.position_scale = float(self.param('position_scale', 1.0))
        self.yaw_offset = float(self.param('dog_yaw_offset', 0.0))
        self.dog_quaternion = None
        # 队友/对手位姿（2v2）：用于最近对手观测、日志和 8 s 僵持判定。
        self.bodies = {'teammate': None, 'opponent': None,
                       'opponent1': None, 'opponent2': None}
        self.body_sequence = 0
        self.posture_timeout = float(self.param('posture_timeout', 1.0))
        self.stand_timeout = float(self.param('stand_timeout', 25.0))
        self.walk_motion_id = int(self.param('walk_motion_id', 305))
        if self.walk_motion_id not in (303, 305):
            raise ValueError('walk_motion_id 只支持 303（慢速）或 305（快速）')
        self.stand_guard = StandGuard(self.walk_motion_id, self.posture_timeout,
                                      self.stand_timeout)
        self.stand_client = None
        self.stand_future = None
        # 踢球方式：push = 全力推球（最大速度短脉冲）；leg = 单腿位控；gait = 305 摆动腿
        # 只保留"全力推球"：不换姿态、不抬腿，进入 KICK 时锁定方向给最大速度。
        self.kick_dry_run = self.param('kick_dry_run', False)
        if type(self.kick_dry_run) is not bool:
            raise ValueError('kick_dry_run 必须为 true/false')
        self.walk_step_height = float(self.param('walk_step_height', 0.05))
        self.kick_max_speed = float(self.param('kick_max_speed', 1.5))
        self.kick_max_ms = int(self.param('kick_max_ms', 450))
        if not math.isfinite(self.kick_max_speed) or not 0.1 <= self.kick_max_speed <= 1.5:
            raise ValueError('kick_max_speed 必须在 [0.1, 1.5] m/s（协议 vel_des 上限 1.5）')
        if not 50 <= self.kick_max_ms <= 2000:
            raise ValueError('kick_max_ms 必须在 [50, 2000] ms')
        self.push_burst_until = None
        self.push_direction = None
        self.push_done = False
        self.calibrate_goal = self.param('calibrate_goal', '')
        self.goal_pose_timeout = float(self.param('goal_pose_timeout', 2.0))
        self.calibration_pose_timeout = float(
            self.param('calibration_pose_timeout', 2.0))
        self.calibration_window = float(
            self.param('calibration_window', 8.0))
        self.calibration_min_samples = int(
            self.param('calibration_min_samples', 12))
        self.calibration_spread = float(
            self.param('calibration_spread', 0.05))
        if self.calibrate_goal not in ('', 'goal_left', 'goal_right'):
            raise ValueError('calibrate_goal 为 goal_left/goal_right 或空字符串')
        if self.calibrate_goal and (self.enable_motion or not self.use_mocap_goals):
            raise ValueError('门中心标定需要：-p use_mocap_goals:=true '
                             '-p enable_motion:=false -p calibrate_goal:=goal_left|goal_right')
        if not (self.field_width > values['goal_width'] and
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
        self.body_subs = []
        if not self.calibrate_goal:
            for kind in ('teammate', 'opponent', 'opponent1', 'opponent2'):
                topic = self.topics[kind]
                if topic:
                    self.body_subs.append(self.create_subscription(
                        PoseStamped, topic,
                        lambda msg, k=kind: self.on_body(k, msg), self.qos))
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
                topic = self.param(name + '_topic', '/vrpn/' + name + '/pose')
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
        else:
            cfg = self.controller.cfg
            points = ', '.join(f'({x:.2f},{y:.2f})'
                               for x, y in self.controller.service_points())
            self.get_logger().info(
                f'球场（按尺寸，不用门刚体）：{self.field_length:.2f}m x '
                f'{self.field_width:.2f}m；门线 x=±{self.field_length/2 - self.goal_line_inset:.3f}，'
                f'门宽={cfg.goal_width:.2f}m；进攻方向={self.attack_goal}'
                f'（goal_yaw={cfg.goal_yaw:.2f}rad）；发球点={points}'
                f'（容差 {cfg.service_point_tolerance:.2f}m）')
        self.get_logger().info(
            f'比赛循环：进球/球出界/死区/僵持 → 暂停等裁判，球回到任一发球点并静止 '
            f'{self.controller.cfg.resume_settle_time:.1f}s、狗也静止后自动继续；'
            f'单轮时间上限 {self.controller.cfg.max_run_time:.0f}s，'
            f'射门次数上限 {self.controller.cfg.max_kicks}')
        self.get_logger().info(
            f'速度：行走={self.controller.cfg.approach_speed:.2f}m/s，'
            f'触球={self.controller.cfg.kick_speed:.2f}m/s，'
            f'推球脉冲={self.kick_max_speed:.2f}m/s×{self.kick_max_ms}ms'
            f'{"（dry-run，不发指令）" if self.kick_dry_run else ""}；'
            f'步态 motion_id={self.walk_motion_id}；'
            f'狗朝向修正 dog_yaw_offset={self.yaw_offset:.3f}rad')
        self.get_logger().info(
            '踢球方式=push（全力推球）：进入 KICK 后锁定方向给最大速度，'
            '步高不变、不切位控、无腿部动作')
        if self.control_mode == 'rl':
            actor = self.controller.actor
            layout = {23: '单狗', 29: '1v1', 41: '2v2'}.get(actor.obs_dim, '未知')
            scale = tuple(float(v) for v in actor.scale)
            self.get_logger().info(
                f'RL {self.rl_policy} 已加载（{actor.obs_dim} 维 / {layout}）；'
                f'动作缩放={scale} m/s（权重自带档位）；'
                f'对手话题={self.opponent_topics if duel else "不使用"}；'
                f'队友话题={self.teammate_topic or "不使用"}；'
                f'8s 僵持判定看全部队友/对手刚体；'
                f'平移限速={self.controller.speed_limit:.2f}m/s；'
                '进球/发球后自动继续，只有超时与故障才结束；'
                f'模式={"实机控制" if self.enable_motion else "只推理、不发运动指令"}')
            if self.controller.dash_enabled:
                self.get_logger().info(
                    f'短冲推球已开启：胸前 0.90 m 内且指令≥0.80 时冲刺 '
                    f'{self.controller.dash_speed:.2f} m/s × '
                    f'{self.controller.dash_duration*1000:.0f} ms，'
                    f'冷却 {self.controller.dash_cooldown:.1f} s（步高不变）')
            elif scale[0] > 0.60:
                self.get_logger().warning(
                    '高速档权重但未开 rl_dash：高位推球只能靠连续走位，'
                    '射门距离受限；确认需要时加 -p rl_dash:=true')
            if self.controller.speed_limit < scale[0]:
                self.get_logger().warning(
                    f'rl_speed_limit={self.controller.speed_limit:.2f} 低于动作缩放 '
                    f'前进分量 {scale[0]:.2f}，实际会被限速压掉一部分动作幅度')
            if actor.obs_dim == 41:
                if not self.teammate_topic:
                    self.get_logger().warning(
                        '41 维策略的队友槽位（29-34）需要 teammate_pose_topic；'
                        '未配置时该槽位恒为 0，策略会按"看不到队友"行动')
                if len(self.opponent_topics) < 2:
                    self.get_logger().warning(
                        '41 维策略的次近对手槽位（35-40）需要 2 个对手话题 '
                        'opponent1/opponent2_pose_topic；当前只有 '
                        f'{len(self.opponent_topics)} 个，第二槽位恒为 0')
            elif self.teammate_topic or len(self.opponent_topics) > 1:
                self.get_logger().warning(
                    f'已配置队友/多个对手，但当前权重是 {actor.obs_dim} 维（1v1），'
                    '队友与次近对手槽位不生效；2v2 需要 41 维权重：把'
                    f' policy_team_{self.rl_policy}_actor.npz 放到 pkg_soccer/ 下，'
                    '或 -p rl_model_path:=<41 维 npz>')
        elif self.opponent_topics or self.teammate_topic:
            self.get_logger().info(
                f'队友/对手话题已配置（规则模式只记录、不参与决策）：'
                f'teammate={self.teammate_topic or "无"}，'
                f'opponents={self.opponent_topics or "无"}')

    def static_field(self, values):
        """不用门刚体时，按场地尺寸和进攻方向直接给出球门与场地坐标。

        复赛场地：长 10 m、宽 5.5 m、门宽 1.5 m、门线距端线 0.415 m（两门相距 9.17 m）。
        进攻方向只由 attack_goal 决定，换边时用另一个取值重启即可。
        """
        half_l = self.field_length / 2.0
        half_w = self.field_width / 2.0
        goal_x = half_l - self.goal_line_inset
        if self.attack_goal == 'goal_left':
            goal_x, goal_yaw = -goal_x, math.pi
        else:
            goal_yaw = 0.0
        out = dict(values)
        out.update(goal_x=goal_x, goal_y=0.0, goal_yaw=goal_yaw,
                   field_center_x=0.0, field_center_y=0.0, field_yaw=0.0,
                   field_min_x=-half_l, field_max_x=half_l,
                   field_min_y=-half_w, field_max_y=half_w)
        return out

    def param(self, name, default):
        """启动参数只读，避免运行时变更却未实际应用。"""
        return self.declare_parameter(
            name, default, ParameterDescriptor(read_only=True)).value

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

    def on_body(self, kind, msg):
        """队友/对手位姿：校验坐标系与数值，RL 模式交给"最近对手"选择。"""
        now = time.monotonic()
        try:
            frame = self.message_frame(msg)
            p, q = msg.pose.position, msg.pose.orientation
            if not all(math.isfinite(v) for v in (p.x, p.y, p.z)):
                raise ValueError('动捕位置包含无效数值')
            yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
            # 队友与对手用各自的刚体朝向修正（同型号狗通常都是 0）。
            yaw = wrap_angle(yaw + (self.teammate_yaw_offset if kind == 'teammate'
                                    else self.opponent_yaw_offset))
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            x = p.x * self.position_scale
            y = p.y * self.position_scale
            self.body_sequence += 1
            self.bodies[kind] = Pose(x, y, yaw, now, frame, stamp,
                                     self.body_sequence)
            if self.control_mode == 'rl':
                self.controller.update_body(kind, x, y, yaw, now, frame, stamp)
        except (ValueError, TypeError, AttributeError) as exc:
            self.bodies[kind] = None
            if self.control_mode == 'rl':
                self.controller.invalidate_body(kind, 'invalid_pose')
            if now - self.last_error_report >= 1.0:
                self.get_logger().warning(f'{kind} 位姿无效：{exc}')
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
        """按运动反馈确认坐/趴/站姿；运行后姿态改变则停止，不再次起身。

        进球/死球暂停期间裁判会搬狗，这段不更新姿态门控，避免把"被抱起"当成故障
        结束整场比赛；恢复比赛后重新按新鲜状态判定。
        """
        guard = self.stand_guard
        if self.controller.paused:
            # 进球/死球暂停期间裁判会搬狗，姿态跳变不算故障；恢复比赛后重新判定。
            return
        try:
            healthy = (msg.ori_error == 0 and msg.footpos_error == 0 and
                       len(msg.motor_error) > 0 and
                       all((int(v) & 0x3fffffff) == 0 for v in msg.motor_error) and
                       0 <= msg.order_process_bar <= 100)
            guard.update(msg.motion_id, msg.switch_status, msg.order_process_bar,
                         healthy, time.monotonic())
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
        """发送可配置的 303/305 步态；步高固定，不切位控。"""
        if self.pub is None:
            return
        msg = MotionServoCmd()
        msg.motion_id = self.walk_motion_id
        msg.cmd_type = MotionServoCmd.SERVO_END if end else MotionServoCmd.SERVO_DATA
        msg.cmd_source = 0
        msg.value = 2
        msg.vel_des = [float(v) for v in velocity]
        msg.step_height = [self.walk_step_height, self.walk_step_height]
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
                nearest = self.controller.opponent
                positions += (f' nearest_opponent=({nearest.x:.2f},{nearest.y:.2f},'
                              f'{nearest.yaw:.2f}rad) age={now-nearest.received:.2f}s'
                              if nearest else ' nearest_opponent=未收到')
                mate = self.controller.teammate
                if self.controller.actor.obs_dim == 41:
                    positions += (f' teammate=({mate.x:.2f},{mate.y:.2f},'
                                  f'{mate.yaw:.2f}rad) age={now-mate.received:.2f}s'
                                  if mate else ' teammate=未收到')
            tracked = []
            for kind, pose in self.bodies.items():
                if pose is None:
                    continue
                age = now - pose.received
                tracked.append(
                    f'{kind}=({pose.x:.2f},{pose.y:.2f})' +
                    ('' if age <= self.controller.cfg.pose_timeout
                     else f'(断流{age:.1f}s)'))
            if tracked:
                positions += ' ' + ' '.join(tracked)
            target = self.controller.target_yaw
            heading = (f'；目标朝向={target:.2f}rad，'
                       f'偏差={math.degrees(wrap_angle(target-d.yaw)):.1f}度'
                       if d is not None and target is not None else '')
            orientation = (f'；dog原始四元数(x,y,z,w)='
                           f'{tuple(round(v, 4) for v in self.dog_quaternion)}'
                           if self.dog_quaternion is not None else '')
            score = (f'；进球 {self.controller.goals}，发球后继续 '
                     f'{self.controller.resumes} 次'
                     if self.controller.goals or self.controller.resumes else '')
            progress = ('；PPO 连续控球' if self.control_mode == 'rl' else
                        f'；射门 {self.controller.kicks}/{self.controller.cfg.max_kicks}')
            progress += score
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
        if self.enable_motion and not c.finished and not c.paused:
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
                self.report(now, 'RL_SHADOW' if c.state == 'RL' else c.state,
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
        # 进入 KICK 时锁定方向，直接给最大速度持续 kick_max_ms（不切位控、不抬腿）。
        # 脉冲跑满整段时间：KICK 会因为"球已移动/已冲出 kick_travel"提前结束，
        # 但断流、暂停、终态必须立刻停手（不再盲推）。
        if not c.finished:
            if c.state == 'KICK' and self.push_burst_until is None and not self.push_done:
                speed = math.hypot(velocity[0], velocity[1])
                if speed > 1e-6:
                    self.push_direction = (velocity[0] / speed, velocity[1] / speed)
                elif c.dog is not None:
                    self.push_direction = (math.cos(c.dog.yaw), math.sin(c.dog.yaw))
                else:
                    self.push_direction = (1.0, 0.0)
                self.push_burst_until = now + self.kick_max_ms / 1000.0
                self.push_done = True
                self.get_logger().info(
                    f'全力推球：{self.kick_max_speed:.2f} m/s × '
                    f'{self.kick_max_ms} ms（步高不变，无腿部动作）'
                    + ('（dry-run，不发指令）' if self.kick_dry_run else ''))
            if self.push_burst_until is not None:
                stalled = (c.paused or
                           c.state in ('WAIT_POSE', 'WAIT_REFEREE'))
                if stalled or now >= self.push_burst_until:
                    self.push_burst_until = None
                else:
                    burst = (self.push_direction[0] * self.kick_max_speed,
                             self.push_direction[1] * self.kick_max_speed, 0.0)
                    if self.kick_dry_run:
                        self.report(now, 'PUSH_DRY_RUN',
                                    f'[dry-run] 全力推球 {self.kick_max_speed:.2f}m/s',
                                    burst)
                        return
                    self.publish(burst)
                    self.report(now, 'PUSH_MAX',
                                f'全力推球 {self.kick_max_speed:.2f}m/s', burst)
                    return
            if c.state != 'KICK':
                self.push_done = False
        if c.finished or c.state in ('WAIT_POSE', 'WAIT_PLAY', 'WAIT_REFEREE'):
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
