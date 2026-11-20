"""动捕足球智能体：绕到球后、对门、触球、观察结果、重新组织射门。

纯 Python，所有位置使用 Z-up 世界坐标（米），速度使用机身坐标。
只规划狗与球的相对运动；场地应空旷，不包含其他机器人/障碍物避让。
KICK 使用可配置的 303/305 步态前冲触球，不假定存在专用抬腿动作编号。

比赛循环（按复赛规则）：进球、球出界、球停在门线后都不再是终态，而是进入
`WAIT_PLAY` 暂停等裁判；球被放回任意发球点并静止、狗也静止后 `rearm()` 重新
开始一整轮。只有人工停止、姿态故障和超时才让 `finished` 为真。
"""

import math
from typing import NamedTuple


def wrap_angle(angle):
    """归一化到 [-pi, pi]。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(x, y, z, w):
    """课件中的 Z-Y-X 偏航公式，拒绝无效四元数。"""
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError('无效的狗姿态四元数')
    x, y, z, w = (v / norm for v in (x, y, z, w))
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


class Config(NamedTuple):
    """可通过 ROS 启动参数覆盖，角度用弧度、位置用米。"""

    goal_x: float = 2.0
    goal_y: float = 0.0
    goal_yaw: float = 0.0  # 从场内指向球门后方的法线方向；0 表示 +X。
    goal_width: float = 1.53
    ball_radius: float = 0.12
    pose_timeout: float = 0.5
    sync_tolerance: float = 0.15
    approach_speed: float = 0.70   # 走位速度（2026-11-17 提速：0.45 → 0.70）
    turn_speed: float = 0.7
    turn_response_grace: float = 1.0  # 起转/换向后等待步态和通信响应，再检查方向。
    angle_tolerance: float = 0.25
    behind_distance: float = 0.70
    ball_clearance: float = 0.48
    position_tolerance: float = 0.07
    kick_lateral_tolerance: float = 0.22
    follow_distance: float = 0.32
    kick_distance: float = 0.38  # 球心到狗刚体原点，需现场标定。
    min_distance: float = 0.20
    kick_speed: float = 0.90   # 触球前的推进速度（2026-11-17 提速：0.55 → 0.90）
    kick_duration: float = 0.8
    kick_travel: float = 0.20
    ball_move_distance: float = 0.15
    settle_time: float = 0.5
    ball_still_speed: float = 0.08
    assess_time: float = 1.5
    # 发球点（规则里的十字标记）：以场地中心为原点的 3x3 网格，含开球点。
    # 坐标与训练端 train/obs.py::SERVICE_POINTS 完全一致（±2.5, ±1.25）。
    # 球回到任一发球点并静止即认为裁判已发球，比赛继续。
    service_point_x: float = 2.5
    service_point_y: float = 1.25
    service_point_tolerance: float = 0.40
    resume_settle_time: float = 0.40   # 球在发球点静止多久才放行
    resume_timeout: float = 25.0       # 兜底：暂停过久且球已静止在场内就放行
    resume_dog_speed: float = 0.15     # 裁判还在搬运时不要恢复（m/s）
    max_run_time: float = 600.0        # 复赛半场 4min，留足余量
    max_kicks: int = 200
    field_min_x: float = -3.0
    field_max_x: float = 3.0
    field_min_y: float = -2.0
    field_max_y: float = 2.0
    field_center_x: float = 0.0
    field_center_y: float = 0.0
    field_yaw: float = 0.0

    def validate(self):
        """拒绝无效坐标、速度和互相矛盾的几何参数。"""
        if any(not math.isfinite(v) for v in self):
            raise ValueError('参数必须为有限数值')
        signed = {'goal_x', 'goal_y', 'goal_yaw', 'field_min_x',
                  'field_max_x', 'field_min_y', 'field_max_y',
                  'field_center_x', 'field_center_y', 'field_yaw'}
        if any(v <= 0 for k, v in self._asdict().items() if k not in signed):
            raise ValueError('距离、时间、速度和次数必须为正数')
        if not (self.min_distance < self.kick_distance < self.ball_clearance <
                self.behind_distance <= 1.5):
            raise ValueError('需要 min_distance < kick_distance < '
                             'ball_clearance < behind_distance <= 1.5')
        if self.approach_speed > 1.0 or self.kick_speed > 1.2:
            raise ValueError('approach_speed <= 1.0，kick_speed <= 1.2 m/s（协议上限 1.5）')
        if self.turn_speed > 1.0 or self.angle_tolerance > 0.5:
            raise ValueError('turn_speed <= 1.0，angle_tolerance <= 0.5')
        if self.turn_response_grace > 2.0:
            raise ValueError('turn_response_grace <= 2 秒')
        if self.kick_lateral_tolerance > 0.5:
            raise ValueError('kick_lateral_tolerance 不能大于 0.5m')
        if self.kick_duration > 2 or self.kick_travel > 0.4:
            raise ValueError('kick_duration <= 2 秒，kick_travel <= 0.4 米')
        if type(self.max_kicks) is not int or self.max_kicks < 1:
            raise ValueError('max_kicks 必须是正整数')
        if self.goal_width <= 2*self.ball_radius:
            raise ValueError('球门宽度必须大于球直径')
        gx = self.goal_x - self.field_center_x
        gy = self.goal_y - self.field_center_y
        fx = gx*math.cos(self.field_yaw) + gy*math.sin(self.field_yaw)
        fy = -gx*math.sin(self.field_yaw) + gy*math.cos(self.field_yaw)
        if not (self.field_min_x < fx < self.field_max_x and
                self.field_min_y < fy < self.field_max_y):
            raise ValueError('球门中心必须位于设定的场地范围内')
        if self.position_tolerance >= self.behind_distance - self.ball_clearance:
            raise ValueError('position_tolerance 太大，无法稳定进入球后位置')
        if self.resume_settle_time > self.resume_timeout:
            raise ValueError('resume_settle_time 不能大于 resume_timeout')


class Pose(NamedTuple):
    """接收时间和源时间戳分别保存，不要求远程时钟与本机同步。"""

    x: float
    y: float
    yaw: float
    received: float
    frame: str
    stamp: float
    sequence: int


class Controller:
    """持续闭环；进球、场外、次数/时间上限或人工故障才进入终态。"""

    ZERO = (0.0, 0.0, 0.0)

    def __init__(self, config=Config()):
        config.validate()
        self.cfg = config
        self.dog = None
        self.ball = None
        self.state = 'WAIT_POSE'
        self.reason = '等待狗和球的连续同步位姿'
        self.started_at = None
        self.settle_at = None
        self.kick_at = None
        self.kick_origin = None
        self.ball_origin = None
        self.kick_yaw = None
        self.assess_at = None
        self.kicks = 0
        self.goals = 0
        self.distance = None
        self.ball_speed = math.inf
        self.dog_speed = math.inf
        self.ball_velocity = (0.0, 0.0)
        self.paused = False
        self.pause_since = None
        self.service_since = None
        self.resumes = 0
        self.confirmed = 0
        self.confirm_pair = (0, 0)
        self.sequence = 0
        self.route_side = None
        self.scored = False
        self.target_yaw = None
        self.turn_started = None
        self.turn_checkpoint = None
        self.turn_checkpoint_yaw = None
        self.turn_sign = None
        self.turn_origin = None
        self.turn_response_until = None

    @property
    def finished(self):
        """终态不自动重新启动。"""
        return self.state in ('GOAL', 'STOPPED')

    def stop(self, reason):
        """姿态故障或人工退出时锁定。"""
        if not self.finished:
            self.state, self.reason = 'STOPPED', reason

    def wait_for_pose(self, reason):
        """断流立即停车；恢复后重新绕球，不续发中断的前冲。"""
        if self.finished:
            return
        self.state, self.reason = 'WAIT_POSE', reason
        self.confirmed = 0
        self.settle_at = None
        self.route_side = None

    def service_points(self):
        """规则里的发球点：以场地中心为原点的 3x3 网格（含开球点）。

        与训练端 `train/obs.py::SERVICE_POINTS`（±2.5, ±1.25）一致；
        坐标按场地朝向旋转到世界系。
        """
        c = self.cfg
        co, si = math.cos(c.field_yaw), math.sin(c.field_yaw)
        points = []
        for fx in (-c.service_point_x, 0.0, c.service_point_x):
            for fy in (-c.service_point_y, 0.0, c.service_point_y):
                points.append((c.field_center_x + fx*co - fy*si,
                               c.field_center_y + fx*si + fy*co))
        return tuple(points)

    def ball_at_service_point(self, x, y, tolerance=None):
        """球是否已被放到某个发球点附近。"""
        tol = (self.cfg.service_point_tolerance
               if tolerance is None else tolerance)
        return any(math.hypot(x - px, y - py) <= tol
                   for px, py in self.service_points())

    def ball_at_restart_region(self, x, y, tolerance=0.90):
        """球是否在靠近边界的随机重发区域内。

        裁判实际发球点不一定落在九宫格标记上；允许球停在距任一边界
        ``tolerance`` 米以内的位置。门口中路不算端线危险区。
        """
        fx, fy = self.field_coordinates(x, y)
        near_side = (fy <= self.cfg.field_min_y + tolerance or
                     fy >= self.cfg.field_max_y - tolerance)
        near_end = (fx <= self.cfg.field_min_x + tolerance or
                    fx >= self.cfg.field_max_x - tolerance)
        outside_goal_mouth = abs(fy) > self.cfg.goal_width / 2.0
        return near_side or (near_end and outside_goal_mouth)

    def pause_play(self, now, reason):
        """进球/出界/死区：停车等裁判发球，不结束比赛。"""
        if self.finished:
            return
        self.paused = True
        self.state, self.reason = 'WAIT_PLAY', reason
        if self.pause_since is None:
            self.pause_since = now
        self.confirmed = 0
        self.settle_at = None
        self.route_side = None

    def _resume_ready(self, now):
        """球回到发球点并静止、狗也静止后，允许恢复比赛。"""
        c, b, d = self.cfg, self.ball, self.dog
        if b is None or d is None or not b.frame or b.frame != d.frame:
            self.service_since = None
            return False
        if not (0 <= now - b.received <= c.pose_timeout and
                0 <= now - d.received <= c.pose_timeout):
            self.service_since = None
            return False
        if not self.inside_field(b.x, b.y):
            self.service_since = None
            return False
        if self.ball_speed > c.ball_still_speed or self.dog_speed > c.resume_dog_speed:
            self.service_since = None
            return False
        # 兜底：暂停过久且球已静止在场内（裁判漏放或球自己滚回来），别再干等。
        if self.pause_since is not None and now - self.pause_since >= c.resume_timeout:
            return True
        if not self.ball_at_restart_region(b.x, b.y):
            self.service_since = None
            return False
        if self.service_since is None:
            self.service_since = now
        return now - self.service_since >= c.resume_settle_time

    def rearm(self, now, reason):
        """裁判发球完成：重新开始一整轮，不结束比赛。"""
        self.paused = False
        self.pause_since = None
        self.service_since = None
        self.state = 'WAIT_POSE'
        self.reason = reason
        self.scored = False
        self.kicks = 0
        self.confirmed = 0
        self.confirm_pair = (0, 0)
        self.started_at = now
        self.settle_at = None
        self.kick_at = None
        self.kick_origin = None
        self.ball_origin = None
        self.kick_yaw = None
        self.assess_at = None
        self.route_side = None
        self.target_yaw = None
        self.turn_started = None
        self.turn_checkpoint = None
        self.turn_checkpoint_yaw = None
        self.turn_sign = None
        self.turn_origin = None
        self.turn_response_until = None
        self.distance = None
        self.resumes += 1

    def goal_coordinates(self, x, y):
        """返回距球门线的有符号距离及沿门线的横向位置。"""
        c = self.cfg
        dx, dy = x - c.goal_x, y - c.goal_y
        co, si = math.cos(c.goal_yaw), math.sin(c.goal_yaw)
        return dx*co + dy*si, -dx*si + dy*co

    def update(self, kind, x, y, yaw, now, frame, stamp):
        """仅新消息刷新新鲜度；球越过门线且整个球处于门宽内才记进球。"""
        if kind not in ('dog', 'ball'):
            raise ValueError('kind 应为 dog 或 ball')
        previous = getattr(self, kind)
        if not all(math.isfinite(v) for v in (x, y, yaw, now, stamp)) or stamp < 0:
            setattr(self, kind, None)
            self.wait_for_pose('动捕数值无效，停车等待恢复')
            return
        c = self.cfg
        gap = previous is None or now - previous.received > c.pose_timeout
        if gap:
            self.wait_for_pose('等待连续动捕数据')
        if previous is not None and stamp > 0 and previous.stamp > 0:
            # 部分 VRPN 封装会复用 header.stamp；消息本身仍是新帧，
            # 不能因为源时间戳相同而停止刷新接收时间。
            if stamp <= previous.stamp:
                stamp = previous.stamp
        self.sequence += 1
        pose = Pose(x, y, yaw, now, frame, stamp, self.sequence)
        if kind == 'dog':
            # 只用于判断"裁判是否还在搬狗"；不影响规划。
            self.dog_speed = math.inf
            if previous is not None and not gap and previous.frame == frame:
                source_dt = stamp - previous.stamp
                dt = source_dt if source_dt > 0 else now - previous.received
                if dt > 0:
                    self.dog_speed = math.hypot(x - previous.x,
                                                y - previous.y) / dt
        if kind == 'ball':
            self.ball_speed = math.inf
            if previous is not None and not gap and previous.frame == frame:
                source_dt = stamp - previous.stamp
                dt = source_dt if source_dt > 0 else now - previous.received
                if dt > 0:
                    self.ball_velocity = ((x - previous.x)/dt, (y - previous.y)/dt)
                    self.ball_speed = math.hypot(*self.ball_velocity)
                before, side_before = self.goal_coordinates(previous.x, previous.y)
                after, side_after = self.goal_coordinates(x, y)
                # 以整个球越过门线为准，线性插值定位越线瞬间，避免低帧率漏判。
                if (self.started_at is not None and not self.paused and
                        self.state not in ('WAIT_POSE', 'WAIT_PLAY', 'WAIT_REFEREE') and
                        self.dog is not None and self.dog.frame == frame and
                        0 <= now - self.dog.received <= c.pose_timeout and
                        0 < dt <= c.pose_timeout and
                        before < c.ball_radius <= after):
                    fraction = (c.ball_radius - before) / (after - before)
                    side = side_before + fraction*(side_after - side_before)
                    if abs(side) <= c.goal_width/2 - c.ball_radius:
                        self.scored = True
        setattr(self, kind, pose)

    def poses_valid(self, now):
        """要求同一坐标系且各自接收新鲜，不要求回调同时发生。"""
        d, b, c = self.dog, self.ball, self.cfg
        return (d is not None and b is not None and bool(d.frame) and
                d.frame == b.frame and
                0 <= now - d.received <= c.pose_timeout and
                0 <= now - b.received <= c.pose_timeout)

    def field_coordinates(self, x, y):
        """Convert mocap coordinates into the configured field frame."""
        c = self.cfg
        dx, dy = x - c.field_center_x, y - c.field_center_y
        co, si = math.cos(c.field_yaw), math.sin(c.field_yaw)
        return co*dx + si*dy, -si*dx + co*dy

    def inside_field(self, x, y):
        """场地范围包括球门后的小块区域，供越线判定。"""
        c = self.cfg
        dx, dy = x - c.field_center_x, y - c.field_center_y
        fx = dx*math.cos(c.field_yaw) + dy*math.sin(c.field_yaw)
        fy = -dx*math.sin(c.field_yaw) + dy*math.cos(c.field_yaw)
        return c.field_min_x <= fx <= c.field_max_x and c.field_min_y <= fy <= c.field_max_y

    def drive_to(self, x, y, yaw, speed):
        """近球时世界速度转机身速度，减速对准；偏航容差内不再转动。"""
        d, c = self.dog, self.cfg
        dx, dy = x - d.x, y - d.y
        distance = math.hypot(dx, dy)
        scale = min(speed, max(0.04, 1.5*distance)) / max(distance, 1e-9)
        wx, wy = dx*scale, dy*scale
        vx = math.cos(d.yaw)*wx + math.sin(d.yaw)*wy
        vy = -math.sin(d.yaw)*wx + math.cos(d.yaw)*wy
        wz = self.turn_rate(yaw)
        # 预测短时间运动仍须在场内；边界不足时保持停止而不是穿越边界。
        if not self.inside_field(d.x + wx*0.5, d.y + wy*0.5):
            self.stop('绕球路径接近场地边界，请调整场地范围或球的位置')
            return self.ZERO
        return vx, vy, wz

    def turn_rate(self, target_yaw):
        """使用最短角误差；容差内停止，容差外避免指令小到无法转动。"""
        self.target_yaw = target_yaw
        error = wrap_angle(target_yaw - self.dog.yaw)
        if abs(error) <= self.cfg.angle_tolerance:
            return 0.0
        rate = min(self.cfg.turn_speed, max(0.20, 1.5*abs(error)))
        return math.copysign(rate, error)

    def navigate_to(self, x, y):
        """绕球阶段先朝向路径点再前进，避免一边横移一边强行朝向球门。"""
        dx, dy = x - self.dog.x, y - self.dog.y
        heading = math.atan2(dy, dx)
        error = wrap_angle(heading - self.dog.yaw)
        wz = self.turn_rate(heading)
        if abs(error) > 0.45:
            return 0.0, 0.0, wz
        # 将导航速度限制在机身正前方，横移仅用于接近球时的小幅修正。
        # 提前减速，避免有执行延迟时越过准备点，再追逐身后的路径点。
        vx = min(self.cfg.approach_speed, max(0.04, 0.8*math.hypot(dx, dy)))
        vx *= max(0.0, math.cos(error))
        if not self.inside_field(self.dog.x + vx*math.cos(self.dog.yaw)*0.5,
                                 self.dog.y + vx*math.sin(self.dog.yaw)*0.5):
            self.stop('前进路径接近场地边界')
            return self.ZERO
        return vx, 0.0, wz

    def command(self, now):
        """检查所有转向指令的反馈，反向/无进展时结束控制。"""
        velocity = self._command(now)
        vx, vy, wz = velocity
        if self.finished or abs(wz) < 1e-6:
            self.turn_started = None
            return velocity
        direction = math.copysign(1.0, wz)
        if self.turn_started is None:
            self.turn_started = self.turn_checkpoint = now
            self.turn_checkpoint_yaw = self.dog.yaw
            self.turn_sign = direction
            self.turn_origin = (self.dog.x, self.dog.y)
            self.turn_response_until = now + self.cfg.turn_response_grace
        elif direction != self.turn_sign:
            self.turn_checkpoint = now
            self.turn_checkpoint_yaw = self.dog.yaw
            self.turn_sign = direction
            self.turn_response_until = now + self.cfg.turn_response_grace
        if math.hypot(self.dog.x-self.turn_origin[0],
                      self.dog.y-self.turn_origin[1]) >= 0.15:
            self.turn_started = now
            self.turn_origin = (self.dog.x, self.dog.y)
        # 换向后仍可能沿上一条指令转动。宽限期内持续更新基准，
        # 不把旧方向残余转角累计成新指令的反向反馈。
        if now < self.turn_response_until:
            self.turn_checkpoint = now
            self.turn_checkpoint_yaw = self.dog.yaw
            if now - self.turn_started >= 12.0:
                self.stop('连续原地转向超过 12 秒：检查狗刚体朝向与 dog_yaw_offset')
            return self.ZERO if self.finished else velocity
        delta = wrap_angle(self.dog.yaw - self.turn_checkpoint_yaw)*direction
        if now - self.turn_checkpoint >= 0.4 and delta < -0.10:
            self.stop('等待转向响应后仍持续反向：检查执行延迟、控制冲突和姿态坐标')
        elif delta >= 0.04:
            self.turn_checkpoint = now
            self.turn_checkpoint_yaw = self.dog.yaw
        elif now - self.turn_checkpoint >= 3.0:
            self.stop('转向 3 秒但动捕偏航无进展：检查狗刚体姿态是否更新')
        if now - self.turn_started >= 12.0:
            self.stop('连续原地转向超过 12 秒：检查狗刚体朝向与 dog_yaw_offset')
        return self.ZERO if self.finished else velocity

    def route_target(self, target_x, target_y):
        """直线会穿过球周围的保留空间时，沿固定一侧绕行。"""
        d, b, c = self.dog, self.ball, self.cfg
        rx, ry = d.x - b.x, d.y - b.y
        radius = math.hypot(rx, ry)
        if radius < c.ball_clearance:
            if radius < 1e-6:
                rx, ry = target_x - b.x, target_y - b.y
                radius = math.hypot(rx, ry)
            return (b.x + rx/radius*(c.ball_clearance + 0.10),
                    b.y + ry/radius*(c.ball_clearance + 0.10))
        tx, ty = target_x - d.x, target_y - d.y
        t = max(0.0, min(1.0, -(rx*tx + ry*ty) / max(tx*tx + ty*ty, 1e-9)))
        if math.hypot(rx + t*tx, ry + t*ty) >= c.ball_clearance:
            self.route_side = None
            return target_x, target_y
        angle = math.atan2(ry, rx)
        target_angle = math.atan2(target_y - b.y, target_x - b.x)
        delta = wrap_angle(target_angle - angle)
        if self.route_side is None:
            self.route_side = 1 if delta >= 0 else -1
        waypoint_angle = angle + self.route_side*0.4
        route_radius = c.ball_clearance + 0.15
        return (b.x + route_radius*math.cos(waypoint_angle),
                b.y + route_radius*math.sin(waypoint_angle))

    def _command(self, now):
        """闭环决策，每周期返回 (前进, 横移, 转向)，无需阻塞等待。"""
        c = self.cfg
        if self.finished:
            return self.ZERO
        if self.paused:
            if self._resume_ready(now):
                self.rearm(now, '球已回到发球点，重新开始比赛')
            else:
                self.state = 'WAIT_PLAY'
            return self.ZERO
        if self.started_at is not None and now - self.started_at >= c.max_run_time:
            self.stop('任务达到时间上限')
            return self.ZERO
        if not self.poses_valid(now):
            self.wait_for_pose('定位断流/不同步/坐标系不一致，停车等待')
            return self.ZERO
        d, b = self.dog, self.ball
        if self.scored:
            self.goals += 1
            self.pause_play(now, f'进球（本场第 {self.goals} 个），等待裁判发球')
            return self.ZERO
        if not self.inside_field(d.x, d.y):
            self.stop('狗已超出设定场地')
            return self.ZERO
        if not self.inside_field(b.x, b.y):
            self.pause_play(now, '球已出界，等待裁判就近发球')
            return self.ZERO
        if self.goal_coordinates(b.x, b.y)[0] >= c.ball_radius:
            self.pause_play(now, '球在球门线后，等待裁判发球')
            return self.ZERO
        if self.state == 'WAIT_POSE':
            pair = (d.sequence, b.sequence)
            if pair[0] != self.confirm_pair[0] and pair[1] != self.confirm_pair[1]:
                self.confirmed += 1
                self.confirm_pair = pair
            if self.confirmed < 3:
                return self.ZERO
            if self.started_at is None:
                self.started_at = now
            self.state = 'POSITION'
        dx, dy = b.x - d.x, b.y - d.y
        self.distance = math.hypot(dx, dy)
        gx, gy = c.goal_x - b.x, c.goal_y - b.y
        goal_distance = math.hypot(gx, gy)
        if goal_distance < 1e-6:
            self.stop('球位于球门中心，无法确定射门方向')
            return self.ZERO
        ux, uy = gx/goal_distance, gy/goal_distance
        shot_yaw = math.atan2(uy, ux)
        heading_error = wrap_angle(shot_yaw - d.yaw)
        # 球在狗前方的投影以及球与射门轴之间的横向误差。
        forward = dx*ux + dy*uy
        lateral = -dx*uy + dy*ux
        if self.state == 'KICK':
            travel = math.hypot(d.x - self.kick_origin[0], d.y - self.kick_origin[1])
            moved = math.hypot(b.x - self.ball_origin[0], b.y - self.ball_origin[1])
            if (now - self.kick_at >= c.kick_duration or travel >= c.kick_travel or
                    moved >= c.ball_move_distance or self.distance <= c.min_distance):
                self.state, self.reason = 'ASSESS', '停止前冲，观察球的实际运动'
                self.assess_at = now
                return self.ZERO
            if abs(wrap_angle(d.yaw - self.kick_yaw)) > 2*c.angle_tolerance:
                self.state, self.reason = 'ASSESS', '前冲朝向偏离，停止并重新规划'
                self.assess_at = now
                return self.ZERO
            bearing = wrap_angle(math.atan2(dy, dx) - self.kick_yaw)
            if abs(bearing) > 0.35:
                self.state, self.reason = 'ASSESS', '球偏离前冲方向，观察后重新规划'
                self.assess_at = now
                return self.ZERO
            return self.drive_to(d.x + math.cos(self.kick_yaw),
                                 d.y + math.sin(self.kick_yaw),
                                 self.kick_yaw, c.kick_speed)
        if self.state == 'ASSESS':
            if now - self.assess_at < c.assess_time:
                return self.ZERO
            # 球若仍沿球门方向滚动，继续跟球，避免每次触球后退回初始准备点。
            gx, gy = c.goal_x - b.x, c.goal_y - b.y
            goal_norm = max(math.hypot(gx, gy), 1e-9)
            toward_goal = (self.ball_velocity[0]*gx + self.ball_velocity[1]*gy) / goal_norm
            self.state = 'FOLLOW' if toward_goal > c.ball_still_speed else 'POSITION'
            self.settle_at = None
        if self.state == 'FOLLOW':
            gx, gy = c.goal_x - b.x, c.goal_y - b.y
            goal_norm = max(math.hypot(gx, gy), 1e-9)
            ux, uy = gx/goal_norm, gy/goal_norm
            forward = (b.x-d.x)*ux + (b.y-d.y)*uy
            lateral = -(b.x-d.x)*uy + (b.y-d.y)*ux
            shot_yaw = math.atan2(uy, ux)
            if (self.ball_speed > c.ball_still_speed and forward > 0 and
                    abs(lateral) <= c.kick_lateral_tolerance and
                    abs(wrap_angle(shot_yaw-d.yaw)) <= 0.4):
                return self.drive_to(b.x-ux*c.follow_distance,
                                     b.y-uy*c.follow_distance,
                                     shot_yaw, c.approach_speed)
            self.state = 'POSITION'
        if self.kicks >= c.max_kicks:
            self.stop('已达到射门次数上限，尚未确认进球')
            return self.ZERO
        stage_distance = c.behind_distance
        # 后续射门保持当前进攻侧，只退到安全触球距离，不再绕回 0.70m。
        if self.kicks > 0 and forward > 0 and abs(lateral) <= c.kick_lateral_tolerance:
            stage_distance = min(c.behind_distance,
                                 max(c.kick_distance + 0.06, self.distance + 0.05))
        stage_x, stage_y = b.x - ux*stage_distance, b.y - uy*stage_distance
        if not self.inside_field(stage_x, stage_y):
            self.stop('球后射门准备点在场外，无法组织射门')
            return self.ZERO
        if self.state == 'POSITION':
            self.reason = '绕到球后方，建立朝向球门的射门路线'
            # 已进入球后射门走廊就朝球门对齐，无需精确踩中准备点。
            # 否则接近/越过准备点时 atan2 会骤变，使狗背对球旋转。
            in_shot_corridor = (
                c.kick_distance <= forward <= stage_distance + 0.15 and
                abs(lateral) <= c.kick_lateral_tolerance)
            if (not in_shot_corridor and
                    math.hypot(stage_x - d.x, stage_y - d.y) > c.position_tolerance):
                target = self.route_target(stage_x, stage_y)
                # 已经在球后且面向球门时，允许倒退到安全距离，避免触球后转身。
                if (self.kicks > 0 and forward > 0 and
                        abs(lateral) <= c.kick_lateral_tolerance and
                        abs(heading_error) <= 0.4):
                    return self.drive_to(*target, shot_yaw, c.approach_speed)
                return self.navigate_to(*target)
            self.state = 'ALIGN'
        if forward <= -0.05 or abs(lateral) > c.kick_lateral_tolerance:
            self.state, self.reason = 'POSITION', '球偏离射门轴，重新绕球'
            self.settle_at = None
            return self.ZERO
        if abs(heading_error) > c.angle_tolerance:
            self.state, self.reason = 'ALIGN', '调整机身朝向球门'
            self.settle_at = None
            return 0.0, 0.0, self.turn_rate(shot_yaw)
        if self.distance < c.min_distance:
            self.state, self.reason = 'POSITION', '与球过近，退出并重新组织'
            self.settle_at = None
            return self.ZERO
        if forward > c.kick_distance + 0.05 or abs(lateral) > c.kick_lateral_tolerance:
            self.state, self.reason = 'APPROACH', '保持射门方向，慢速靠近球'
            self.settle_at = None
            # 目标略小于触发距离，避免比例控制永远渐近到达不了阈值。
            return self.drive_to(b.x - ux*c.kick_distance,
                                 b.y - uy*c.kick_distance,
                                 shot_yaw, c.approach_speed)
        self.state, self.reason = 'SETTLE', '等待球静止并稳定对准'
        if self.ball_speed > c.ball_still_speed:
            self.settle_at = None
            return self.ZERO
        if self.settle_at is None:
            self.settle_at = now
        if now - self.settle_at >= c.settle_time:
            self.state, self.reason = 'KICK', '向球门方向前冲触球'
            self.kicks += 1
            self.kick_at = now
            self.kick_origin = (d.x, d.y)
            self.ball_origin = (b.x, b.y)
            self.kick_yaw = shot_yaw
            return self.drive_to(d.x + ux, d.y + uy, shot_yaw, c.kick_speed)
        return self.ZERO
