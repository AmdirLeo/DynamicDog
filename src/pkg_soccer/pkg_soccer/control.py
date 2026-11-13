"""动捕足球智能体：绕到球后、对门、触球、观察结果、重新组织射门。

纯 Python，所有位置使用 Z-up 世界坐标（米），速度使用机身坐标。
只规划狗与球的相对运动；场地应空旷，不包含其他机器人/障碍物避让。
KICK 使用课程 303 步态前冲触球，不假定存在专用抬腿动作编号。
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
    approach_speed: float = 0.12
    turn_speed: float = 0.4
    angle_tolerance: float = 0.10
    behind_distance: float = 0.70
    ball_clearance: float = 0.48
    position_tolerance: float = 0.07
    kick_distance: float = 0.38  # 球心到狗刚体原点，需现场标定。
    min_distance: float = 0.20
    kick_speed: float = 0.25
    kick_duration: float = 0.8
    kick_travel: float = 0.20
    ball_move_distance: float = 0.15
    settle_time: float = 0.5
    ball_still_speed: float = 0.08
    assess_time: float = 1.5
    max_run_time: float = 120.0
    max_kicks: int = 8
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
        if self.approach_speed > 0.3 or self.kick_speed > 0.3:
            raise ValueError('前进速度上限为 0.3 m/s')
        if self.turn_speed > 0.8 or self.angle_tolerance > 0.3:
            raise ValueError('turn_speed <= 0.8，angle_tolerance <= 0.3')
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
        self.distance = None
        self.ball_speed = math.inf
        self.confirmed = 0
        self.confirm_pair = (0, 0)
        self.sequence = 0
        self.route_side = None
        self.scored = False

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
            if stamp < previous.stamp:
                self.stop('动捕时间戳倒退，请检查 VRPN 后重启')
                return
            if stamp == previous.stamp:
                return
        self.sequence += 1
        pose = Pose(x, y, yaw, now, frame, stamp, self.sequence)
        if kind == 'ball':
            self.ball_speed = math.inf
            if previous is not None and not gap and previous.frame == frame:
                dt = (stamp - previous.stamp if stamp > 0 and previous.stamp > 0
                      else now - previous.received)
                if dt > 0:
                    self.ball_speed = math.hypot(x - previous.x, y - previous.y) / dt
                before, side_before = self.goal_coordinates(previous.x, previous.y)
                after, side_after = self.goal_coordinates(x, y)
                # 以整个球越过门线为准，线性插值定位越线瞬间，避免低帧率漏判。
                if (self.started_at is not None and self.state != 'WAIT_POSE' and
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
        """同坐标系、接收新鲜度和两个源时间戳必须同时成立。"""
        d, b, c = self.dog, self.ball, self.cfg
        return (d is not None and b is not None and bool(d.frame) and
                d.frame == b.frame and
                0 <= now - d.received <= c.pose_timeout and
                0 <= now - b.received <= c.pose_timeout)

    def inside_field(self, x, y):
        """场地范围包括球门后的小块区域，供越线判定。"""
        c = self.cfg
        dx, dy = x - c.field_center_x, y - c.field_center_y
        fx = dx*math.cos(c.field_yaw) + dy*math.sin(c.field_yaw)
        fy = -dx*math.sin(c.field_yaw) + dy*math.cos(c.field_yaw)
        return c.field_min_x <= fx <= c.field_max_x and c.field_min_y <= fy <= c.field_max_y

    def drive_to(self, x, y, yaw, speed):
        """世界速度转到机身坐标；利用 303 步态横移保持面向球门。"""
        d, c = self.dog, self.cfg
        dx, dy = x - d.x, y - d.y
        distance = math.hypot(dx, dy)
        scale = min(1.0, speed / max(distance, 1e-9))
        wx, wy = dx*scale, dy*scale
        vx = math.cos(d.yaw)*wx + math.sin(d.yaw)*wy
        vy = -math.sin(d.yaw)*wx + math.cos(d.yaw)*wy
        wz = max(-c.turn_speed, min(c.turn_speed, 1.5*wrap_angle(yaw - d.yaw)))
        # 预测短时间运动仍须在场内；边界不足时保持停止而不是穿越边界。
        if not self.inside_field(d.x + wx*0.5, d.y + wy*0.5):
            self.stop('绕球路径接近场地边界，请调整场地范围或球的位置')
            return self.ZERO
        return vx, vy, wz

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

    def command(self, now):
        """闭环决策，每周期返回 (前进, 横移, 转向)，无需阻塞等待。"""
        c = self.cfg
        if self.finished:
            return self.ZERO
        if self.started_at is not None and now - self.started_at >= c.max_run_time:
            self.stop('任务达到时间上限')
            return self.ZERO
        if not self.poses_valid(now):
            self.wait_for_pose('定位断流/不同步/坐标系不一致，停车等待')
            return self.ZERO
        d, b = self.dog, self.ball
        if self.scored:
            self.state, self.reason = 'GOAL', '球已从场内越过球门线，进球'
            return self.ZERO
        if not self.inside_field(d.x, d.y) or not self.inside_field(b.x, b.y):
            self.stop('狗或球已超出设定场地')
            return self.ZERO
        if self.goal_coordinates(b.x, b.y)[0] >= c.ball_radius:
            self.stop('球在球门线后且未确认进球，请重新摆球')
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
            if now - self.assess_at < c.assess_time or self.ball_speed > c.ball_still_speed:
                return self.ZERO
            self.state = 'POSITION'
            self.settle_at = None
        if self.kicks >= c.max_kicks:
            self.stop('已达到射门次数上限，尚未确认进球')
            return self.ZERO
        stage_x, stage_y = b.x - ux*c.behind_distance, b.y - uy*c.behind_distance
        if not self.inside_field(stage_x, stage_y):
            self.stop('球后射门准备点在场外，无法组织射门')
            return self.ZERO
        if self.state == 'POSITION':
            self.reason = '绕到球后方，建立朝向球门的射门路线'
            if math.hypot(stage_x - d.x, stage_y - d.y) > c.position_tolerance:
                target = self.route_target(stage_x, stage_y)
                return self.drive_to(*target, shot_yaw, c.approach_speed)
            self.state = 'ALIGN'
        if forward <= 0 or abs(lateral) > c.position_tolerance*1.5:
            self.state, self.reason = 'POSITION', '球偏离射门轴，重新绕球'
            self.settle_at = None
            return self.ZERO
        if abs(heading_error) > c.angle_tolerance:
            self.state, self.reason = 'ALIGN', '调整机身朝向球门'
            self.settle_at = None
            return 0.0, 0.0, max(-c.turn_speed, min(c.turn_speed, 1.5*heading_error))
        if self.distance < c.min_distance:
            self.state, self.reason = 'POSITION', '与球过近，退出并重新组织'
            self.settle_at = None
            return self.ZERO
        if forward > c.kick_distance + 0.02 or abs(lateral) > c.position_tolerance/2:
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
