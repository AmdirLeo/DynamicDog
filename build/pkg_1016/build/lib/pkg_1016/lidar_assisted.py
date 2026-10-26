"""超声波与 LaserScan 联合避障，纯 Python 实现，兼容 Python 3.6。

超声波独立判断前方停车和恢复，雷达左右前侧用于选择并保持转向方向。
前方雷达扇区仅用于诊断，不参与运动决策；雷达不可用时禁止转向。
只做安装偏航角修正，不实现 TF 平移变换、全局规划或完整机身碰撞检测。
"""

import math
from collections import namedtuple

from pkg_1016.topic_logic import AvoidanceController


class LidarConfig(namedtuple('LidarConfigBase', (
        'scan_timeout', 'turn_clearance', 'direction_margin',
        'min_valid_points', 'min_valid_fraction', 'min_coverage',
        'scan_yaw_offset', 'scan_frame_id'))):
    """雷达辅助参数；距离相对于雷达测量原点，阈值需实机确认。"""

    __slots__ = ()

    def __new__(cls, scan_timeout=0.50, turn_clearance=0.40,
                direction_margin=0.10, min_valid_points=5,
                min_valid_fraction=0.50, min_coverage=0.80,
                scan_yaw_offset=0.0, scan_frame_id='laser_frame'):
        """声明所需覆盖和有效点比例，避免用零星点推断整个方向。"""
        self = super().__new__(
            cls, scan_timeout, turn_clearance, direction_margin,
            min_valid_points, min_valid_fraction, min_coverage,
            scan_yaw_offset, scan_frame_id)
        if any(not math.isfinite(v) or v <= 0 for v in (
                scan_timeout, turn_clearance)):
            raise ValueError('scan_timeout 和 turn_clearance 必须是有限正数')
        if not math.isfinite(direction_margin) or direction_margin < 0:
            raise ValueError('direction_margin 必须是有限非负数')
        if any(not math.isfinite(v) or not 0 < v <= 1 for v in (
                min_valid_fraction, min_coverage)):
            raise ValueError('min_valid_fraction 和 min_coverage 需在 (0, 1]')
        if type(min_valid_points) is not int or min_valid_points < 1:
            raise ValueError('min_valid_points 必须是正整数')
        if not math.isfinite(scan_yaw_offset):
            raise ValueError('scan_yaw_offset 必须是有限弧度值')
        if not isinstance(scan_frame_id, str) or not scan_frame_id.strip():
            raise ValueError('scan_frame_id 不能为空')
        return self


def control_sectors(ranges, angle_min, angle_max, angle_increment,
                    range_min, range_max, config):
    """前方 ±30°，左前 30～90°，右前 −90～−30°，取有效最小距离。

    coverage 表示角度采样覆盖，valid_fraction 表示其中有效距离的比例。
    侧方即使覆盖不足，检测到的近距离仍可阻止转向；未知侧不能用于转向。
    """
    sectors = {
        name: {'distance_m': None, 'sample_count': 0, 'valid_points': 0,
               'coverage': 0.0, 'valid_fraction': 0.0, 'known': False}
        for name in ('front', 'left', 'right')
    }
    report = {'status': 'INVALID', 'sectors': sectors}
    if (not all(math.isfinite(v) for v in (
            angle_min, angle_max, angle_increment, range_min, range_max))
            or angle_increment == 0 or not 0 <= range_min < range_max
            or len(ranges) < 2):
        return report
    span = (len(ranges) - 1) * angle_increment
    if (not math.isfinite(span) or abs(span) > 2 * math.pi + 1e-5
            or abs(angle_min + span - angle_max)
            > max(1e-5, abs(angle_increment) * 0.1)):
        return report
    for i, distance in enumerate(ranges):
        angle = (angle_min % (2 * math.pi) + i * angle_increment
                 + config.scan_yaw_offset % (2 * math.pi))
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        # 允许浮点数产生的 ±90° 端点误差。
        if abs(angle) > math.pi / 2 + 1e-6:
            continue
        name = ('left' if angle >= math.pi / 6 - 1e-9 else
                'right' if angle < -math.pi / 6 - 1e-9 else 'front')
        sector = sectors[name]
        sector['sample_count'] += 1
        if (not math.isfinite(distance)
                or not range_min <= distance <= range_max):
            continue
        sector['valid_points'] += 1
        previous = sector['distance_m']
        if previous is None or distance < previous:
            sector['distance_m'] = float(distance)
    expected = (math.pi / 3) / abs(angle_increment)
    for sector in sectors.values():
        total = sector['sample_count']
        sector['coverage'] = min(1.0, total / expected)
        sector['valid_fraction'] = (
            sector['valid_points'] / total if total else 0.0)
        sector['known'] = (
            sector['coverage'] >= config.min_coverage
            and sector['valid_fraction'] >= config.min_valid_fraction
            and sector['valid_points'] >= config.min_valid_points)
    report['status'] = 'OK'
    return report


class LidarAssistedController(AvoidanceController):
    """超声波负责前方；雷达仅在需要转向时判断侧方空间。"""

    def __init__(self, config=None, lidar_config=None):
        """选定的转向方向保持到前进恢复，选定侧被挡则停下等待。"""
        super().__init__(config)
        self.lidar_config = lidar_config or LidarConfig()
        if self.lidar_config.turn_clearance <= self.config.emergency_distance:
            raise ValueError('turn_clearance 必须大于 emergency_distance')
        self.scan_received = None
        self.scan_frame = ''
        self.scan_report = None
        self.turn_direction = None
        self.reason = '等待超声波数据'

    def update_range(self, distance, minimum, maximum, now):
        """超声波断流或无效后重新确认前方，不沿用旧前进状态。"""
        gap = (self.last_received is None
               or now - self.last_received > self.config.sensor_timeout)
        valid = super().update_range(distance, minimum, maximum, now)
        if gap or not valid:
            self._enter('WAIT_SENSOR', now)
        elif distance <= self.config.emergency_distance:
            self._enter('TOO_CLOSE', now)
        return valid

    def update_scan(self, ranges, angle_min, angle_max, angle_increment,
                    range_min, range_max, frame_id, now):
        """更新侧方观测；异常扫描打断转向，但不影响超声波空旷计数。"""
        config = self.lidar_config
        gap = (self.scan_received is None
               or now - self.scan_received > config.scan_timeout)
        self.scan_received = now
        self.scan_frame = frame_id
        self.scan_report = control_sectors(
            ranges, angle_min, angle_max, angle_increment,
            range_min, range_max, config)
        if frame_id != config.scan_frame_id:
            self.scan_report['status'] = 'FRAME_MISMATCH'
        sectors = self.scan_report['sectors']
        near = any(s['distance_m'] is not None
                   and s['distance_m'] <= self.config.emergency_distance
                   for s in (sectors['left'], sectors['right']))
        # 记录控制周期之间的断流或受阻，恢复后先制动再转向。
        if self.state in ('BRAKE', 'TURN'):
            if gap or self.scan_report['status'] != 'OK':
                self._enter('WAIT_SCAN', now)
            elif near:
                self._enter('LIDAR_TOO_CLOSE', now)
            elif self.turn_direction is not None:
                side = sectors['left' if self.turn_direction > 0 else 'right']
                if (not side['known']
                        or side['distance_m'] < config.turn_clearance):
                    self._enter('BLOCKED', now)

    def _stop(self, state, now, reason, reset=False):
        if reset:
            self.clear_count = 0
        self.reason = reason
        self._enter(state, now)
        return (0.0, 0.0)

    def command(self, now):
        """先按超声波判断前方；仅需转向时检查雷达，不限制转向时长。"""
        c = self.config
        lc = self.lidar_config
        if (self.last_received is None or self.distance is None
                or now - self.last_received > c.sensor_timeout):
            return self._stop('WAIT_SENSOR', now, '超声波无效或超时', True)
        if self.distance <= c.emergency_distance:
            return self._stop('TOO_CLOSE', now, '超声波距离过近', True)
        if self.clear_count >= c.clear_samples:
            self.turn_direction = None
            self.reason = '超声波连续确认前方空旷'
            self._enter('FORWARD', now)
            return (c.forward_speed, 0.0)
        if self.state == 'FORWARD' and self.distance > c.stop_distance:
            self.reason = '前进中，超声波距离高于停车阈值'
            return (c.forward_speed, 0.0)
        if self.distance >= c.clear_distance:
            return self._stop('WAIT_CLEAR', now, '等待超声波连续空旷确认')
        # 此处需要转向：雷达缺失、过期或无可用侧方时停车等待。
        if (self.scan_received is None
                or now - self.scan_received > lc.scan_timeout):
            return self._stop('WAIT_SCAN', now, '需要转向，但雷达未接收或超时')
        if self.scan_report['status'] != 'OK':
            return self._stop(
                'WAIT_SCAN', now, '需要转向，但雷达异常：' + self.scan_report['status'])
        sectors = self.scan_report['sectors']
        if any(s['distance_m'] is not None
               and s['distance_m'] <= c.emergency_distance
               for s in (sectors['left'], sectors['right'])):
            return self._stop(
                'LIDAR_TOO_CLOSE', now, '雷达侧方距离过近，暂停转向')
        available = {
            side: sectors[side]['distance_m']
            for side in ('left', 'right')
            if sectors[side]['known']
            and sectors[side]['distance_m'] >= lc.turn_clearance
        }
        if self.turn_direction is not None:
            side = 'left' if self.turn_direction > 0 else 'right'
            if side not in available:
                return self._stop(
                    'BLOCKED', now, '选定转向侧过近或缺测，保持方向并等待')
        elif not available:
            return self._stop('BLOCKED', now, '左右均无已确认可转向空间')
        if self.state not in ('BRAKE', 'TURN'):
            return self._stop('BRAKE', now, '转向前制动')
        if self.state == 'BRAKE':
            if now - self.state_since < c.brake_duration:
                return (0.0, 0.0)
            if self.turn_direction is None:
                if len(available) == 2:
                    delta = available['left'] - available['right']
                    if abs(delta) <= lc.direction_margin:
                        self.turn_direction = 1 if c.turn_speed > 0 else -1
                    else:
                        self.turn_direction = 1 if delta > 0 else -1
                else:
                    self.turn_direction = 1 if 'left' in available else -1
            self._enter('TURN', now)
        self.reason = ('保持向左转' if self.turn_direction > 0 else '保持向右转')
        return (0.0, self.turn_direction * abs(c.turn_speed))
