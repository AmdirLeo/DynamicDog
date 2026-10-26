"""超声波与 LaserScan 联合避障，纯 Python 实现，兼容 Python 3.6。

前方雷达距离参与停车和恢复，左右前侧距离用于选择并保持转向方向。
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
    即使覆盖不足，检测到的近距离仍可用于停车；未知方向不能用于放行。
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
    """任一前向传感器近障即制动；两者确认空旷才恢复前进。"""

    def __init__(self, config=None, lidar_config=None):
        """选定的转向方向保持到前进恢复，选定侧被挡则停下等待。"""
        super().__init__(config)
        self.lidar_config = lidar_config or LidarConfig()
        if self.lidar_config.turn_clearance <= self.config.emergency_distance:
            raise ValueError('turn_clearance 必须大于 emergency_distance')
        self.scan_received = None
        self.scan_frame = ''
        self.scan_report = None
        self.scan_clear_count = 0
        self.turn_direction = None
        self.reason = '等待超声波与雷达数据'

    def update_range(self, distance, minimum, maximum, now):
        """超声波断流后必须重新确认雷达空旷，拒绝沿用旧计数。"""
        gap = (self.last_received is None
               or now - self.last_received > self.config.sensor_timeout)
        valid = super().update_range(distance, minimum, maximum, now)
        if gap or not valid or distance < self.config.clear_distance:
            self.scan_clear_count = 0
        if gap or not valid:
            self._enter('WAIT_SENSOR', now)
        elif distance <= self.config.emergency_distance:
            self._enter('TOO_CLOSE', now)
        return valid

    def update_scan(self, ranges, angle_min, angle_max, angle_increment,
                    range_min, range_max, frame_id, now):
        """新扫描帧到来才增加确认次数；不使用未知坐标系的扫描。"""
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
        front = sectors['front']
        near = any(s['distance_m'] is not None
                   and s['distance_m'] <= self.config.emergency_distance
                   for s in sectors.values())
        clear = (self.scan_report['status'] == 'OK'
                 and front['known'] and not near
                 and front['distance_m'] >= self.config.clear_distance)
        if gap or not clear:
            self.scan_clear_count = 0
            self.clear_count = 0
        if (gap or self.scan_report['status'] != 'OK'
                or not front['known']):
            self._enter('WAIT_SCAN', now)
        elif near:
            self._enter('LIDAR_TOO_CLOSE', now)
        if clear:
            self.scan_clear_count += 1

    def _stop(self, state, now, reason, reset=False):
        if reset:
            self.clear_count = 0
            self.scan_clear_count = 0
        self.reason = reason
        self._enter(state, now)
        return (0.0, 0.0)

    def command(self, now):
        """执行融合判断，不因转向持续时间而退出或锁死。"""
        c = self.config
        lc = self.lidar_config
        if (self.last_received is None or self.distance is None
                or now - self.last_received > c.sensor_timeout):
            return self._stop('WAIT_SENSOR', now, '超声波无效或超时', True)
        if self.distance <= c.emergency_distance:
            return self._stop('TOO_CLOSE', now, '超声波距离过近', True)
        if (self.scan_received is None
                or now - self.scan_received > lc.scan_timeout):
            return self._stop('WAIT_SCAN', now, '雷达未接收或超时', True)
        if self.scan_report['status'] != 'OK':
            return self._stop(
                'WAIT_SCAN', now, self.scan_report['status'], True)
        sectors = self.scan_report['sectors']
        if any(s['distance_m'] is not None
               and s['distance_m'] <= c.emergency_distance
               for s in sectors.values()):
            return self._stop(
                'LIDAR_TOO_CLOSE', now, '雷达前方或侧方距离过近', True)
        front = sectors['front']
        if not front['known']:
            return self._stop('WAIT_SCAN', now, '前方雷达覆盖或有效点不足', True)
        if (self.clear_count >= c.clear_samples
                and self.scan_clear_count >= c.clear_samples):
            self.turn_direction = None
            self.reason = '超声波与雷达连续确认前方空旷'
            self._enter('FORWARD', now)
            return (c.forward_speed, 0.0)
        if (self.state == 'FORWARD' and self.distance > c.stop_distance
                and front['distance_m'] > c.stop_distance):
            self.reason = '前进中，两项前向距离均高于停车阈值'
            return (c.forward_speed, 0.0)
        if (self.distance >= c.clear_distance
                and front['distance_m'] >= c.clear_distance):
            return self._stop('WAIT_CLEAR', now, '等待两项测距的连续空旷确认')
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
