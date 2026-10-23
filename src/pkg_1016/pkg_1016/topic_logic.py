"""独立于 ROS 的避障状态机与激光雷达扇区分析算法。"""

import math
from collections import namedtuple


class AvoidanceConfig(namedtuple('AvoidanceConfigBase', (
        'emergency_distance', 'stop_distance', 'clear_distance',
        'forward_speed', 'turn_speed', 'sensor_timeout', 'brake_duration',
        'max_turn_duration', 'clear_samples'))):
    """兼容 Python 3.6 的不可变配置；阈值仍需实机标定。"""

    __slots__ = ()

    def __new__(cls, emergency_distance=0.20, stop_distance=0.40,
                clear_distance=0.65, forward_speed=0.10, turn_speed=0.35,
                sensor_timeout=0.50, brake_duration=0.50,
                max_turn_duration=6.0, clear_samples=3):
        """沿用原默认值，用标准库 namedtuple 避免 dataclasses 依赖。"""
        self = super().__new__(
            cls, emergency_distance, stop_distance, clear_distance,
            forward_speed, turn_speed, sensor_timeout, brake_duration,
            max_turn_duration, clear_samples)
        self._validate()
        return self

    def _validate(self):
        """检查阈值顺序、有限性及本实验允许的低速范围。"""
        positive = (
            self.emergency_distance, self.stop_distance,
            self.clear_distance, self.forward_speed, self.sensor_timeout,
            self.brake_duration, self.max_turn_duration,
        )
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError('距离、前向速度和时间必须是有限正数')
        if not (self.emergency_distance < self.stop_distance
                < self.clear_distance):
            raise ValueError('需要 emergency_distance < stop_distance '
                             '< clear_distance')
        if self.forward_speed > 0.30:
            raise ValueError('本实验 forward_speed 不得超过 0.30 m/s')
        if not math.isfinite(self.turn_speed) or not (
                0 < abs(self.turn_speed) <= 0.80):
            raise ValueError('需要 0 < abs(turn_speed) <= 0.80 rad/s')
        if type(self.clear_samples) is not int or self.clear_samples < 1:
            raise ValueError('clear_samples 必须是正整数')


class AvoidanceController:
    """接收测距事件，使用调用者提供的单调时间输出 (vx, wz)。"""

    def __init__(self, config=None):
        """初始等待测距，前进前需要连续空旷读数。"""
        self.config = config or AvoidanceConfig()
        self.state = 'WAIT_SENSOR'
        self.distance = None
        self.last_received = None
        self.clear_count = 0
        self.state_since = 0.0
        self.turn_since = None

    def update_range(self, distance, minimum, maximum, now):
        """每条新消息调用一次；无穷值、越界值和非法量程均无效。"""
        c = self.config
        valid = (
            all(math.isfinite(v) for v in (distance, minimum, maximum))
            and 0 <= minimum < maximum
            and minimum <= distance <= maximum
        )
        if (self.last_received is None
                or now - self.last_received > c.sensor_timeout):
            self.clear_count = 0
        self.last_received = now
        self.distance = distance if valid else None
        if valid and distance >= c.clear_distance:
            self.clear_count += 1
        else:
            self.clear_count = 0
        return valid

    def _enter(self, state, now):
        if self.state != state:
            self.state = state
            self.state_since = now

    def command(self, now):
        """计算本周期命令；转向超时后锁定 FAULT，重启节点才能解除。"""
        c = self.config
        stopped = (0.0, 0.0)
        if self.state == 'FAULT':
            return stopped
        if (self.turn_since is not None
                and now - self.turn_since >= c.max_turn_duration):
            self._enter('FAULT', now)
            return stopped
        if (self.last_received is None or self.distance is None
                or now - self.last_received > c.sensor_timeout):
            self.clear_count = 0
            self._enter('WAIT_SENSOR', now)
            return stopped
        if self.distance <= c.emergency_distance:
            self._enter('TOO_CLOSE', now)
            return stopped
        if self.clear_count >= c.clear_samples:
            self.turn_since = None
            self._enter('FORWARD', now)
            return (c.forward_speed, 0.0)
        if self.state == 'FORWARD' and self.distance > c.stop_distance:
            return (c.forward_speed, 0.0)
        # 空旷读数不足时不转向，继续等待新采样确认。
        if self.distance >= c.clear_distance:
            self._enter('WAIT_CLEAR', now)
            return stopped
        if self.state not in ('BRAKE', 'TURN'):
            self._enter('BRAKE', now)
            return stopped
        if self.state == 'BRAKE':
            if now - self.state_since < c.brake_duration:
                return stopped
            if self.turn_since is None:
                self.turn_since = now
            self._enter('TURN', now)
        return (0.0, c.turn_speed)


def analyse_scan(ranges, angle_min, angle_increment, range_min, range_max,
                 warning_distance=0.40):
    """计算传感器坐标系四个 90° 扇区的最近距离，不推断未知区域。"""
    if not math.isfinite(warning_distance) or warning_distance <= 0:
        raise ValueError('warning_distance 必须是有限正数')
    names = ('front', 'left', 'rear', 'right')
    sectors = {
        name: {'distance_m': None, 'valid_points': 0, 'near': None}
        for name in names
    }
    result = {'status': 'INVALID', 'sectors': sectors}
    if (not all(math.isfinite(v) for v in (
            angle_min, angle_increment, range_min, range_max))
            or angle_increment == 0 or not 0 <= range_min < range_max):
        return result
    for i, distance in enumerate(ranges):
        if (not math.isfinite(distance)
                or not range_min <= distance <= range_max):
            continue
        angle = angle_min + i * angle_increment
        if not math.isfinite(angle):
            continue
        # [-45°, 45°) 为前方；其余按左、后、右依次划分。
        index = int(((angle + math.pi / 4) % (2 * math.pi))
                    / (math.pi / 2)) % 4
        sector = sectors[names[index]]
        sector['valid_points'] += 1
        previous = sector['distance_m']
        if previous is None or distance < previous:
            sector['distance_m'] = float(distance)
    for sector in sectors.values():
        if sector['distance_m'] is not None:
            sector['near'] = sector['distance_m'] <= warning_distance
    count = sum(s['valid_points'] for s in sectors.values())
    result['status'] = 'OK' if count else 'NO_VALID_POINTS'
    return result
