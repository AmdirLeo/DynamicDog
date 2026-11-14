"""被动标定计算；不依赖 ROS，也不产生运动指令。"""

import math
from collections import deque

from .control import wrap_angle


class Track:
    """约 0.1 秒滑窗回归速度，拒绝断流、跳点和跨坐标系差分。"""

    def __init__(self, max_speed, gap=0.20):
        self.max_speed, self.gap = max_speed, gap
        self.samples = deque()
        self.latest = None
        self.velocity = None
        self.basis = ''
        self.quality = 'missing'

    def invalidate(self, reason):
        self.samples.clear()
        self.velocity = None
        self.quality = reason

    def update(self, now, stamp, x, y, yaw, frame):
        if not all(math.isfinite(v) for v in (now, stamp, x, y, yaw)) or stamp < 0:
            self.invalidate('invalid_pose')
            return
        previous = self.latest
        self.latest = (now, stamp, x, y, yaw, frame)
        if previous is not None:
            dt = now - previous[0]
            if frame != previous[5] or dt <= 0 or dt > self.gap:
                self.invalidate('gap_or_frame_change')
            elif stamp > 0 and previous[1] > 0 and stamp < previous[1]:
                self.invalidate('source_clock_reset')
            elif math.hypot(x-previous[2], y-previous[3]) > self.max_speed*dt + .03:
                self.invalidate('position_jump')
                return
        angle = (self.samples[-1][4] + wrap_angle(yaw-self.samples[-1][4])
                 if self.samples else yaw)
        self.samples.append((now, stamp, x, y, angle))
        while len(self.samples) > 2 and now-self.samples[1][0] >= .10:
            self.samples.popleft()
        if len(self.samples) < 3 or now-self.samples[0][0] < .08:
            self.velocity = None
            return
        # 一些 VRPN 发布端复用 stamp；此时用本机单调接收时间并明确标记。
        points = list(self.samples)
        span = points[-1][0]-points[0][0]
        source_span = points[-1][1]-points[0][1]
        source_ok = (points[0][1] > 0 and .5*span <= source_span <= 2*span and
                     all(b[1] > a[1] for a, b in zip(points, points[1:])))
        axis = 1 if source_ok else 0
        self.basis = 'source' if source_ok else 'receive'
        ts = [p[axis]-points[0][axis] for p in points]
        mean = sum(ts)/len(ts)
        denom = sum((t-mean)**2 for t in ts)
        self.velocity = tuple(sum((t-mean)*(p[j]-points[0][j])
                                  for t, p in zip(ts, points))/denom
                              for j in (2, 3, 4))
        if math.hypot(*self.velocity[:2]) > self.max_speed:
            self.invalidate('velocity_outlier')
        else:
            self.quality = 'ok'

    def valid(self, now):
        return (self.latest is not None and self.velocity is not None and
                self.quality == 'ok' and 0 <= now-self.latest[0] <= self.gap)

    @property
    def speed(self):
        return math.hypot(*self.velocity[:2]) if self.velocity else None


class Measurements:
    """停车按明确零指令/结束帧触发；触球只报告近狗的球起动候选。"""

    def __init__(self):
        self.dog, self.ball = Track(3.), Track(12.)
        self.command = None
        self.stop = None
        self.launch = None
        self.still_since = None
        self.ball_was_still = False
        self.events = []

    def emit(self, event, **values):
        self.events.append(dict(event=event, **values))

    def command_update(self, now, velocity, end=False):
        previous = self.command
        velocity = (0., 0., 0.) if end else tuple(velocity)
        moving = any(abs(v) > .02 for v in velocity)
        was_moving = previous and any(abs(v) > .02 for v in previous[1])
        self.command = (now, velocity)
        if moving and self.stop is not None:
            self.emit('stop_invalid', note='command_resumed')
            self.stop = None
        if not moving and was_moving:
            if (now-previous[0] > .25 or not self.dog.valid(now) or
                    now-self.dog.latest[0] > .10 or self.dog.speed < .08):
                self.emit('stop_invalid', note='no_fresh_moving_dog_or_command')
                return
            x, y = self.dog.latest[2:4]
            self.stop = dict(start=now, x=x, y=y, last=(x, y), path=0.,
                             excursion=0., still=None)
            self.emit('stop_start', stop_start_speed_mps=self.dog.speed)

    def tick(self, now):
        if self.stop is not None:
            if not self.dog.valid(now) or now-self.stop['start'] > 5.:
                self.emit('stop_invalid', note='tracking_invalid_or_stop_timeout')
                self.stop = None
        if self.launch is not None and not self.ball.valid(now):
            self.emit('ball_launch_invalid', note='tracking_invalid')
            self.launch = None
        if not self.ball.valid(now):
            self.still_since = None
            self.ball_was_still = False

    def pose_update(self, kind, now, stamp, x, y, yaw, frame):
        track = getattr(self, kind)
        track.update(now, stamp, x, y, yaw, frame)
        self.tick(now)
        if not track.valid(now):
            return
        if kind == 'dog' and self.stop is not None:
            s = self.stop
            s['path'] += math.hypot(x-s['last'][0], y-s['last'][1])
            s['last'] = (x, y)
            distance = math.hypot(x-s['x'], y-s['y'])
            s['excursion'] = max(s['excursion'], distance)
            if track.speed < .05:
                if s['still'] is None:
                    s['still'] = now
                if now-s['still'] >= .5:
                    self.emit('stop_result', stop_distance_m=distance,
                              stop_max_excursion_m=s['excursion'],
                              stop_path_m=s['path'],
                              stop_time_s=s['still']-s['start'])
                    self.stop = None
            else:
                s['still'] = None
        if kind != 'ball':
            return
        if self.launch is not None:
            self.launch['peak'] = max(self.launch['peak'], track.speed)
            if now-self.launch['start'] >= .5:
                self.emit('ball_launch_result', ball_peak_speed_mps=self.launch['peak'],
                          note='candidate_not_verified_contact;window=0.5s')
                self.launch = None
        near = (self.dog.valid(now) and self.dog.latest[5] == frame and
                math.hypot(x-self.dog.latest[2], y-self.dog.latest[3]) < .65)
        if track.speed < .08:
            if self.still_since is None:
                self.still_since = now
            self.ball_was_still = now-self.still_since >= .3
        elif track.speed > .20:
            if self.ball_was_still and near and self.launch is None:
                self.launch = dict(start=now, peak=track.speed)
                self.emit('ball_launch_start', note='candidate_not_verified_contact')
            self.still_since = None
            self.ball_was_still = False

    def finish(self):
        if self.stop is not None:
            self.emit('stop_invalid', note='recording_ended_before_settling')
        if self.launch is not None:
            self.emit('ball_launch_invalid', note='recording_ended_before_window_end')
        self.stop = self.launch = None
