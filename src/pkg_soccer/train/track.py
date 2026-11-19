"""训练用最小速度估计器（与 pkg_soccer.measurement.Track 行为一致）。

优先复用部署端实现；若训练机没装 pkg_soccer，则使用本文件的同名副本。
"""

from collections import deque
import math

from ._wrap import wrap_angle


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
            elif math.hypot(x - previous[2], y - previous[3]) > self.max_speed * dt + .03:
                self.invalidate('position_jump')
                return
        angle = (self.samples[-1][4] + wrap_angle(yaw - self.samples[-1][4])
                 if self.samples else yaw)
        self.samples.append((now, stamp, x, y, angle))
        while len(self.samples) > 2 and now - self.samples[1][0] >= .10:
            self.samples.popleft()
        if len(self.samples) < 3 or now - self.samples[0][0] < .08:
            self.velocity = None
            return
        points = list(self.samples)
        span = points[-1][0] - points[0][0]
        source_span = points[-1][1] - points[0][1]
        source_ok = (points[0][1] > 0 and .5 * span <= source_span <= 2 * span and
                     all(b[1] > a[1] for a, b in zip(points, points[1:])))
        axis = 1 if source_ok else 0
        self.basis = 'source' if source_ok else 'receive'
        ts = [p[axis] - points[0][axis] for p in points]
        mean = sum(ts) / len(ts)
        denom = sum((t - mean) ** 2 for t in ts)
        self.velocity = tuple(sum((t - mean) * (p[j] - points[0][j])
                                  for t, p in zip(ts, points)) / denom
                              for j in (2, 3, 4))
        if math.hypot(*self.velocity[:2]) > self.max_speed:
            self.invalidate('velocity_outlier')
        else:
            self.quality = 'ok'

    def valid(self, now):
        return (self.latest is not None and self.velocity is not None and
                self.quality == 'ok' and 0 <= now - self.latest[0] <= self.gap)

    @property
    def speed(self):
        return math.hypot(*self.velocity[:2]) if self.velocity else None
