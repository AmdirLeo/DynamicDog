"""Fixed defensive override for the danger zone near our own goal.

When the ball is in our half, close to our goal, and an opponent is carrying
it, a rule takes over from the network policy and drives our dog straight at
the carrier until contact. A hysteresis band on the danger distance prevents
the rule from flapping on/off at the boundary. Outside the danger zone the
planner returns None and the network policy keeps full control.

This is a safety layer, not a learned behaviour: it never changes observations
or tracks, and the caller still applies the shared speed/field limits after the
override.
"""

import math

from .control import wrap_angle


class DefensePlanner:
    """Rule-based "charge the carrier" override near our own goal.

    Danger is the conjunction of three conditions:
      1. an opponent is near the ball (carrying it),
      2. the ball is in our half (field x < 0),
      3. the ball is within ``fraction`` of the field length from our goal line.

    The distance threshold uses hysteresis: while idle the ball must come
    closer than ``fraction - hysteresis`` to engage; while engaged it must
    move farther than ``fraction + hysteresis`` to disengage. The gap keeps
    the state from oscillating when the ball hovers at the boundary.
    """

    def __init__(self, fraction=1.0/3.0, hysteresis=0.40,
                 carrier_dist=0.60, contact_dist=0.50):
        self.fraction = float(fraction)
        self.hysteresis = float(hysteresis)
        self.carrier_dist = float(carrier_dist)
        self.contact_dist = float(contact_dist)
        if not (0.05 <= self.fraction <= 0.60):
            raise ValueError('defense_fraction 必须在 [0.05, 0.60]')
        if not (0.0 <= self.hysteresis <= 0.50):
            raise ValueError('defense_hysteresis 必须在 [0, 0.50] m')
        if not (0.20 <= self.carrier_dist <= 1.50):
            raise ValueError('defense_carrier_dist 必须在 [0.20, 1.50] m')
        if not (0.20 <= self.contact_dist <= 1.20):
            raise ValueError('defense_contact_dist 必须在 [0.20, 1.20] m')
        self.reset()

    def reset(self):
        self.active = False
        self.state, self.reason = 'RL', ''

    def _carrier(self, c, ball):
        """Return (opponent_pose, gap) of the nearest opponent to the ball."""
        best, best_gap = None, math.inf
        for name in ('opponent', 'opponent2'):
            o = getattr(c, name, None)
            if o is None:
                continue
            gap = math.hypot(o.x-ball.x, o.y-ball.y)
            if gap < best_gap:
                best, best_gap = o, gap
        if best is None or best_gap > self.carrier_dist:
            return None, best_gap
        return best, best_gap

    def _danger(self, c, ball):
        """Return (danger, distance_from_own_goal, field_length, threshold)."""
        bx, _ = c.field_coordinates(ball.x, ball.y)
        # 对方门在场地坐标 +half，本方门线在 -half。
        gx, _ = c.field_coordinates(c.cfg.goal_x, c.cfg.goal_y)
        field_length = c.cfg.field_max_x - c.cfg.field_min_x
        if field_length <= 1e-6:
            return False, math.inf, field_length, 0.0
        # 球到本方门线的距离：越大越远离本方球门。
        distance = bx + gx
        margin = self.fraction*field_length
        threshold = (margin + self.hysteresis if self.active
                     else margin - self.hysteresis)
        # 我方半场 (bx<0) 且足够靠近本方球门。
        danger = bx < 0.0 and distance <= threshold
        return danger, distance, field_length, threshold

    def command(self, c, now):
        """Return (vx, vy, wz) body-frame override, or None to keep RL."""
        d, b = getattr(c, 'dog', None), getattr(c, 'ball', None)
        if d is None or b is None:
            self.active = False
            return None
        carrier, gap = self._carrier(c, b)
        danger, distance, field_length, threshold = self._danger(c, b)
        if carrier is None or not danger:
            was_active = self.active
            self.active = False
            if was_active:
                self.state, self.reason = 'RL', ''
            return None

        self.active = True
        self.state = 'DEFENSE'
        dx, dy = carrier.x-d.x, carrier.y-d.y
        dist = math.hypot(dx, dy)
        # 世界系 → 机身系单位向量（机身 +x 朝狗头、+y 朝左）。
        co, si = math.cos(d.yaw), math.sin(d.yaw)
        ux, uy = dx/max(dist, 1e-9), dy/max(dist, 1e-9)
        fx, fy = co*ux + si*uy, -si*ux + co*uy
        if dist <= self.contact_dist:
            # 已经贴上对手：停止平移避免持续挤压/穿模，保留转向对准。
            vx = vy = 0.0
        else:
            # 越接近越减速，撞上之前收住；统一限速由调用方再做。
            speed = min(float(c.speed_limit), dist-self.contact_dist)
            if speed < 0.0:
                speed = 0.0
            vx, vy = fx*speed, fy*speed
        yaw_to = math.atan2(dy, dx)
        turn = max(-c.cfg.turn_speed,
                   min(c.cfg.turn_speed, 1.4*wrap_angle(yaw_to-d.yaw)))
        self.reason = (
            '危险区固定防守：球距本方门 {:.2f}m (阈值 {:.2f}m，场长 {:.2f}m)'
            '；持球对手距球 {:.2f}m，本狗距对手 {:.2f}m；{}'.format(
                distance, threshold, field_length, gap, dist,
                '已贴身，停止平移' if dist <= self.contact_dist
                else '正向持球对手移动直到碰撞'))
        return (vx, vy, turn)
