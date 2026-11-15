"""ATTACK supervisor: short rule-based repositioning around blocked shots.

One controller owns velocity output. Rules never move or reset observations,
and no success is inferred merely from avoiding the opponent.
"""

import math

from .control import wrap_angle


def clearance(start, end, obstacle):
    dx, dy = end[0]-start[0], end[1]-start[1]
    t = max(0., min(1., ((obstacle[0]-start[0])*dx +
                         (obstacle[1]-start[1])*dy)/max(dx*dx+dy*dy, 1e-9)))
    return math.hypot(start[0]+t*dx-obstacle[0], start[1]+t*dy-obstacle[1])


class AttackPlanner:
    def __init__(self):
        self.reset()

    def reset(self):
        self.side = 0
        self.target = None
        self.started = self.progress_at = self.cooldown = 0.
        self.best_distance = math.inf
        self.clear_since = None
        self.switched = False
        self.state, self.reason = 'ATTACK_RL', 'ATTACK：RL 接近、控球和射门'

    @staticmethod
    def safe_point(c, point, inset):
        x, y = c.field_coordinates(*point)
        cfg = c.cfg
        return (cfg.field_min_x+inset <= x <= cfg.field_max_x-inset and
                cfg.field_min_y+inset <= y <= cfg.field_max_y-inset)

    def candidate(self, c, side, ux, uy):
        d, b, o = c.dog, c.ball, c.opponent
        # Set up a diagonal push: stand behind and to one side of the ball.
        # Check the other side too, where this push would send the ball.
        target = (b.x-.60*ux-side*.65*uy, b.y-.60*uy+side*.65*ux)
        push = (b.x+.45*ux+side*.49*uy, b.y+.45*uy-side*.49*ux)
        start, opponent = (d.x, d.y), (o.x, o.y)
        if not self.safe_point(c, target, .32) or not self.safe_point(c, push, .85):
            return None
        # Close contact may be escaped by increasing separation. Never route
        # through the defender or through the ball to get behind it.
        for point, radius in ((opponent, .60), ((b.x, b.y), .46)):
            before = math.dist(start, point)
            if clearance(start, target, point) < min(radius, before)-1e-6:
                return None
            if math.dist(target, point) < radius:
                return None
        if clearance((b.x, b.y), push, opponent) < .43:
            return None
        return target

    def choose(self, c, ux, uy, exclude=0):
        options = []
        for side in (-1, 1):
            if side == exclude:
                continue
            target = self.candidate(c, side, ux, uy)
            if target is not None:
                options.append((math.hypot(target[0]-c.dog.x, target[1]-c.dog.y),
                                side, target))
        return min(options)[1:] if options else (0, None)

    def command(self, c, now):
        d, b, o = c.dog, c.ball, c.opponent
        gx, gy = c.cfg.goal_x-b.x, c.cfg.goal_y-b.y
        length = math.hypot(gx, gy)
        self.state, self.reason = 'ATTACK_RL', 'ATTACK：RL 接近、控球和射门'
        if length < .01:
            return None
        ux, uy = gx/length, gy/length
        ox, oy = o.x-b.x, o.y-b.y
        along, lateral = ox*ux+oy*uy, -ox*uy+oy*ux
        blocked = (-.15 < along < min(1.5, length) and abs(lateral) < .65 and
                   math.hypot(o.x-d.x, o.y-d.y) < 1.4 and
                   math.hypot(b.x-d.x, b.y-d.y) < 1.3)
        if not blocked:
            if self.clear_since is None:
                self.clear_since = now
            if self.side and now-self.clear_since >= .4:
                self.side, self.target = 0, None
            if not self.side:
                return None
        else:
            self.clear_since = None
        if now < self.cooldown:
            return None
        if not self.side:
            self.side, self.target = self.choose(c, ux, uy)
            self.started = self.progress_at = now
            self.best_distance, self.switched = math.inf, False
        else:
            # Revalidate the entire path when the defender or ball moves.
            self.target = self.candidate(c, self.side, ux, uy)
        if self.target is None:
            self.side = 0
            self.state, self.reason = 'ATTACK_BLOCKED', 'ATTACK：两侧无可行路线，等待空位'
            return (0., 0., 0.)
        dx, dy = self.target[0]-d.x, self.target[1]-d.y
        distance = math.hypot(dx, dy)
        if distance < .14 or now-self.started > 6.:
            # Give RL an uninterrupted opportunity to actually play the ball.
            self.side, self.target = 0, None
            self.cooldown = now+1.5
            return None
        if distance < self.best_distance-.06:
            self.best_distance, self.progress_at = distance, now
        if now-self.progress_at > 1.5 and not self.switched:
            side, target = self.choose(c, ux, uy, exclude=self.side)
            self.switched = True
            if target is not None:
                self.side, self.target = side, target
                self.best_distance, self.progress_at = math.inf, now
                # Pause one tick on a route change; the next tick revalidates.
                self.state, self.reason = 'ATTACK_REPOSITION', 'ATTACK：绕行无进展，换侧'
                return (0., 0., 0.)
        speed = min(.28, distance)
        wx, wy = speed*dx/distance, speed*dy/distance
        co, si = math.cos(d.yaw), math.sin(d.yaw)
        yaw = math.atan2(b.y-d.y, b.x-d.x)
        turn = max(-.7, min(.7, 1.4*wrap_angle(yaw-d.yaw)))
        self.state = 'ATTACK_REPOSITION'
        self.reason = 'ATTACK：规则调整侧面站位，随后交回 RL'
        return (co*wx+si*wy, -si*wx+co*wy, turn)

    @staticmethod
    def limit_contact(c, velocity):
        """Remove own inward motion near an opponent; permit escape/tangency.

        This is a velocity guard, not a guarantee against a moving opponent.
        """
        vx, vy, wz = (float(v) for v in velocity)
        d, o = c.dog, c.opponent
        dx, dy = o.x-d.x, o.y-d.y
        gap = math.hypot(dx, dy)
        if gap < 1e-6:
            return (0., 0., wz)
        co, si = math.cos(d.yaw), math.sin(d.yaw)
        wx, wy = co*vx-si*vy, si*vx+co*vy
        nx, ny = dx/gap, dy/gap
        ov = c.tracks['opponent'].velocity
        closing = max(0., -(ov[0]*nx+ov[1]*ny))
        allowed = max(0., (gap-.60)/.5-closing)
        excess = max(0., wx*nx+wy*ny-allowed)
        wx, wy = wx-excess*nx, wy-excess*ny
        return (co*wx+si*wy, -si*wx+co*wy, wz)
