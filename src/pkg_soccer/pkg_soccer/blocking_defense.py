"""Fixed close-blocking role; world positions in, body velocity out."""
import math

import numpy as np

from .control import wrap_angle


class BlockingDefense:
    def __init__(self):
        self.target = None

    def command(self, controller):
        c = controller
        d, b = c.dog, c.ball
        names = ('opponent', 'opponent2')
        distances = {name: math.hypot(getattr(c, name).x-b.x,
                                     getattr(c, name).y-b.y) for name in names}
        nearest = min(names, key=lambda name: distances[name])
        # Retain the mark through small tracking/possession fluctuations.
        if self.target is None or distances[nearest] + .35 < distances[self.target]:
            self.target = nearest
        opponent = getattr(c, self.target)
        center = np.array([opponent.x, opponent.y])
        own_goal = np.array([2*c.cfg.field_center_x-c.cfg.goal_x,
                             2*c.cfg.field_center_y-c.cfg.goal_y])
        delta = own_goal-center
        length = float(np.linalg.norm(delta))
        unit = delta/length if length > 1e-6 else np.array(
            [-math.cos(c.cfg.field_yaw), -math.sin(c.cfg.field_yaw)])
        target = center + .40*unit + .25*np.asarray(c.tracks[self.target].velocity[:2])
        position = np.array([d.x, d.y])
        gap = position-center
        longitudinal = float(gap @ unit)
        lateral = gap-longitudinal*unit
        # If on the wrong side, go around the opponent instead of driving
        # straight through it to reach the goal-side station.
        if longitudinal < 0. and np.linalg.norm(gap) < 1.0:
            side = np.array([-unit[1], unit[0]])
            sign = 1. if float(gap @ side) >= 0. else -1.
            target = center + .40*unit + sign*.65*side
        fx, fy = c.field_coordinates(*target)
        target = np.asarray(c.world_coordinates(
            np.clip(fx, c.cfg.field_min_x+.25, c.cfg.field_max_x-.25),
            np.clip(fy, c.cfg.field_min_y+.25, c.cfg.field_max_y-.25)))
        world = 1.5*(target-position)
        if 0. < longitudinal < .85 and np.linalg.norm(lateral) < .6:
            along = float(world @ unit)
            world += unit*(min(along, -.12)-along)
        # Contact is intentional, but approach speed is reduced close to a dog.
        cap = min(c.speed_limit, .18) if np.linalg.norm(gap) < .65 else c.speed_limit
        speed = float(np.linalg.norm(world))
        if speed > cap:
            world *= cap/speed
        # Do not deliberately squeeze our own attacker at close range.
        mate_delta = np.array([c.teammate.x, c.teammate.y])-position
        mate_gap = float(np.linalg.norm(mate_delta))
        if 1e-6 < mate_gap < .65:
            toward = mate_delta/mate_gap
            world -= toward*max(0., float(world @ toward))
        co, si = math.cos(d.yaw), math.sin(d.yaw)
        heading = math.atan2(opponent.y-d.y, opponent.x-d.x)
        yaw_rate = np.clip(1.5*wrap_angle(heading-d.yaw), -.7, .7)
        velocity = np.array([co*world[0]+si*world[1],
                             -si*world[0]+co*world[1], yaw_rate])
        return velocity, '持续贴身阻挡 {}：占住对手与己方球门之间'.format(self.target)
