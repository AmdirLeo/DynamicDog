"""Keep useful ball-based control when peer tracking or velocity is unavailable.

Never fabricate peers or feed missing observations into the trained actor.
Requires a current, finite dog/ball pair in the same world frame.
"""
import math

import numpy as np

from .control import wrap_angle


def pose_available(c, name, now):
    pose = getattr(c, name, None)
    return (pose is not None and c.dog is not None and bool(pose.frame) and
            pose.frame == c.dog.frame and
            all(math.isfinite(v) for v in (pose.x, pose.y, pose.yaw)) and
            0 <= now-pose.received <= c.cfg.pose_timeout)


def issues(c, now):
    return [name for name, track in c.tracks.items()
            if not pose_available(c, name, now) or not track.valid(now)]


def core_reason(c, now):
    reasons = []
    for name in ('dog', 'ball'):
        raw = c._observed.get(name)
        if raw is None:
            reasons.append(name + '=未收到或无效')
        elif c.dog is not None and raw.frame != c.dog.frame:
            reasons.append(name + '=坐标系不一致')
        elif not pose_available(c, name, now):
            reasons.append('{}=定位不可用，末帧距今{:.2f}s'.format(name, now-raw.received))
    return '；'.join(reasons) or '本狗/球缺少有效坐标系'


def command(c, now):
    d, b = c.dog, c.ball
    p = np.array([d.x, d.y])
    ball = np.array([b.x, b.y])
    goal = np.array([c.cfg.goal_x, c.cfg.goal_y])
    direction = goal-ball
    norm = float(np.linalg.norm(direction))
    unit = direction/norm if norm > 1e-6 else np.array(
        [math.cos(c.cfg.field_yaw), math.sin(c.cfg.field_yaw)])
    side = np.array([-unit[1], unit[0]])
    if c.defending:
        mark = c.team_tactics.mark if c.team_tactics else c.blocking_defense.target
        if mark is None:
            mark = next((name for name in ('opponent', 'opponent2')
                         if pose_available(c, name, now)), None)
        if mark and pose_available(c, mark, now):
            velocity, reason = c.blocking_defense.command(c, mark)
            return velocity, '定位降级，继续' + reason
        # Missing marked opponent: protect the goal-side ball lane, with
        # complementary offsets so both dogs do not converge on one station.
        slot = -1. if mark == 'opponent' else 1.
        target = ball-.7*unit+slot*.5*side
        heading_point = ball
        reason = '盯防对手不可用，按球位置补防'
    elif c.team_role == 'assist':
        target = ball-1.0*unit+.7*side
        heading_point = ball
        reason = '队友/对手或测速未就绪，按球位置策应'
    else:
        relative = p-ball
        along, lateral = float(relative @ unit), float(relative @ side)
        if along < -.12 and abs(lateral) < .22:
            target = ball+.5*unit
            heading_point = goal if np.linalg.norm(relative) < .7 else target
        else:
            target = ball-.55*unit
            if along > -.12 and np.linalg.norm(relative) < 1.:
                target += (.65 if lateral >= 0. else -.65)*side
            heading_point = target
        reason = '队友/对手或测速未就绪，按球位置低速接近和推球'
    world = 1.2*(target-p)
    # Use only available peers for close-range avoidance; never revive stale
    # poses or reuse an unrelated coordinate frame as an obstacle.
    for name in ('teammate', 'opponent', 'opponent2'):
        if not pose_available(c, name, now):
            continue
        peer = getattr(c, name)
        delta = np.array([peer.x, peer.y])-p
        distance = float(np.linalg.norm(delta))
        if 1e-6 < distance < .65:
            toward = delta/distance
            closing = max(0., float(world @ toward))
            world -= closing*toward
            if closing > 0. and np.linalg.norm(world) < .05:
                world += .12*np.array([-toward[1], toward[0]])
    co, si = math.cos(d.yaw), math.sin(d.yaw)
    heading = math.atan2(heading_point[1]-p[1], heading_point[0]-p[0])
    return np.array([co*world[0]+si*world[1], -si*world[0]+co*world[1],
                     np.clip(1.5*wrap_angle(heading-d.yaw), -.7, .7)]), reason
