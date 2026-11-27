"""Team attack/defense phase based on the ball's calibrated field half.

Adapt racing7.main_loop's half-field split to either attack direction. Unlike
its guard/positive split, both dogs block opponents in the defensive phase.
"""

import math


class TeamTactics:
    def __init__(self, initial_role, kickoff_seconds=0.0):
        if initial_role not in ('attack', 'assist'):
            raise ValueError('自动攻防需要互补的初始 attack/assist 角色')
        self.kickoff_seconds = float(kickoff_seconds)
        if not math.isfinite(self.kickoff_seconds) or not 0. <= self.kickoff_seconds <= 60.:
            raise ValueError('开场强制进攻时长必须在 [0, 60] 秒')
        self.kickoff_started_at = None
        self.phase = None
        # Stable complementary marks: do not independently choose the same
        # ball carrier or exchange opponents when distances briefly cross.
        self.mark = 'opponent' if initial_role == 'attack' else 'opponent2'

    def kickoff_active(self, now):
        return (self.kickoff_started_at is not None and
                now-self.kickoff_started_at < self.kickoff_seconds)

    def update(self, controller, now):
        # Start only once valid observations reach the control loop, not while
        # importing Python, standing up or waiting for initial mocap discovery.
        if self.kickoff_started_at is None:
            self.kickoff_started_at = now
        if self.kickoff_active(now):
            self.phase = 'attack'
            return self.phase
        bx, _ = controller.field_coordinates(controller.ball.x, controller.ball.y)
        # Midfield (including 1 m into our half) always attacks, even after
        # defense. A stateless threshold also avoids retaining defense at kickoff.
        self.phase = 'attack' if bx >= -1.0 else 'defense'
        return self.phase
