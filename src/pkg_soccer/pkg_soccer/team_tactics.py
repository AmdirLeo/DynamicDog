"""Team attack/defense phase based on the ball's calibrated field half.

Adapt racing7.main_loop's half-field split to either attack direction. Unlike
its guard/positive split, both dogs block opponents in the defensive phase.
"""


class TeamTactics:
    def __init__(self, initial_role):
        if initial_role not in ('attack', 'assist'):
            raise ValueError('自动攻防需要互补的初始 attack/assist 角色')
        self.phase = None
        # Stable complementary marks: do not independently choose the same
        # ball carrier or exchange opponents when distances briefly cross.
        self.mark = 'opponent' if initial_role == 'attack' else 'opponent2'

    def update(self, controller):
        bx, _ = controller.field_coordinates(controller.ball.x, controller.ball.y)
        # Midfield (including 1 m into our half) always attacks, even after
        # defense. A stateless threshold also avoids retaining defense at kickoff.
        self.phase = 'attack' if bx >= -1.0 else 'defense'
        return self.phase
