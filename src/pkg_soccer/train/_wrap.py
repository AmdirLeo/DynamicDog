"""角度工具与共用常量。"""

import math

ACTION_SCALE = (0.45, 0.20, 0.70)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))
