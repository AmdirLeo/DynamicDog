"""用快走步态（motion_id=305）的摆动腿踢球。

思路（不切位控、不抬腿单脚站）：
- 狗一直用 305 快走朝球走；305 是快速对角小跑，摆动腿在腾空相会向前摆
  （足端相对机身的前向速度约 0.5~1.0 m/s，再叠加机身速度）；
- 当目标腿刚离地（`motion_status.contact` 的对应位由 1→0）时，给一个
  短促的前进速度脉冲 + 抬腿高度脉冲（`step_height` 前 0.06 m），
  让摆动腿在球的位置扫过去，把球"踢"出去，而不是用身体顶球；
- 脉冲结束立即回落，避免整段都在推球。

优点：全程走 305，不需要位控姿态切换（避开真机 3028 那类问题）。
缺点：机身会随球前进；出球方向取决于脉冲方向，精度不如单腿位控踢。

本文件是纯逻辑（相位检测 + 状态机），`gait_kick_action.py` 单文件测试脚本
与 `soccer.py` 的 `kick_mode:=gait` 共用这套逻辑。
"""

import math

# 低 4 位触地状态：1=触地、0=抬起；位序 FR、FL、RR、RL
LEG_BITS = {'fr': 0, 'fl': 1, 'rr': 2, 'rl': 3}
ALL_CONTACT = 0x0F


def _normalize(x, y):
    norm = math.hypot(x, y)
    if norm < 1e-6:
        return (1.0, 0.0)
    return (x / norm, y / norm)


class GaitKickConfig:
    """快走踢球的参数。"""

    def __init__(self, burst_speed=0.75, burst_ms=200,
                 step_front=0.06, step_back=0.05, step_normal=0.05,
                 phase_timeout_ms=350, sync_leg='fr',
                 warmup_ms=250, warmup_speed=0.15,
                 gap_ms=0):
        self.burst_speed = self._number('burst_speed', burst_speed, 0.05, 1.5)
        self.warmup_speed = self._number('warmup_speed', warmup_speed, 0.0, 0.6)
        self.step_front = self._number('step_front', step_front, 0.0, 0.08)
        self.step_back = self._number('step_back', step_back, 0.0, 0.08)
        self.step_normal = self._number('step_normal', step_normal, 0.0, 0.08)
        self.burst_ms = self._ms('burst_ms', burst_ms, 40, 600)
        self.phase_timeout_ms = self._ms('phase_timeout_ms', phase_timeout_ms,
                                         50, 1200)
        self.warmup_ms = self._ms('warmup_ms', warmup_ms, 0, 1000)
        self.gap_ms = self._ms('gap_ms', gap_ms, 0, 2000)
        leg = str(sync_leg).lower()
        self.sync_leg = leg if leg in LEG_BITS else None

    @staticmethod
    def _number(name, value, low, high):
        value = float(value)
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{name} 必须在 [{low}, {high}]，实得 {value}')
        return value

    @staticmethod
    def _ms(name, value, low, high):
        value = int(value)
        if not low <= value <= high:
            raise ValueError(f'{name} 必须在 [{low}, {high}] ms，实得 {value}')
        return value

    @classmethod
    def from_params(cls, get):
        """从 ROS 参数读取（get(name, default) 返回参数值）。"""
        return cls(
            burst_speed=float(get('kick_gait_speed', 0.75)),
            burst_ms=int(get('kick_gait_burst_ms', 200)),
            step_front=float(get('kick_gait_step_front', 0.06)),
            step_back=float(get('kick_gait_step_back', 0.05)),
            step_normal=float(get('kick_step_height', 0.05)),
            phase_timeout_ms=int(get('kick_gait_phase_timeout_ms', 350)),
            sync_leg=get('kick_gait_sync_leg', 'fr'),
            warmup_ms=int(get('kick_gait_warmup_ms', 250)),
            warmup_speed=float(get('kick_gait_warmup_speed', 0.15)),
            gap_ms=int(get('kick_gait_gap_ms', 0)),
        )


class SwingDetector:
    """从 motion_status.contact 检测目标腿的离地时刻（1→0）。"""

    def __init__(self, leg):
        self.leg = leg
        self.bit = LEG_BITS.get(leg)
        self.last = None
        self.edge_at = None
        self.period = None
        self._last_edge = None

    def update(self, contact, now):
        if contact is None or self.bit is None:
            return False
        bit = (int(contact) >> self.bit) & 1
        edge = self.last == 1 and bit == 0
        self.last = bit
        if edge:
            if self._last_edge is not None:
                self.period = now - self._last_edge
            self._last_edge = now
            self.edge_at = now
        return edge


class GaitKickRunner:
    """一次"等摆动腿离地 → 速度/抬腿脉冲"的状态机。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.detector = SwingDetector(cfg.sync_leg) if cfg.sync_leg else None
        self.state = 'idle'
        self.started = None
        self.wait_started = None
        self.until = None
        self.direction = (1.0, 0.0)
        self.base_speed = 0.0
        self.base_wz = 0.0
        self.reason = ''
        self.detected_phase = None

    # ------------------------------------------------------------------ 控制
    def start(self, now, velocity):
        vx, vy, wz = (float(v) for v in velocity)
        self.direction = _normalize(vx, vy)
        self.base_speed = math.hypot(vx, vy)
        self.base_wz = wz
        self.started = now
        self.until = None
        self.detected_phase = None
        if self.cfg.warmup_ms > 0:
            self.state = 'warmup'
            self.until = now + self.cfg.warmup_ms / 1000.0
            self.reason = '起走 305 步态'
        else:
            self.state = 'waiting'
            self.wait_started = now
            self.reason = '等待摆动腿离地'

    def update(self, now, contact):
        """返回 None 表示结束；否则返回 {phase, velocity, step_height}。"""
        if self.state == 'idle':
            return None
        if self.state == 'warmup':
            if now >= self.until:
                self.state = 'waiting'
                self.wait_started = now
                self.reason = '等待摆动腿离地'
            return self._plan('warmup', self.cfg.warmup_speed, normal=True)
        if self.state == 'waiting':
            if self.detector is None:
                self.state = 'burst'
                self.until = now + self.cfg.burst_ms / 1000.0
                self.reason = '未启用相位检测，直接踢球脉冲'
                return self._plan('burst', self.cfg.burst_speed, normal=False)
            edge = self.detector.update(contact, now) if self.detector else False
            if edge:
                self.detected_phase = now - self.started
            if edge or now - self.wait_started >= self.cfg.phase_timeout_ms / 1000.0:
                self.state = 'burst'
                self.until = now + self.cfg.burst_ms / 1000.0
                self.reason = ('检测到摆动腿离地，开始踢球脉冲'
                               if edge else '未检测到相位，按超时踢球')
            return self._plan('waiting', self.base_speed, normal=True)
        if self.state == 'burst':
            if now >= self.until:
                self.state = 'gap' if self.cfg.gap_ms > 0 else 'done'
                self.until = now + self.cfg.gap_ms / 1000.0
                self.reason = '踢球脉冲结束'
            return self._plan('burst', self.cfg.burst_speed, normal=False)
        if self.state == 'gap':
            if now >= self.until:
                self.state = 'done'
            return self._plan('gap', self.cfg.warmup_speed, normal=True)
        return None

    def _plan(self, phase, speed, normal):
        # waiting 阶段沿用调用方给的速度方向与大小（由控制器决定），
        # 只有 burst 阶段强制用配置的前进速度
        if phase in ('warmup', 'gap'):
            velocity = (self.direction[0] * speed, self.direction[1] * speed, 0.0)
        elif phase == 'waiting':
            velocity = (self.direction[0] * speed, self.direction[1] * speed,
                        self.base_wz)
        else:
            velocity = (self.direction[0] * speed, self.direction[1] * speed,
                        self.base_wz)
        if normal:
            step = (self.cfg.step_normal, self.cfg.step_normal)
        else:
            step = (self.cfg.step_front, self.cfg.step_back)
        return {'phase': phase, 'velocity': velocity, 'step_height': step,
                'reason': self.reason}

    @property
    def done(self):
        return self.state == 'done'

    def stop(self):
        self.state = 'idle'


def _self_test():
    """python -m pkg_soccer.gait_kick"""
    cfg = GaitKickConfig(warmup_ms=0)
    runner = GaitKickRunner(cfg)
    runner.start(0.0, (0.4, 0.0, 0.0))
    # 接触状态：全接触 → 右前离地（0b1110）→ 全接触
    contacts = [0x0F] * 5 + [0x0E] + [0x0E] * 6 + [0x0F] * 3
    phases = []
    burst_plan = None
    for tick in range(80):
        now = tick * 0.02
        contact = contacts[min(tick, len(contacts) - 1)]
        plan = runner.update(now, contact)
        if plan is None:
            break
        phases.append(plan['phase'])
        if plan['phase'] == 'burst':
            burst_plan = plan
    print('阶段序列：', phases)
    assert 'burst' in phases and burst_plan is not None
    assert burst_plan['step_height'] == (0.06, 0.05), burst_plan
    assert abs(burst_plan['velocity'][0] - 0.75) < 1e-6
    print('快走踢球状态机自检通过')


if __name__ == '__main__':
    _self_test()
