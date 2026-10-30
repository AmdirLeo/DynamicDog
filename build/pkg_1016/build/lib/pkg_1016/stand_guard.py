"""启动姿态门控：坐/趴时请求一次站立，确认完成后才能行走。

依据官方 MotionStatus、MotionResultCmd 和 motion_id_map；不推断未知姿态。
纯 Python，兼容 Python 3.6；时间由调用方提供，便于验证超时和回调顺序。
"""

import math


class StandGuard:
    """只在启动阶段自动站立；运行后失去姿态条件则锁定，需人工检查。"""

    def __init__(self, walk_id=303, status_timeout=1.0, stand_timeout=25.0):
        """姿态超时与站立请求超时均使用单调时间。"""
        if any(not math.isfinite(v) or v <= 0
               for v in (status_timeout, stand_timeout)):
            raise ValueError('posture_timeout 和 stand_timeout 必须是有限正数')
        self.walk_id = walk_id
        self.status_timeout = status_timeout
        self.stand_timeout = stand_timeout
        self.state = 'WAIT_POSTURE'
        self.reason = '等待新鲜的运动状态'
        self.last_received = None
        self.motion_id = None
        self.switch_status = None
        self.progress = None
        self.healthy = False
        self.stable_count = 0
        self.requested_at = None
        self.response_ok = False
        self.response_code = None
        self.ready_once = False

    def fail(self, reason):
        """失败锁定，不自动重发站立或恢复行走。"""
        self.state = 'POSTURE_FAILED'
        self.reason = reason

    def _standing(self):
        return (self.motion_id == 112
                or self.motion_id == 111 and self.progress >= 96)

    def update(self, motion_id, switch_status, progress, healthy, now):
        """每条状态消息调用一次；定时器不能代替新消息确认。"""
        gap = (self.last_received is not None
               and now - self.last_received > self.status_timeout)
        if self.ready_once and gap:
            self.fail('行走后姿态数据断流，请检查并重启节点')
        if gap:
            self.stable_count = 0
        previous = self.motion_id
        self.last_received = now
        self.motion_id = motion_id
        self.switch_status = switch_status
        self.progress = progress
        self.healthy = healthy
        normal = healthy and switch_status == 0
        completed = self._standing() or motion_id == self.walk_id
        resting = motion_id in (101, 143) and progress >= 96
        if normal and (completed or resting):
            self.stable_count = self.stable_count + 1 if previous == motion_id else 1
        else:
            self.stable_count = 0
        if self.state == 'POSTURE_FAILED':
            return
        if not healthy or switch_status not in (0, 1):
            self.fail('运动状态报告故障或禁止切换，不能自动站立/行走')
        elif self.ready_once and motion_id not in (111, 112, self.walk_id):
            self.fail('运行中姿态改变，停止控制且不重新自动站立')

    def command(self, now):
        """返回 WAIT、REQUEST 或 WALK；REQUEST 必须由调用方显式确认发送。"""
        if self.state == 'POSTURE_FAILED':
            return 'WAIT'
        if (self.requested_at is not None
                and not self.ready_once
                and now - self.requested_at >= self.stand_timeout):
            self.fail('站立请求或站稳确认超时；不重试，请检查机器狗')
            return 'WAIT'
        if (self.last_received is None
                or now - self.last_received > self.status_timeout):
            if self.ready_once:
                self.fail('姿态数据超时，停止控制')
            else:
                self.state = 'WAIT_POSTURE'
                self.reason = '等待新鲜的运动状态'
                self.stable_count = 0
            return 'WAIT'
        if self.ready_once:
            # 正常站立到当前步态的切换允许继续发送，故障与其他姿态已在回调拦截。
            self.state = 'READY'
            return 'WALK'
        if self.requested_at is not None:
            if (self.response_ok and self._standing()
                    and self.stable_count >= 3):
                self.ready_once = True
                self.state = 'READY'
                self.reason = '站立服务成功，且连续状态确认站稳'
                return 'WALK'
            self.state = 'STANDING'
            self.reason = '等待站立服务成功与连续站稳确认'
            return 'WAIT'
        if self.stable_count < 3:
            self.state = 'WAIT_POSTURE'
            self.reason = '等待姿态正常且连续稳定三帧'
            return 'WAIT'
        if self._standing() or self.motion_id == self.walk_id:
            self.ready_once = True
            self.state = 'READY'
            self.reason = '已确认处于站立或当前行走步态'
            return 'WALK'
        if self.motion_id in (101, 143):
            self.state = 'WAIT_STAND_SERVICE'
            self.reason = '确认坐下/趴下，等待站立服务可用'
            return 'REQUEST'
        self.state = 'WAIT_POSTURE'
        self.reason = '姿态不在可自动起步范围内'
        return 'WAIT'

    def requested(self, now):
        """记录唯一一次请求，禁止同一进程重复起身。"""
        if self.requested_at is not None:
            raise RuntimeError('不能重复发送站立请求')
        self.requested_at = now
        self.stable_count = 0
        self.state = 'STANDING'
        self.reason = '已请求恢复站立，等待结果'

    def result(self, success, motion_id, code):
        """服务成功也不直接放行，仍须新鲜且完成的站立状态。"""
        if self.state == 'POSTURE_FAILED':
            return
        self.response_code = code
        if not success or motion_id != 111:
            self.fail('站立服务未成功：motion_id={} code={}'.format(motion_id, code))
        else:
            self.response_ok = True
