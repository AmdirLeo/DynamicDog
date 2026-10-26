"""启动站立门控测试；合成状态不代表真实姿态验证。"""

import math

import pytest

from pkg_1016.stand_guard import StandGuard


def feed(guard, start=0.0, motion_id=143, progress=100):
    """连续送入三个正常姿态样本。"""
    for offset in (0.0, 0.1, 0.2):
        guard.update(motion_id, 0, progress, True, start + offset)


@pytest.mark.parametrize('motion_id', [101, 143])
def test_resting_requests_once_then_requires_result_and_standing(motion_id):
    """坐/趴只请求一次；成功服务响应和站稳反馈缺一不可。"""
    guard = StandGuard()
    assert guard.command(0.0) == 'WAIT'
    feed(guard, motion_id=motion_id)
    assert guard.command(0.2) == 'REQUEST'
    guard.requested(0.2)
    guard.result(True, 111, 0)
    assert guard.command(0.3) == 'WAIT'
    guard.update(111, 1, 60, True, 0.4)
    assert guard.command(0.4) == 'WAIT'
    feed(guard, 0.5, 111)
    assert guard.command(0.7) == 'WALK'
    assert guard.command(0.8) == 'WALK'
    with pytest.raises(RuntimeError):
        guard.requested(0.9)


def test_standing_feedback_before_response_does_not_release():
    """站立反馈先于服务结果到达，也必须等待服务成功。"""
    guard = StandGuard()
    feed(guard)
    guard.requested(0.2)
    feed(guard, 0.3, 111)
    assert guard.command(0.5) == 'WAIT'
    guard.result(True, 111, 0)
    assert guard.command(0.6) == 'WALK'


@pytest.mark.parametrize('motion_id', [111, 112, 303])
def test_already_standing_skips_service(motion_id):
    """已站立或处于当前步态，无需重复起身。"""
    guard = StandGuard()
    feed(guard, motion_id=motion_id)
    assert guard.command(0.2) == 'WALK'
    assert guard.requested_at is None


def test_timer_cannot_replace_fresh_posture_samples():
    """同一帧重复检查不能确认姿态稳定。"""
    guard = StandGuard()
    guard.update(143, 0, 100, True, 0.0)
    for now in (0.1, 0.2, 0.3):
        assert guard.command(now) == 'WAIT'


@pytest.mark.parametrize('motion_id,progress', [(111, 50), (143, 50), (0, 100),
                                             (121, 100), (-1, 100)])
def test_unknown_or_incomplete_posture_cannot_start(motion_id, progress):
    """不从急停、未知或未完成的动作推断能够行走。"""
    guard = StandGuard()
    feed(guard, motion_id=motion_id, progress=progress)
    assert guard.command(0.2) == 'WAIT'


@pytest.mark.parametrize('switch_status,healthy', [(2, True), (3, True),
                                                (4, True), (14, True),
                                                (0, False)])
def test_fault_is_latched(switch_status, healthy):
    """急停、阻尼、抬起、充电、错误等不能被新正常帧自动解除。"""
    guard = StandGuard()
    guard.update(143, switch_status, 100, healthy, 0.0)
    feed(guard, 0.1)
    assert guard.command(0.3) == 'WAIT'
    assert guard.state == 'POSTURE_FAILED'


@pytest.mark.parametrize('success,motion_id', [(False, 111), (True, 143)])
def test_failed_service_does_not_retry(success, motion_id):
    """失败响应锁定，不能自动再次起身。"""
    guard = StandGuard()
    feed(guard)
    guard.requested(0.2)
    guard.result(success, motion_id, 123)
    feed(guard, 0.3, 111)
    assert guard.command(0.5) == 'WAIT'
    assert guard.state == 'POSTURE_FAILED'
    assert guard.response_code == 123


def test_service_or_confirmation_timeout_ignores_late_success():
    """超时后晚到的成功结果不能重新放行。"""
    guard = StandGuard(stand_timeout=2.0)
    feed(guard)
    guard.requested(0.2)
    assert guard.command(2.3) == 'WAIT'
    assert guard.state == 'POSTURE_FAILED'
    guard.result(True, 111, 0)
    feed(guard, 2.4, 111)
    assert guard.command(2.6) == 'WAIT'


def test_stale_standing_cannot_release_after_success_response():
    """服务成功不覆盖过期的状态消息。"""
    guard = StandGuard()
    feed(guard)
    guard.requested(0.2)
    feed(guard, 0.3, 111)
    guard.result(True, 111, 0)
    assert guard.command(2.0) == 'WAIT'
    feed(guard, 2.1, 111)
    assert guard.command(2.3) == 'WALK'


@pytest.mark.parametrize('timer_in_gap', [False, True])
def test_posture_gap_after_ready_latches_stop(timer_in_gap):
    """行走后状态断流，即使回调先恢复也不能自动继续。"""
    guard = StandGuard()
    feed(guard, motion_id=111)
    assert guard.command(0.2) == 'WALK'
    if timer_in_gap:
        assert guard.command(1.5) == 'WAIT'
    feed(guard, 1.6, 111)
    assert guard.command(1.8) == 'WAIT'
    assert guard.state == 'POSTURE_FAILED'


def test_normal_gait_transition_allowed_but_sitting_again_latches_stop():
    """进入当前步态允许正常切换；后来坐下不自动反抗人工操作。"""
    guard = StandGuard()
    feed(guard, motion_id=111)
    assert guard.command(0.2) == 'WALK'
    guard.update(303, 1, 0, True, 0.3)
    assert guard.command(0.3) == 'WALK'
    guard.update(303, 0, 0, True, 0.4)
    assert guard.command(0.4) == 'WALK'
    guard.update(143, 0, 100, True, 0.5)
    assert guard.command(0.5) == 'WAIT'
    assert guard.state == 'POSTURE_FAILED'


@pytest.mark.parametrize('value', [0, -1, math.inf, math.nan])
def test_invalid_timeouts_rejected(value):
    """非法超时不能静默关闭新鲜度检查。"""
    with pytest.raises(ValueError):
        StandGuard(status_timeout=value)
    with pytest.raises(ValueError):
        StandGuard(stand_timeout=value)
