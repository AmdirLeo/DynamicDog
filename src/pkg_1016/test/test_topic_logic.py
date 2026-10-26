"""无需 ROS/实机的算法边界测试：pytest -q test/test_topic_logic.py。"""

import math

import pytest

from pkg_1016.topic_logic import AvoidanceConfig, AvoidanceController
from pkg_1016.topic_logic import analyse_scan


def receive(controller, distance, now):
    """模拟连续量程为 0.05～4 m 的超声波。"""
    controller.update_range(distance, 0.05, 4.0, now)
    return controller.command(now)


def moving_controller():
    """用三条真实采样事件进入前进状态。"""
    controller = AvoidanceController()
    for now in (0.0, 0.1, 0.2):
        receive(controller, 1.0, now)
    return controller


def test_startup_requires_three_new_samples():
    """重复调用控制定时器不应增加空旷采样次数。"""
    controller = AvoidanceController()
    assert controller.command(0.0) == (0.0, 0.0)
    assert receive(controller, 1.0, 0.0) == (0.0, 0.0)
    for now in (0.01, 0.02, 0.03):
        assert controller.command(now) == (0.0, 0.0)
    assert controller.clear_count == 1
    assert receive(controller, 1.0, 0.1) == (0.0, 0.0)
    assert receive(controller, 1.0, 0.2) == (0.15, 0.0)


def test_hysteresis_brake_and_turn():
    """前进中超过停车阈值继续行走，遇障先制动后转向。"""
    controller = moving_controller()
    assert receive(controller, 0.50, 0.3) == (0.15, 0.0)
    assert receive(controller, 0.40, 0.4) == (0.0, 0.0)
    assert controller.state == 'BRAKE'
    assert receive(controller, 0.30, 0.8) == (0.0, 0.0)
    assert receive(controller, 0.30, 1.0) == (0.0, 0.50)
    assert controller.state == 'TURN'


def test_turn_stops_for_close_object_and_requires_clear_confirmation():
    """超近距离时不转动；移走障碍后重新确认三次。"""
    controller = AvoidanceController()
    receive(controller, 0.3, 0.0)
    receive(controller, 0.3, 0.5)
    assert receive(controller, 0.2, 0.6) == (0.0, 0.0)
    assert controller.state == 'TOO_CLOSE'
    assert receive(controller, 0.65, 0.7) == (0.0, 0.0)
    assert receive(controller, 0.65, 0.8) == (0.0, 0.0)
    assert receive(controller, 0.65, 0.9) == (0.15, 0.0)


def test_receive_timeout_stops_and_resets_confirmation():
    """接收间隔超过 0.5 s 停止，恢复后不能沿用旧确认计数。"""
    controller = moving_controller()
    assert controller.command(0.71) == (0.0, 0.0)
    assert controller.state == 'WAIT_SENSOR'
    assert controller.clear_count == 0
    assert receive(controller, 1.0, 0.8) == (0.0, 0.0)
    assert receive(controller, 1.0, 0.9) == (0.0, 0.0)
    assert receive(controller, 1.0, 1.0) == (0.15, 0.0)


def test_sampling_gap_resets_count_even_without_timer():
    """即使定时器没有在断流期间运行，采样入口仍重置计数。"""
    controller = AvoidanceController()
    receive(controller, 1.0, 0.0)
    receive(controller, 1.0, 0.1)
    assert receive(controller, 1.0, 1.0) == (0.0, 0.0)
    assert controller.clear_count == 1


@pytest.mark.parametrize('distance', [
    math.nan, math.inf, -math.inf, -1.0, 4.1,
])
def test_invalid_range_immediately_stops(distance):
    """无效单帧不允许维持上一帧非零速度。"""
    controller = moving_controller()
    assert receive(controller, distance, 0.3) == (0.0, 0.0)
    assert controller.distance is None


@pytest.mark.parametrize('minimum,maximum', [
    (0.3, 0.3), (1.0, 0.5), (-0.1, 4.0), (0.0, math.inf),
])
def test_invalid_range_metadata_stops(minimum, maximum):
    """不把固定距离二值传感器或非法量程当作连续测距。"""
    controller = moving_controller()
    assert not controller.update_range(0.3, minimum, maximum, 0.3)
    assert controller.command(0.3) == (0.0, 0.0)


def test_continuous_obstacle_keeps_turning_then_clear_samples_resume():
    """持续有效遇障超过 60 秒仍转向，三条新空旷采样后前进。"""
    controller = AvoidanceController()
    receive(controller, 0.3, 0.0)
    assert receive(controller, 0.3, 0.5) == (0.0, 0.50)
    for i in range(6, 606):
        assert receive(controller, 0.3, i / 10) == (0.0, 0.50)
        assert controller.state == 'TURN'
    for now in (60.6, 60.7):
        assert receive(controller, 1.0, now) == (0.0, 0.0)
    assert receive(controller, 1.0, 60.8) == (0.15, 0.0)
    assert controller.state == 'FORWARD'


def test_dropout_stops_turning_and_valid_obstacle_resumes_after_brake():
    """持续转向仍受断流保护，恢复有效遇障数据后先制动再转向。"""
    controller = AvoidanceController()
    receive(controller, 0.3, 0.0)
    receive(controller, 0.3, 0.5)
    assert controller.command(1.1) == (0.0, 0.0)
    assert controller.state == 'WAIT_SENSOR'
    assert receive(controller, 0.3, 12.0) == (0.0, 0.0)
    assert controller.state == 'BRAKE'
    assert receive(controller, 0.3, 12.5) == (0.0, 0.50)
    assert controller.state == 'TURN'


@pytest.mark.parametrize('distance,state', [
    (0.2, 'TOO_CLOSE'), (math.inf, 'WAIT_SENSOR'),
])
def test_turn_stops_for_close_or_invalid_sample(distance, state):
    """删除转向时限不影响超近距离和无效数据的停车保护。"""
    controller = AvoidanceController()
    receive(controller, 0.3, 0.0)
    for i in range(1, 101):
        receive(controller, 0.3, i / 10)
    assert controller.state == 'TURN'
    assert receive(controller, distance, 10.1) == (0.0, 0.0)
    assert controller.state == state


def test_configuration_defaults_and_startup_mapping():
    """启动参数映射保留已调速度，配置对象不可变。"""
    config = AvoidanceConfig()
    assert config.forward_speed == 0.15
    assert config.turn_speed == 0.50
    assert len(config._asdict()) == 8
    assert AvoidanceConfig(**config._asdict()) == config
    with pytest.raises(AttributeError):
        config.forward_speed = 0.2


@pytest.mark.parametrize('kwargs', [
    {'stop_distance': 0.1}, {'clear_distance': 0.4},
    {'forward_speed': 0.31}, {'sensor_timeout': math.nan},
    {'brake_duration': 0.0}, {'turn_speed': 0.0},
    {'turn_speed': math.inf}, {'turn_speed': 0.81},
    {'clear_samples': 0}, {'clear_samples': 2.5},
])
def test_invalid_configuration_rejected(kwargs):
    """拒绝破坏阈值逻辑或超出本实验速度上限的配置。"""
    with pytest.raises(ValueError):
        AvoidanceConfig(**kwargs)


def test_negative_turn_direction_is_supported():
    """可在确认实机方向后用负参数切换固定转向方向。"""
    controller = AvoidanceController(AvoidanceConfig(turn_speed=-0.35))
    receive(controller, 0.3, 0.0)
    assert receive(controller, 0.3, 0.5) == (0.0, -0.35)


def test_scan_directions_and_warning_boundary():
    """四条正交射线分别映射到前左右后，等于阈值也报警。"""
    result = analyse_scan([1.0, 0.4, 2.0, 0.2], 0.0, math.pi / 2, 0.1, 5.0)
    assert result['status'] == 'OK'
    sectors = result['sectors']
    for name, distance in zip(('front', 'left', 'rear', 'right'),
                              (1.0, 0.4, 2.0, 0.2)):
        assert sectors[name]['distance_m'] == distance
        assert sectors[name]['valid_points'] == 1
        assert sectors[name]['near'] == (distance <= 0.4)


def test_partial_scan_keeps_unobserved_sectors_unknown():
    """局部视场不能推断未覆盖方向无障碍。"""
    result = analyse_scan([1.0, 0.3, 2.0], -0.1, 0.1, 0.1, 5.0)
    assert result['sectors']['front']['distance_m'] == 0.3
    assert result['sectors']['front']['valid_points'] == 3
    for name in ('left', 'rear', 'right'):
        assert result['sectors'][name] == {
            'distance_m': None, 'valid_points': 0, 'near': None,
        }


def test_invalid_scan_samples_are_not_clear_space():
    """NaN/Inf/超量程全部丢弃，不能用无穷代替未知。"""
    result = analyse_scan(
        [math.nan, math.inf, -math.inf, 0.01, 8.0], 0.0, 0.1, 0.1, 5.0)
    assert result['status'] == 'NO_VALID_POINTS'
    assert all(s['near'] is None for s in result['sectors'].values())


@pytest.mark.parametrize('angle_min,increment,minimum,maximum', [
    (0.0, 0.0, 0.1, 5.0), (math.nan, 0.1, 0.1, 5.0),
    (0.0, 0.1, 1.0, 1.0), (0.0, 0.1, 0.1, math.inf),
])
def test_bad_scan_metadata(angle_min, increment, minimum, maximum):
    """非法角度或量程不能产生看似有效的距离。"""
    result = analyse_scan([1.0], angle_min, increment, minimum, maximum)
    assert result['status'] == 'INVALID'


def test_scan_negative_increment_and_wraparound():
    """支持负角增量以及跨越一整圈后的角度归一化。"""
    result = analyse_scan([0.2, 0.3], 2 * math.pi, -math.pi / 2, 0.1, 5.0)
    assert result['sectors']['front']['distance_m'] == 0.2
    assert result['sectors']['right']['distance_m'] == 0.3
