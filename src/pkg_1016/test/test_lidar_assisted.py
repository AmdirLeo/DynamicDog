"""雷达辅助避障测试；扫描数组为合成数据，不冒充实机验证。"""

import math

import pytest

from pkg_1016.lidar_assisted import LidarAssistedController, LidarConfig
from pkg_1016.lidar_assisted import control_sectors
from pkg_1016.topic_logic import AvoidanceConfig


def ranges_for(front=1.0, left=1.5, right=0.8):
    """生成 −90°～90°、1° 间隔的扫描。"""
    result = []
    for degree in range(-90, 91):
        result.append(left if degree >= 30 else
                      right if degree < -30 else front)
    return result


def scan(controller, now, front=1.0, left=1.5, right=0.8,
         frame='laser_frame', ranges=None):
    """直接模拟 ROS 扫描回调传入核心算法。"""
    controller.update_scan(
        ranges_for(front, left, right) if ranges is None else ranges,
        -math.pi / 2, math.pi / 2, math.pi / 180,
        0.01, 30.0, frame, now)


def tick(controller, now, ultrasound=1.0, **kwargs):
    """两项传感器先更新，再执行控制周期。"""
    controller.update_range(ultrasound, 0.1, 1.0, now)
    scan(controller, now, **kwargs)
    return controller.command(now)


def moving():
    """进入超声波确认空旷的前进状态。"""
    controller = LidarAssistedController()
    for now in (0.0, 0.1, 0.2, 0.3):
        command = tick(controller, now)
    assert command == (0.15, 0.0)
    return controller


def turning(left=1.5, right=0.8):
    """前向遇障，等待制动后选择转向侧。"""
    controller = LidarAssistedController()
    assert tick(controller, 0.0, ultrasound=0.3,
                left=left, right=right) == (0.0, 0.0)
    assert tick(controller, 0.5, ultrasound=0.3,
                left=left, right=right)[0] == 0.0
    assert controller.state == 'TURN'
    return controller


def test_start_requires_new_ultrasound_samples_only():
    """定时器和扫描均不能代替超声波的新样本；前进不依赖雷达。"""
    controller = LidarAssistedController()
    assert controller.command(0.0) == (0.0, 0.0)
    controller.update_range(1.0, 0.1, 1.0, 0.0)
    for now in (0.0, 0.1, 0.2):
        scan(controller, now)
        assert controller.command(now) == (0.0, 0.0)
    controller.update_range(1.0, 0.1, 1.0, 0.3)
    assert controller.command(0.3) == (0.0, 0.0)
    controller.update_range(1.0, 0.1, 1.0, 0.4)
    assert controller.command(0.4) == (0.15, 0.0)


@pytest.mark.parametrize('scan_first', [False, True])
def test_sensor_callback_order_does_not_prevent_start(scan_first):
    """两项回调先后顺序不影响最终空旷确认。"""
    controller = LidarAssistedController()
    for i in range(5):
        now = i / 10
        if scan_first:
            scan(controller, now)
        controller.update_range(1.0, 0.1, 1.0, now)
        if not scan_first:
            scan(controller, now)
        command = controller.command(now)
    assert command == (0.15, 0.0)


def test_ultrasound_stops_forward_motion_with_clear_lidar():
    """超声波发现前方障碍即停车，雷达空旷不能否决。"""
    controller = moving()
    assert tick(controller, 0.4, ultrasound=0.3) == (0.0, 0.0)
    assert controller.state == 'BRAKE'
    assert tick(controller, 0.9, ultrasound=0.3) == (0.0, 0.5)


def test_ultrasound_hysteresis_preserves_forward_motion():
    """只有超声波控制前进滞回，雷达前方过近不影响。"""
    controller = moving()
    assert tick(controller, 0.4, ultrasound=0.5, front=0.02) == (0.15, 0.0)
    assert tick(controller, 0.5, ultrasound=0.4, front=1.0) == (0.0, 0.0)


@pytest.mark.parametrize('front', [0.02, 0.18, math.nan, math.inf])
def test_bad_lidar_front_does_not_block_start_turn_or_recovery(front):
    """前方近点和未知不阻止起步、避障转向或恢复，原值仍保留诊断。"""
    controller = LidarAssistedController()
    for now in (0.0, 0.1, 0.2):
        command = tick(controller, now, front=front)
    assert command == (0.15, 0.0)
    sector = controller.scan_report['sectors']['front']
    assert sector['distance_m'] == (front if math.isfinite(front) else None)
    assert tick(controller, 0.3, ultrasound=0.3, front=front) == (0.0, 0.0)
    assert tick(controller, 0.8, ultrasound=0.3, front=front) == (0.0, 0.5)
    for now in (0.9, 1.0):
        assert tick(controller, now, front=front) == (0.0, 0.0)
    assert tick(controller, 1.1, front=front) == (0.15, 0.0)


@pytest.mark.parametrize('left,right,expected', [
    (1.5, 0.8, 0.5), (0.8, 1.5, -0.5),
    (1.0, 1.05, 0.5), (math.nan, 1.5, -0.5),
    (1.5, math.nan, 0.5),
])
def test_direction_uses_more_open_known_side(left, right, expected):
    """比较有效侧方最近距离；未知方向不能被选中。"""
    controller = turning(left, right)
    assert controller.command(0.5) == (0.0, expected)


def test_nearly_equal_sides_use_configured_preference():
    """距离接近时 turn_speed 的符号决定初始偏好。"""
    controller = LidarAssistedController(AvoidanceConfig(turn_speed=-0.5))
    tick(controller, 0.0, ultrasound=0.3, left=1.0, right=1.0)
    assert tick(controller, 0.5, ultrasound=0.3,
                left=1.0, right=1.0) == (0.0, -0.5)


def test_selected_direction_is_not_flipped_by_distance_changes():
    """已选左转后右方变得更空旷，也不反复改向。"""
    controller = turning()
    for i in range(6, 606):
        assert tick(controller, i / 10, ultrasound=0.3,
                    left=0.8, right=2.0) == (0.0, 0.5)


@pytest.mark.parametrize('left', [0.3, math.nan])
def test_selected_side_blocked_waits_without_switching(left):
    """选定侧被挡或缺测时停车，即使另一侧空旷也不突然改向。"""
    controller = turning()
    assert tick(controller, 0.6, ultrasound=0.3,
                left=left, right=2.0) == (0.0, 0.0)
    assert controller.state == 'BLOCKED'
    assert controller.turn_direction == 1
    assert tick(controller, 0.7, ultrasound=0.3) == (0.0, 0.0)
    assert tick(controller, 1.2, ultrasound=0.3) == (0.0, 0.5)


@pytest.mark.parametrize('left,right', [(0.3, 0.3), (math.nan, math.inf)])
def test_no_available_side_stops(left, right):
    """两侧过近或未知时不原地盲转。"""
    controller = LidarAssistedController()
    assert tick(controller, 0.0, ultrasound=0.3,
                left=left, right=right) == (0.0, 0.0)
    assert controller.state == 'BLOCKED'
    assert controller.turn_direction is None


def test_clear_front_resumes_and_releases_direction_lock():
    """恢复前进后，下一次避障可以重新选择另一侧。"""
    controller = turning()
    assert tick(controller, 0.6) == (0.0, 0.0)
    assert tick(controller, 0.7) == (0.0, 0.0)
    assert tick(controller, 0.8) == (0.15, 0.0)
    assert controller.turn_direction is None
    tick(controller, 0.9, ultrasound=0.3, left=0.8, right=1.5)
    tick(controller, 1.2, ultrasound=0.3, left=0.8, right=1.5)
    assert tick(controller, 1.5, ultrasound=0.3,
                left=0.8, right=1.5) == (0.0, -0.5)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_side_emergency_blocks_turning_only(side):
    """两侧近点保留，只在需要转向时阻止运动。"""
    controller = moving()
    assert tick(controller, 0.4, **{side: 0.02}) == (0.15, 0.0)
    assert tick(controller, 0.5, ultrasound=0.3, **{side: 0.02}) == (0.0, 0.0)
    assert controller.state == 'LIDAR_TOO_CLOSE'


def test_ultrasonic_emergency_and_invalid_remain_stop_conditions():
    """雷达空旷不能覆盖超声波的超近或无效读数。"""
    controller = moving()
    assert tick(controller, 0.4, ultrasound=0.2) == (0.0, 0.0)
    assert controller.state == 'TOO_CLOSE'
    assert tick(controller, 0.5, ultrasound=math.inf) == (0.0, 0.0)
    assert controller.state == 'WAIT_SENSOR'


def test_missing_scan_allows_forward_but_stops_at_ultrasound_obstacle():
    """雷达未接收不阻止超声波起步，但不能盲目转向。"""
    controller = LidarAssistedController()
    for now in (0.0, 0.1, 0.2):
        controller.update_range(1.0, 0.1, 1.0, now)
        command = controller.command(now)
    assert command == (0.15, 0.0)
    controller.update_range(0.3, 0.1, 1.0, 0.3)
    assert controller.command(0.3) == (0.0, 0.0)
    assert controller.state == 'WAIT_SCAN'


def test_scan_timeout_does_not_stop_forward_with_fresh_ultrasound():
    """雷达超时不会清空超声波确认或阻止直行；需要转向时停车。"""
    controller = moving()
    for now in (0.4, 0.6, 0.9):
        controller.update_range(1.0, 0.1, 1.0, now)
        assert controller.command(now) == (0.15, 0.0)
    controller.update_range(0.3, 0.1, 1.0, 1.0)
    assert controller.command(1.0) == (0.0, 0.0)
    assert controller.state == 'WAIT_SCAN'


@pytest.mark.parametrize('timer_during_gap', [False, True])
def test_scan_gap_during_turn_requires_braking_on_recovery(timer_during_gap):
    """转向时雷达断流须停；恢复前后均运行或跳过定时器都要重新制动。"""
    controller = turning()
    controller.update_range(0.3, 0.1, 1.0, 0.8)
    controller.update_range(0.3, 0.1, 1.0, 1.1)
    if timer_during_gap:
        assert controller.command(1.1) == (0.0, 0.0)
        assert controller.state == 'WAIT_SCAN'
    scan(controller, 1.1)
    assert controller.command(1.1) == (0.0, 0.0)
    assert controller.state == 'BRAKE'
    tick(controller, 1.4, ultrasound=0.3)
    assert tick(controller, 1.7, ultrasound=0.3) == (0.0, 0.5)
    assert controller.turn_direction == 1


def test_ultrasound_gap_without_timer_requires_new_confirmations():
    """超声波断流后即使回调先恢复，也不能沿用前进状态。"""
    controller = moving()
    assert tick(controller, 1.0) == (0.0, 0.0)
    assert controller.clear_count == 1
    assert tick(controller, 1.1) == (0.0, 0.0)
    assert tick(controller, 1.2) == (0.15, 0.0)


def test_ultrasound_timeout_stops_despite_fresh_scan():
    """雷达不能代替断流超声波执行前进。"""
    controller = moving()
    scan(controller, 0.9)
    assert controller.command(0.9) == (0.0, 0.0)
    assert controller.state == 'WAIT_SENSOR'


def test_wrong_frame_blocks_turn_but_not_forward():
    """错误坐标系不影响超声波直行，但不能用于选择转向。"""
    controller = moving()
    assert tick(controller, 0.4, frame='wrong_frame') == (0.15, 0.0)
    assert tick(controller, 0.5, ultrasound=0.3, frame='wrong_frame') == (0.0, 0.0)
    assert controller.state == 'WAIT_SCAN'


@pytest.mark.parametrize('failure', ['frame', 'metadata', 'near', 'unknown', 'blocked'])
def test_bad_scan_replaced_before_timer_still_interrupts_turn(failure):
    """转向侧失效后立即恢复，也需要重新制动，不能漏过异常帧。"""
    controller = turning()
    if failure == 'metadata':
        controller.update_scan([], 0, 0, 0, 0.01, 30, 'laser_frame', 0.6)
    else:
        kwargs = {'frame': 'wrong_frame'} if failure == 'frame' else {
            'left': {'near': 0.02, 'unknown': math.nan, 'blocked': 0.3}[failure]}
        scan(controller, 0.6, **kwargs)
    scan(controller, 0.61)
    controller.update_range(0.3, 0.1, 1.0, 0.62)
    assert controller.command(0.62) == (0.0, 0.0)
    assert controller.state == 'BRAKE'
    assert controller.turn_direction == 1


def test_sparse_side_is_unknown_but_near_return_still_blocks_turn():
    """一个远点不能确认转向空间，一个近点仍可阻止转向。"""
    controller = LidarAssistedController()
    values = [math.nan] * 181
    values[150] = 1.0
    assert tick(controller, 0.0, ultrasound=0.3, ranges=values) == (0.0, 0.0)
    assert controller.state == 'BLOCKED'
    values[150] = 0.02
    assert tick(controller, 0.1, ultrasound=0.3, ranges=values) == (0.0, 0.0)
    assert controller.state == 'LIDAR_TOO_CLOSE'


def test_front_recovery_works_even_with_unavailable_sides():
    """超声波确认前方空旷即可解除转向受阻；雷达前方及侧方无需确认。"""
    controller = turning()
    assert tick(controller, 0.6, ultrasound=0.3, left=math.nan) == (0.0, 0.0)
    for now in (0.7, 0.8, 0.9):
        command = tick(controller, now, front=0.02, left=math.nan, right=math.nan)
    assert command == (0.15, 0.0)
    assert controller.turn_direction is None


def test_partial_angular_coverage_cannot_be_treated_as_full_sector():
    """只覆盖正前方 10° 的扫描不足以确认整个前方区域。"""
    report = control_sectors(
        [1.0] * 11, -math.pi / 36, math.pi / 36,
        math.pi / 180, 0.01, 30.0, LidarConfig())
    assert not report['sectors']['front']['known']
    assert not report['sectors']['left']['known']


def test_half_valid_scan_is_usable_but_more_missing_points_are_unknown():
    """覆盖完整时仍检查有效点比例，而非只检查存在有效点。"""
    values = ranges_for()
    for i in range(0, len(values), 3):
        values[i] = math.nan
    report = control_sectors(values, -math.pi / 2, math.pi / 2,
                            math.pi / 180, 0.01, 30.0, LidarConfig())
    assert all(s['known'] for s in report['sectors'].values())
    values = [1.0 if i % 3 == 0 else math.nan for i in range(181)]
    report = control_sectors(values, -math.pi / 2, math.pi / 2,
                            math.pi / 180, 0.01, 30.0, LidarConfig())
    assert not any(s['known'] for s in report['sectors'].values())


def test_yaw_offset_maps_reversed_mounting_to_body_forward():
    """雷达 x 轴反向安装时，可用 pi 偏航修正其后半周观测。"""
    config = LidarConfig(scan_yaw_offset=math.pi)
    report = control_sectors(
        ranges_for(), math.pi / 2, 3 * math.pi / 2,
        math.pi / 180, 0.01, 30.0, config)
    assert all(s['known'] for s in report['sectors'].values())
    assert report['sectors']['left']['distance_m'] == 1.5
    assert report['sectors']['right']['distance_m'] == 0.8


def test_negative_angle_increment_preserves_directions():
    """反序扫描仍正确解析左右。"""
    report = control_sectors(
        list(reversed(ranges_for())), math.pi / 2, -math.pi / 2,
        -math.pi / 180, 0.01, 30.0, LidarConfig())
    assert report['sectors']['left']['distance_m'] == 1.5
    assert report['sectors']['right']['distance_m'] == 0.8


@pytest.mark.parametrize('kwargs', [
    {'angle_increment': 0.0}, {'angle_min': math.nan},
    {'angle_max': 0.0}, {'range_min': 30.0}, {'range_max': math.inf},
])
def test_bad_scan_metadata_is_rejected(kwargs):
    """角度终点、采样数或量程矛盾时拒绝该帧。"""
    args = dict(ranges=ranges_for(), angle_min=-math.pi / 2,
                angle_max=math.pi / 2, angle_increment=math.pi / 180,
                range_min=0.01, range_max=30.0, config=LidarConfig())
    args.update(kwargs)
    assert control_sectors(**args)['status'] == 'INVALID'


@pytest.mark.parametrize('kwargs', [
    {'scan_timeout': 0}, {'turn_clearance': math.nan},
    {'direction_margin': -1}, {'min_valid_points': 0},
    {'min_valid_points': 1.5}, {'min_valid_fraction': 0},
    {'min_coverage': 1.1}, {'scan_yaw_offset': math.inf},
    {'scan_frame_id': ''},
])
def test_invalid_lidar_configuration_is_rejected(kwargs):
    """非法配置不能静默降低缺测与近障判断要求。"""
    with pytest.raises(ValueError):
        LidarConfig(**kwargs)


def test_turn_clearance_must_exceed_emergency_distance():
    """转向放行阈值不能落入超近停车范围。"""
    with pytest.raises(ValueError):
        LidarAssistedController(lidar_config=LidarConfig(turn_clearance=0.1))
