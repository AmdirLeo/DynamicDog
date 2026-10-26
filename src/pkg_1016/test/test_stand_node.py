"""使用 ROS 替身验证站立服务与步态发布互斥，不连接机器狗。"""

from concurrent.futures import Future
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


class Message:
    """替代可赋值的 ROS 消息。"""

    SERVO_DATA = 1
    SERVO_END = 2
    ULTRASOUND = 0


class Publisher:
    """记录发布内容。"""

    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)

    def get_subscription_count(self):
        return 1


class Client:
    """可控制服务发现与响应时机。"""

    def __init__(self):
        self.requests = []
        self.future = Future()
        self.available = True

    def service_is_ready(self):
        return self.available

    def call_async(self, request):
        self.requests.append(request)
        return self.future


@pytest.fixture
def node_factory(monkeypatch):
    """隔离导入替身，不污染其他测试对真实包的使用。"""
    params = {}

    class Node:
        def __init__(self, name):
            self.publishers = {}
            self.clients = {}
            self.subscriptions = {}

        def create_publisher(self, kind, topic, qos):
            publisher = Publisher()
            self.publishers[topic] = publisher
            return publisher

        def create_client(self, kind, service):
            client = Client()
            self.clients[service] = client
            return client

        def create_subscription(self, kind, topic, callback, qos):
            self.subscriptions[topic] = callback
            return callback

        def create_timer(self, period, callback, clock):
            return SimpleNamespace(cancel=lambda: None)

        def get_namespace(self):
            return '/test_dog'

        def get_logger(self):
            return SimpleNamespace(info=lambda msg: None)

    replacements = {
        'protocol.msg': SimpleNamespace(MotionServoCmd=Message, MotionStatus=Message),
        'protocol.srv': SimpleNamespace(MotionResultCmd=SimpleNamespace(Request=Message)),
        'rclpy.node': SimpleNamespace(Node=Node),
        'rclpy.clock': SimpleNamespace(Clock=lambda **kwargs: None,
                                       ClockType=SimpleNamespace(STEADY_TIME=1)),
        'sensor_msgs.msg': SimpleNamespace(LaserScan=Message, Range=Message),
        'std_msgs.msg': SimpleNamespace(String=Message),
        'pkg_1016.ros_helpers': SimpleNamespace(
            latest_sensor_qos=lambda: None, run_node=None,
            startup_parameter=lambda node, name, default: params.get(name, default)),
    }
    for name, replacement in replacements.items():
        monkeypatch.setitem(sys.modules, name, replacement)
    path = Path(__file__).parents[1] / 'pkg_1016' / 'ultrasonic_avoidance.py'
    spec = importlib.util.spec_from_file_location('stand_node_under_test', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = [0.0]
    module.time = SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda seconds: None)

    def make(enable_motion=True, auto_stand=True, use_lidar=False):
        params.update(enable_motion=enable_motion, auto_stand=auto_stand,
                      use_lidar=use_lidar)
        return module.UltrasonicAvoidance(), clock

    return make


def tick(node, clock, now, motion_id=143, progress=100, switch_status=0):
    """模拟一个包含姿态和超声波采样的控制周期。"""
    clock[0] = now
    if node.auto_stand:
        node.on_posture(SimpleNamespace(
            motion_id=motion_id, order_process_bar=progress,
            switch_status=switch_status, ori_error=0, footpos_error=0,
            motor_error=[0] * 12))
    node.on_range(SimpleNamespace(header=SimpleNamespace(frame_id='ultrasonic'),
                                 range=0.8, min_range=0.1, max_range=1.0,
                                 radiation_type=0))
    node.on_timer()


def test_observation_never_calls_service_or_publishes_motion(node_factory):
    """即使要求自动站立，只观测也不能产生硬件动作。"""
    node, clock = node_factory(enable_motion=False)
    for now in (0.0, 0.1, 0.2, 0.3):
        tick(node, clock, now)
    assert node.clients == {}
    assert 'motion_servo_cmd' not in node.publishers
    assert node.stand_guard.requested_at is None


def test_service_then_standing_then_new_clear_samples_then_walk(node_factory):
    """起身期间无步态，成功且站稳后重新累计前方空旷样本才前进。"""
    node, clock = node_factory()
    client = node.stand_client
    publisher = node.motion_pub
    for now in (0.0, 0.1, 0.2, 0.3):
        tick(node, clock, now)
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request.motion_id == 111
    assert request.duration == 0
    assert request.cmd_source == 0
    assert len(request.acc_des) == len(request.foot_pose) == 6
    assert publisher.messages == []
    for now in (0.4, 0.5, 0.6):
        tick(node, clock, now, motion_id=111)
    assert publisher.messages == []  # 站稳反馈不能替代服务结果。
    client.future.set_result(SimpleNamespace(result=True, motion_id=111, code=0))
    tick(node, clock, 0.7, motion_id=111)
    assert publisher.messages[-1].vel_des == [0.0, 0.0, 0.0]
    tick(node, clock, 0.8, motion_id=303)
    assert publisher.messages[-1].vel_des == [0.0, 0.0, 0.0]
    tick(node, clock, 0.9, motion_id=303)
    assert publisher.messages[-1].vel_des == [0.15, 0.0, 0.0]
    tick(node, clock, 1.0, motion_id=143)
    assert publisher.messages[-1].cmd_type == Message.SERVO_END
    count = len(publisher.messages)
    tick(node, clock, 1.1, motion_id=143)
    assert len(publisher.messages) == count
    assert len(client.requests) == 1


def test_missing_service_waits_without_motion(node_factory):
    """服务尚未发现时不发步态，也不提交请求。"""
    node, clock = node_factory()
    node.stand_client.available = False
    for now in (0.0, 0.1, 0.2, 0.3):
        tick(node, clock, now)
    assert node.stand_client.requests == []
    assert node.motion_pub.messages == []


def test_failed_service_never_starts_gait(node_factory):
    """站立服务失败后即使姿态报告站立，也不自行继续。"""
    node, clock = node_factory()
    for now in (0.0, 0.1, 0.2):
        tick(node, clock, now)
    node.stand_client.future.set_result(SimpleNamespace(result=False, motion_id=143, code=1))
    for now in (0.3, 0.4, 0.5, 0.6):
        tick(node, clock, now, motion_id=111)
    assert node.motion_pub.messages == []
    assert len(node.stand_client.requests) == 1


def test_stop_during_standing_does_not_send_gait(node_factory):
    """退出起身等待时不发送可能打断结果动作的零速步态。"""
    node, clock = node_factory()
    for now in (0.0, 0.1, 0.2):
        tick(node, clock, now)
    node.stop()
    assert node.motion_pub.messages == []
    assert node.stand_client.future.cancelled()


def test_disabled_feature_preserves_original_start_and_stop(node_factory):
    """不启用时保持原来的手动站立运行方式与正常结束帧。"""
    node, clock = node_factory(auto_stand=False)
    for now in (0.0, 0.1, 0.2):
        tick(node, clock, now)
    assert node.clients == {}
    assert 'motion_status' not in node.subscriptions
    assert node.motion_pub.messages[-1].vel_des == [0.15, 0.0, 0.0]
    node.stop()
    assert node.motion_pub.messages[-1].cmd_type == Message.SERVO_END


def test_posture_gate_overrides_lidar_status_reason(node_factory):
    """融合避障输出须报告实际的姿态等待原因，不能误报已经前进。"""
    node, clock = node_factory(use_lidar=True)
    tick(node, clock, 0.0)
    status = json.loads(node.status_pub.messages[-1].data)
    assert status['state'] == 'WAIT_POSTURE'
    assert status['reason'] == status['posture']['reason']
    assert status['lidar']['front_used_for_control'] is False
    assert status['desired_vx_m_s'] == status['desired_wz_rad_s'] == 0
    assert node.motion_pub.messages == []


@pytest.mark.parametrize('failure', ['request', 'response', 'fields'])
def test_service_exceptions_latch_without_walking(node_factory, failure):
    """请求发送、异步结果或请求字段不兼容都不能绕过站立确认。"""
    node, clock = node_factory()

    def broken(*args):
        raise RuntimeError('simulated service failure')

    if failure == 'request':
        node.stand_client.call_async = broken
    elif failure == 'fields':
        node.stand_service_type = SimpleNamespace(Request=broken)
    for now in (0.0, 0.1, 0.2):
        tick(node, clock, now)
    if failure == 'response':
        node.stand_client.future.set_exception(RuntimeError('simulated response failure'))
    tick(node, clock, 0.3, motion_id=111)
    assert node.stand_guard.state == 'POSTURE_FAILED'
    assert node.motion_pub.messages == []
