"""只读记录运动命令及动捕；最终交付为用户指定的单个 CSV 文件。"""

import csv
import math
from pathlib import Path
import signal
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from protocol.msg import MotionServoCmd
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from .control import wrap_angle, yaw_from_quaternion
from .measurement import Measurements


class MotionRecorder(Node):
    """订阅任一控制来源发到同一个 servo 话题的指令，不发布任何运动。"""

    def __init__(self):
        super().__init__('soccer_motion_recorder')
        self.data = Measurements()
        self.started = time.monotonic()
        self.last_status = -math.inf
        self.file = None
        self.closed = False
        self.rows = 0
        self.motion_id, self.cmd_type = '', ''
        self.scale = float(self.param('position_scale', 1.0))
        self.yaw_offset = float(self.param('dog_yaw_offset', 0.0))
        if not math.isfinite(self.scale) or self.scale <= 0 or not math.isfinite(self.yaw_offset):
            raise ValueError('position_scale 必须为正，dog_yaw_offset 必须有限')
        dog_name = self.param('dog_name', 'mi_desktop_48_b0_2d_7b_03_fd').strip('/')
        default_cmd = '/' + (dog_name+'/' if dog_name else '') + 'motion_servo_cmd'
        self.cmd_topic = self.param('command_topic', default_cmd)
        self.dog_topic = self.param('dog_pose_topic', '/vrpn/Lizj_dog/pose')
        self.ball_topic = self.param('ball_pose_topic', '/vrpn/ball/pose')
        output = self.param('output', '')
        if not output:
            raise ValueError('必须指定 -p output:=/绝对路径/测量.csv')
        path = Path(output).expanduser()
        if not path.is_absolute() or not path.parent.is_dir():
            raise ValueError('output 必须是绝对路径，且父目录已存在')
        qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subs = [self.create_subscription(
            PoseStamped, topic, lambda msg, k=k: self.pose(k, msg), qos)
            for k, topic in (('dog', self.dog_topic), ('ball', self.ball_topic))]
        self.subs.append(self.create_subscription(MotionServoCmd, self.cmd_topic, self.command, qos))
        self.timer = self.create_timer(.05, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        fields = ['time_s', 'event', 'note', 'motion_id', 'cmd_type',
                  'cmd_age_s', 'cmd_vx_mps', 'cmd_vy_mps', 'cmd_wz_radps']
        for k in ('dog', 'ball'):
            fields += [k+'_'+v for v in (
                'stamp_s', 'age_s', 'frame', 'x_m', 'y_m', 'yaw_rad', 'valid',
                'quality', 'time_basis', 'vx_world_mps', 'vy_world_mps', 'speed_mps')]
        fields += ['dog_vx_body_mps', 'dog_vy_body_mps', 'dog_wz_radps',
                   'stop_start_speed_mps', 'stop_distance_m', 'stop_max_excursion_m',
                   'stop_path_m', 'stop_time_s', 'ball_peak_speed_mps']
        self.file = path.open('x', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()
        self.file.flush()
        self.get_logger().info(f'只读记录至 {path}；指令={self.cmd_topic}；'
                               f'dog={self.dog_topic}；ball={self.ball_topic}')

    def param(self, name, default):
        return self.declare_parameter(name, default).value

    def pose(self, kind, msg):
        now = time.monotonic()
        try:
            p, q = msg.pose.position, msg.pose.orientation
            yaw = (wrap_angle(yaw_from_quaternion(q.x, q.y, q.z, q.w)+self.yaw_offset)
                   if kind == 'dog' else 0.)
            stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
            self.data.pose_update(kind, now, stamp, p.x*self.scale, p.y*self.scale,
                                  yaw, msg.header.frame_id.strip())
        except (ValueError, TypeError, AttributeError):
            getattr(self.data, kind).invalidate('invalid_message')
            self.data.tick(now)
        self.write(now, event=kind+'_pose')
        self.drain(now)

    def command(self, msg):
        now = time.monotonic()
        self.motion_id, self.cmd_type = msg.motion_id, msg.cmd_type
        velocity = tuple(msg.vel_des)
        if len(velocity) != 3 or not all(math.isfinite(v) for v in velocity):
            self.data.command = None
            if self.data.stop is not None:
                self.data.emit('stop_invalid', note='invalid_command')
                self.data.stop = None
            self.write(now, event='invalid_command')
            self.drain(now)
            return
        self.data.command_update(now, velocity, msg.cmd_type == MotionServoCmd.SERVO_END)
        self.write(now, event='command')
        self.drain(now)

    def write(self, now, **extra):
        row = dict(time_s=now-self.started, motion_id=self.motion_id, cmd_type=self.cmd_type)
        if self.data.command:
            received, velocity = self.data.command
            row.update(cmd_age_s=now-received, cmd_vx_mps=velocity[0],
                       cmd_vy_mps=velocity[1], cmd_wz_radps=velocity[2])
        for k in ('dog', 'ball'):
            track = getattr(self.data, k)
            valid = track.valid(now)
            row[k+'_valid'] = int(valid)
            row[k+'_quality'] = track.quality if valid or track.quality != 'ok' else 'stale'
            if track.latest:
                received, stamp, x, y, yaw, frame = track.latest
                row.update({k+'_stamp_s': stamp, k+'_age_s': now-received,
                            k+'_x_m': x, k+'_y_m': y, k+'_yaw_rad': yaw, k+'_frame': frame})
            if valid:
                vx, vy, wz = track.velocity
                row.update({k+'_vx_world_mps': vx, k+'_vy_world_mps': vy,
                            k+'_speed_mps': track.speed, k+'_time_basis': track.basis})
                if k == 'dog':
                    co, si = math.cos(track.latest[4]), math.sin(track.latest[4])
                    row.update(dog_vx_body_mps=co*vx+si*vy,
                               dog_vy_body_mps=-si*vx+co*vy, dog_wz_radps=wz)
        row.update(extra)
        self.writer.writerow(row)
        self.rows += 1

    def drain(self, now):
        while self.data.events:
            event = self.data.events.pop(0)
            self.write(now, **event)
            self.get_logger().info(str(event))

    def tick(self):
        now = time.monotonic()
        self.data.tick(now)
        self.drain(now)
        if now-self.last_status >= 2.:
            self.write(now, event='status')
            self.file.flush()
            cmd_age = now-self.data.command[0] if self.data.command else math.inf
            self.get_logger().info(
                f'已记录 {self.rows} 行；dog有效={self.data.dog.valid(now)}；'
                f'ball有效={self.data.ball.valid(now)}；指令年龄={cmd_age:.2f}s')
            self.last_status = now

    def close(self):
        if not self.closed and self.file is not None:
            self.timer.cancel()
            self.data.finish()
            self.drain(time.monotonic())
            self.file.close()
            self.closed = True


def main(args=None):
    rclpy.init(args=args)
    node = None
    stopping = threading.Event()
    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, lambda *_: stopping.set())
        node = MotionRecorder()
        while rclpy.ok() and not stopping.is_set():
            rclpy.spin_once(node, timeout_sec=.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
