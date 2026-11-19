# 参考代码阅读笔记：AmdirLeo/DynamicDog

> 用途：记录从 GitHub 仓库 <https://github.com/AmdirLeo/DynamicDog>（branch `master`）读到的、
> 被 `FindBallDemo.py` 采用的做法，便于以后核对与升级。
> 读取时间：2026-11-15；仓库最新提交时间 2026-11-15。
> 注意：仓库里 `build/`、`install/`、`log/` 是编译产物，下面只记 `src/` 与根目录的源码。

## 1. 仓库结构（与找球相关的部分）

| 路径 | 作用 |
|---|---|
| `racing7.py` / `racing72.py` | 根目录脚本：VRPN 动捕 + 10Hz `MotionServoCmd` 的三狗足球对抗（含踢球/守门/策应） |
| `src/pkg_1016/pkg_1016/move.py` | 最小移动节点：10Hz 发 `motion_id=303, cmd_type=1, value=2, vel_des=[0,0,0], step_height=[0.05,0.05]` |
| `src/pkg_1016/pkg_1016/lidar_observer.py` | 激光雷达观察（扇区最近距离/角度的封装） |
| `src/pkg_1016/pkg_1016/lidar_assisted.py` | 雷达辅助行走/避障 |
| `src/pkg_1016/pkg_1016/ultrasonic_avoidance.py` | 超声波避障 |
| `src/pkg_1016/pkg_1016/stand_guard.py` | 与 pkg_soccer 里同款的站立门控 |
| `src/pkg_1016/pkg_1016/topic_logic.py` | 话题/消息逻辑工具 |
| `src/pkg_soccer/pkg_soccer/soccer.py` | 动捕足球主节点（ROS 参数、话题订阅、姿态门控、定时器 0.05s、发布 servo、Ctrl+C 收尾） |
| `src/pkg_soccer/pkg_soccer/control.py` | 纯逻辑控制：Config/Pose/Controller 状态机（POSITION/ALIGN/APPROACH/SETTLE/KICK/ASSESS/FOLLOW） |
| `src/pkg_soccer/pkg_soccer/measurement.py` | 纯逻辑测量：Track 滑窗速度估计、停车/触球事件 |
| `src/pkg_soccer/pkg_soccer/stand_guard.py` | 纯逻辑姿态门控（motion_status + motion_result_cmd(111)） |
| `src/pkg_soccer/pkg_soccer/rl_control.py` + `policy_*.npz` | RL 策略推理（本程序未用） |
| `src/pkg_1106/pkg_1106/camera_hsv.py` | HSV 颜色标定/检测（本程序不用相机） |
| `output.md` | 一段 `ros2 topic echo /ultrasonic_payload` 的调试记录（不是文档） |

## 2. 直接采用的做法

### 2.1 `racing7.py`

- 订阅 `/vrpn/{dog}/pose`、`/vrpn/{ball}/pose`（`geometry_msgs/PoseStamped`），
  用 `create_timer(0.1, main_loop)` 10 Hz 发 `/<dog>/motion_servo_cmd`；
- `MotionServoCmd` 字段：`motion_id=308`（足球步态）、`cmd_type=1`、`value=2`、
  `step_height=[0.05, 0.05]`；
- **航向不是直接取 yaw**：`EulerAndQuaternionTransform()` 返回 `[roll, pitch, yaw]`，
  实际把 `roll` 当狗头航向（`self.dire2 = roll`）；
- 每只狗有固定角度偏置（注释：x 轴相对狗 48°，另一只 -7°），`cal(x0,y0,x,y,base,dire)`
  先算相对世界 x 轴的角度、再减航向并归一化；
- `toward(x)` 分级转向：|误差| > 12° → ±0.4，> 5° → ±0.2，否则 0（度制）；
- 距离分级速度：r > 1.5 → vx=0.6，r > 0.7 → 0.3，r < 0.5 → -0.1（后退），否则 0；
- `vy` 用于横向对准（±0.2 / ±0.3），并带"和球/门保持水平"的死区 ±0.1 m；
- 前冲触球用 while + 时间窗口，发 `[1.0, 0, 0]` 若干帧（本程序没有照搬，只借鉴思路）。

### 2.2 `pkg_soccer/control.py`

- `Config`：`approach_speed=0.45`、`turn_speed=0.7`、`angle_tolerance=0.25`、
  `pose_timeout=0.5`、`kick_distance=0.38`、`min_distance=0.20`、`ball_clearance=0.48`、
  `follow_distance=0.32`、`ball_still_speed=0.08`、`max_run_time=120`；
  并 `validate()` 拒绝 >0.8 m/s 的前进/触球速度、>1.0 rad/s 转向等；
- `Pose` 同时保存 `received`（本机接收时刻）与 `stamp`（源时间戳）、`frame`、`sequence`；
  `update()` 只在"新消息"时刷新新鲜度（VRPN stamp 复用也不误判）；
- `poses_valid()` 要求狗/球**同一 frame** 且各自新鲜；
- `turn_rate()`：误差在容差内 0，否则 `min(turn_speed, max(0.20, 1.5*|error|))`，
  并明确"避免指令小到无法转动"；
- `drive_to()`：`scale = min(speed, max(0.04, 1.5*distance)) / distance`，
  世界速度→机身速度（含 `vy`），并预测 0.5 s 内是否出场地；
- `command()` 对转向做反馈校验：换向后有 `turn_response_grace` 宽限；宽限后
  反向超过 0.10 rad 或 3 s 无进展 → `stop()`；连续原地转 12 s → `stop()`
  （提示检查刚体朝向与 `dog_yaw_offset`）；
- 场地边界/球门/绕球（`route_target`）逻辑本程序不需要，未采用。

### 2.3 `pkg_soccer/measurement.py`

- `Track`：`deque` 保存约 0.1 s 样本做线性回归，`update()` 拒绝：
  非有限值、`stamp<0`、`dt<=0`、`dt>gap(0.20)`、跨 frame、源时钟回退、
  位置跳变 > `max_speed*dt+0.03`；`valid()` 要求速度存在、质量 ok、`0<=now-latest<=gap`；
- 球速上限 12 m/s、狗速上限 3 m/s；`basis` 标记用源时间戳还是接收时间做回归；
- 触球检测：球先静（<0.08 m/s 持续 0.3 s）后突然 >0.20 m/s，且距狗 <0.65 m。
  本程序只用了"速度估计 + 跳点/断流剔除"，未做触球事件。

### 2.4 `pkg_soccer/stand_guard.py`

- `StandGuard(walk_id, status_timeout=1.0, stand_timeout=25.0)`，
  状态 WAIT_POSTURE/STANDING/READY/POSTURE_FAILED；
- `_standing()`：`motion_id==112` 或 `motion_id==111 and progress>=96`；
- 启动阶段坐/趴（101/143）→ 返回 `REQUEST`，主程序发一次 `motion_result_cmd(motion_id=111)`；
- 服务成功也**不直接放行**，还要新鲜且连续 3 帧站稳；
- 运行中姿态改变/数据断流/故障 → `fail()` 锁停，不自动重试；
- `soccer.py` 里在 `on_timer` 每拍先过门控，`publish()` 发 `303/305`，
  退出时 `publish(ZERO, end=True)` 发 `SERVO_END` 3 帧。

### 2.5 `src/pkg_1016/pkg_1016/move.py`

- 最小示例：`create_timer(0.1)` + 固定 `motion_id=303`、`cmd_type=1`、`value=2`、
  `step_height=[0.05,0.05]`、`vel_des=[0,0,0]`。

## 3. 本程序的对应实现（FindBallDemo.py）

| 参考点 | 本程序实现 |
|---|---|
| roll 当航向 + 角度偏置 | `--yaw-axis {yaw,roll,pitch}` + `--yaw-offset <度>`；`euler_from_quaternion()` |
| 位姿接收时间/源时间戳分离 | 回调里用 `time.monotonic()` 记 `_dog_stamp/_ball_stamp`；`header.stamp` 不参与新鲜度 |
| 同 frame 校验 | `observation()` 比较 `_dog_frame/_ball_frame`，不一致按无观测 |
| 连续帧确认 | `--confirm-samples`（默认 3），起步前连续有效帧 |
| 转向整形 | `turn_rate()`：死区 + 最小 0.20 rad/s + 限幅 |
| 转向反馈保护 | `TurnMonitor`：持续转向 3 s 无 yaw 进展 → 停车 |
| 按距离减速 | `_control()`：`min(max_speed, max(0.04, 0.8*剩余距离))` |
| vy 横向微调 | `--max-vy`（默认 0，参考实现用 0.2~0.3） |
| 球速/预测 | `VelocityTracker`（dt 用接收时间、跳点/断流/超速剔除）+ `--lead-time` |
| 姿态门控 | `StandGate`：111/112/行走步态放行；坐/趴请求一次 111；故障锁停 |
| END 帧 | `end_motion()`：3 帧 `cmd_type = MotionServoCmd.SERVO_END`（无常量时退化为 1） |
| 步态 | `--motion-id`（默认 303；参考 racing 用 308、soccer 用 303/305） |

## 4. 未采用/未验证

- RL 策略（`rl_control.py` + `policy_*.npz`）需要训练时的观测/动作约定，未接入；
- 足球场的球门/场地/绕球状态机只服务于射门，找球不需要；
- 多狗协同、UWB、视觉 HSV 等未使用（本程序只做动捕找球）；
- 所有参考代码都来自公开仓库，**未在真机验证**；默认值（0.4 m 停止距离、0.3 m/s 速度、
  0.7 rad/s 转向等）需要按本机狗的腿长/步态重新标定。
