# 机器狗足球（2v2）

ROS 2 足球比赛主程序：两只狗各跑一份**同一套 41 维强化学习策略**（`shared_2v2`），
靠动捕拿到自己、球、队友和两个对手的位姿，输出机身系速度指令，由运控接口执行。

场地按《机器狗大赛复赛规则》建模：10 × 5.5 m 围挡、门线 x = ±4.585 m、门宽 1.5 m、
9 个发球点（±2.5 / ±1.25 网格）、死区为距围挡 0.8 m 的条带。

---

## 1. 目录结构

```
DynamicDog/
├── start.sh                    # 比赛启动脚本（按 A1/A2/B1/B2 选角色）
├── train_2v2_large.py          # 4 狗共享策略的 PPO 训练器（PyTorch，独立仿真）
├── policy_2v2_shared.pt        # 训练产物（PyTorch 权重）
└── src/pkg_soccer/
    ├── pkg_soccer/
    │   ├── soccer.py            # 主节点：话题、姿态门控、比赛循环
    │   ├── rl_control.py        # 策略加载、41 维观测、速度/短冲、暂停恢复
    │   ├── control.py           # 状态机基类：场地几何、发球点、暂停/复位
    │   ├── attack.py            # 可选进攻规划（混合控制）
    │   ├── stand_guard.py       # 姿态门控（坐/趴/站、故障检测）
    │   ├── measurement.py       # 位姿轨迹与速度估计
    │   ├── policy_2v2_shared_actor.npz      # 部署用策略（41 维，0.45 档）
    │   ├── policy_duel_*_actor.npz          # 旧 1v1 策略（29 维）
    │   └── policy_stage2_actor.npz          # 旧单狗策略（23 维）
    ├── train/                   # 早期训练套件（槽位顺序与当前部署不同，见 §6）
    ├── runs/                    # 历史实验权重与配置
    └── 训练指南.md
```

---

## 2. 快速开始

### 2.1 编译

```bash
cd ~/A_ws            # 你的 colcon 工作区（start.sh 默认 source ~/A_ws/install/setup.bash）
colcon build --packages-select pkg_soccer
source install/setup.bash
```

### 2.2 填话题

`start.sh` 顶部有四个话题变量，**必须按当天动捕的实际刚体名改**：

```bash
TEAM_A_1=/vrpn/xll_dog/pose      # A 队 1 号（本机）
TEAM_A_2=/vrpn/team_a_2/pose     # A 队 2 号
TEAM_B_1=/vrpn/team_b_1/pose     # B 队 1 号
TEAM_B_2=/vrpn/team_b_2/pose     # B 队 2 号
```

仍然是占位值时脚本会直接报错退出（防止拿错话题上场）。

### 2.3 启动

```bash
# 只推理、不发运动指令（第一次上狗/换场地先用这个核对观测）
ENABLE_MOTION=false ./start.sh A1

# 真正比赛
ENABLE_MOTION=true ./start.sh A1
```

四个角色的区别只有"自己是哪只、队友是谁、进攻方向"：

| 角色 | 本狗 | 队友 | 对手 | 进攻方向 |
|---|---|---|---|---|
| `A1` | TEAM_A_1 | TEAM_A_2 | B1 + B2 | `goal_right` |
| `A2` | TEAM_A_2 | TEAM_A_1 | B1 + B2 | `goal_right` |
| `B1` | TEAM_B_1 | TEAM_B_2 | A1 + A2 | `goal_left` |
| `B2` | TEAM_B_2 | TEAM_B_1 | A1 + A2 | `goal_left` |

**两只狗必须用不同角色启动**，队友话题指向对方，否则两只狗会把彼此当成对手。

---

## 3. 比赛循环：进球/出界后自动继续

这是与旧版最大的行为差异。旧版进球或球出界会直接 `stop()` 锁死，整场不再动；
现在按规则 4.2 / 4.4 处理：

| 事件 | 行为 |
|---|---|
| 进球（整球过门线且在门宽内） | 计 1 分 → `WAIT_PLAY` 暂停，等裁判把球放回发球点 |
| 球出边线/端线 | `WAIT_PLAY` 暂停，等裁判就近发球 |
| 球停在门线后但未进球 | `WAIT_PLAY` 暂停，等裁判发球 |
| 球进死区 / 双方僵持 8 s | `WAIT_PLAY` 暂停，等裁判发球 |
| **恢复条件** | 球静止在**任一**发球点（容差 0.40 m）且狗也静止 0.40 s → 自动 `rearm()` 继续 |
| 兜底 | 暂停超过 25 s 且球已静止在场内 → 直接放行（防裁判漏放球） |
| 真正的故障（狗离场、姿态异常、超时） | 仍然 `stop()`，不自动恢复 |

暂停期间不发运动指令；超时计时暂停，恢复后重新计一整轮。

9 个发球点：`x ∈ {-2.5, 0, +2.5}`，`y ∈ {-1.25, 0, +1.25}`（场地坐标）。
**赛前用卷尺核对十字标记实际位置**，偏差大就调 `-p service_point_tolerance:=0.5`。

---

## 4. 最高速度 1.2 m/s

策略权重的动作缩放是 `[0.45, 0.20, 0.70]`（m/s、m/s、rad/s），即原始满舵只有 0.45 m/s。
现在部署端把**前进/侧移**两维按增益放大，转向不放大：

```bash
-p rl_speed_limit:=1.2     # 硬上限（默认 1.2）
-p rl_speed_gain:=0        # 0 = 自动：满舵前进恰好等于 rl_speed_limit
```

自动标定：`gain = 1.2 / 0.45 ≈ 2.667` → 有效缩放 `[1.20, 0.533, 0.70]`，
实测满舵输出 1.200 m/s（对角会被限速裁到 1.2）。

想回退到策略原始速度（0.45 m/s）：

```bash
RL_SPEED_LIMIT=0.45 RL_SPEED_GAIN=1.0 ./start.sh A1
```

也可以直接给绝对值：`RL_SPEED_LIMIT=1.2 RL_SPEED_GAIN=2.0` → 满舵 0.9 m/s。

> ⚠️ 放大动作缩放是**线性外推**，策略训练时只见过 0.45 m/s。速度越高，
> 起步惯性、撞球能量、刹车距离都超出训练分布。建议按 0.6 → 0.9 → 1.2 逐档上机，
> 每档先跑 `ENABLE_MOTION=false` 看指令是否合理，再实跑。

---

## 5. 主要启动参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `control_mode` | `rl`（start.sh） | `rl` / `rules` |
| `rl_policy` | `shared_2v2` | `shared_2v2`（41 维，推荐）/ `shared_2v2_large` / `attacker` / `defender`（29 维）/ `stage2`（23 维） |
| `rl_model_path` | 空 | 指定 npz；空则按 `rl_policy` 在包内找 |
| `rl_speed_limit` | `1.2` | 平移合速度上限（m/s），上限 1.5 受协议限制 |
| `rl_speed_gain` | `0` | 0=按限速自动标定；1.0=策略原速 |
| `rl_dash` | `false` | 高速短冲（1.5 m/s × 450 ms），与训练端同规则 |
| `rl_dash_goal_cone` | `0.9` | 短冲只在朝对方球门 ±52° 内触发 |
| `enable_motion` | `false`（start.sh） | false = 只推理、不发运动指令 |
| `dog_pose_topic` | `/vrpn/xll_dog/pose` | 本狗刚体 |
| `teammate_pose_topic` | 空 | 队友刚体（2v2 必填） |
| `opponent_pose_topic` | 空 | 对手 1 刚体 |
| `opponent2_pose_topic` | 空 | 对手 2 刚体 |
| `attack_goal` | `goal_right` | `goal_right` / `goal_left`，半场换边要改 |
| `max_run_time` | 600 s | 单个半场足够，暂停时间不计入 |
| `service_point_tolerance` | 0.40 m | 判定"球回到发球点"的半径 |
| `resume_settle_time` | 0.40 s | 球静止多久算稳定 |
| `resume_timeout` | 25 s | 暂停兜底放行时间 |

参数都是**只读**启动参数，改要重启节点。

---

## 6. 观测（41 维）

| 索引 | 内容 | 归一化 |
|---|---|---|
| 0-1 | 球相对狗（机身系） | /3 |
| 2-3 | 球速（机身系） | /2 |
| 4-5 | 对方球门相对狗（机身系） | /5 |
| 6-8 | 狗速（机身系）+ 偏航角速度 | 无 |
| 9-11 | 上一步动作（限速后 ÷ 动作缩放） | 无 |
| 12-13 | 狗相对场地朝向 sin/cos | 无 |
| 14-17 | 到四周边界距离 | /9, /9, /6, /6 |
| 18 | 球距 | /3 |
| 19 | 是否曾进入触球圈 | 0/1 |
| 20-22 | 位姿年龄 + 新鲜标志 | /0.5 |
| **23-28** | **队友**：相对位置/相对速度/相对朝向 | /3, /2, sin·cos |
| **29-34** | **对手 1**（同上 6 维） | /3, /2, sin·cos |
| **35-40** | **对手 2**（同上 6 维） | /3, /2, sin·cos |

队友/对手断流时该 6 维填 0（合法朝向 sin/cos 不会同时为 0，策略能识别"看不到"），
**不会**因此进入停车等待；只有本狗和球的位姿断流才会 `WAIT_POSE`。

> ⚠️ **与 2026-11-19 之前的版本不兼容**：旧版是 `23-28 最近对手 / 29-34 队友`。
> `src/pkg_soccer/train/` 那套训练代码仍是旧顺序，直接用它训出的权重配当前
> `rl_control.py` 会喂错特征（不报错、行为异常）。当前部署权重对应
> `train_2v2_large.py`。

---

## 7. 训练

```bash
python train_2v2_large.py --help
```

4 只狗共享一份策略、团队奖励，导出成 `policy_2v2_shared_actor.npz`。
超级参数与导出流程见脚本头部注释。

`src/pkg_soccer/train/` 是早期 2v2 套件（含 `aim.py` 方向感评估、`verify.py`
规则自检等工具），**观测顺序与当前部署相反**，仅在把顺序对齐后复用。

---

## 8. 自检与排查

```bash
# 包内权重能否加载、维度/角色是否匹配
python3 -c "
import numpy as np
w = np.load('src/pkg_soccer/pkg_soccer/policy_2v2_shared_actor.npz')
k = [x for x in w.files if 'weight' in x][0]
print(w[k].shape, w['action_scale'], w['role'])"

# 话题是否在发（动捕正常时每帧都有）
ros2 topic hz /vrpn/xll_dog/pose
ros2 topic echo /vrpn/ball/pose --once
```

| 现象 | 常见原因 |
|---|---|
| 节点起来但狗不动，日志 `WAIT_POSE` | 位姿话题断流、坐标系不一致（`frame_id` 不同）、球/狗刚体名写错 |
| 日志 `等待连续动捕数据` | VRPN 刚起，等 3~5 帧即可；持续出现说明丢包严重 |
| 两只狗互相抢球 | 两个节点用了同一角色，检查 `teammate_pose_topic` |
| 进球后一直停着 | 球没放回发球点（容差 0.40 m）或球还在滚；等 25 s 会兜底放行 |
| 动作很猛/容易撞球出界 | `rl_speed_gain` 放大过多，降到 1.5~2.0 再试 |
| 狗在充电/刚上电不动 | `switch_status=14` 或未站稳，运控会拒绝指令（日志有 3028） |

---

## 9. 与旧版（≤ 2026-11-19）的差异

- 观测槽位顺序改为「队友 23-28 / 对手 29-34 / 对手 2 35-40」，权重不通用。
- 进球、出界、死区不再 `stop()`，改为暂停等裁判发球后自动继续。
- 最高速度通过 `rl_speed_gain` 标定到 `rl_speed_limit`（默认 1.2 m/s）。
- 短冲门控默认 52°（只在朝对方球门时触发）。
- 新增 `start.sh`，按角色选话题，避免手工拼 8 个参数。
