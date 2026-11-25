# 机器狗足球（2v2）

ROS 2 足球比赛主程序：两只狗各跑一份**同一套 41 维强化学习策略**（`shared_2v2`），
靠动捕拿到自己、球、队友和两个对手的位姿，输出机身系速度指令，由运控接口执行。

场地按《机器狗大赛复赛规则》建模：10 × 5.5 m 围挡、门线 x = ±4.585 m、门宽 1.5 m、
9 个发球点（±2.5 / ±1.25 网格）、死区为距围挡 0.8 m 的条带。

---

## 1. 目录结构

**仓库根**（训练/启动辅助，不需要装进包）：

```
DynamicDog/
├── start.sh                    # 比赛启动脚本（按 A1/A2/B1/B2 选角色）
├── train_2v2_large.py          # 4 狗共享策略的 PPO 训练器（PyTorch，独立仿真）
├── policy_2v2_shared.pt        # 训练产物（PyTorch 权重，只在训练侧用）
└── src/pkg_soccer/             # ← 这个目录才要放到狗的工作区里
```

**包内部**（狗上必须是这个结构，`ament_python` 的固定约定）：

```
src/pkg_soccer/                      ← 包根，名字随便，但里面要有下面四项
├── pkg_soccer/                      ← Python 模块目录（有 __init__.py）
│   ├── __init__.py
│   ├── soccer.py                    # 主节点：话题、姿态门控、比赛循环
│   ├── rl_control.py                # 策略加载、41 维观测、速度/短冲、暂停恢复
│   ├── control.py                   # 状态机基类：场地几何、发球点、暂停/复位
│   ├── attack.py                    # 进攻规划（rl_policy:=attacker 时才 import）
│   ├── stand_guard.py               # 姿态门控（坐/趴/站、故障检测）
│   ├── measurement.py               # 位姿轨迹与速度估计
│   ├── record_motion.py             # 位姿录制小工具
│   ├── policy_2v2_shared_actor.npz  # ★ 部署用策略（41 维，0.45 档）
│   ├── policy_duel_*_actor.npz      # 旧 1v1 策略（29 维）
│   ├── policy_stage2_actor.npz      # 旧单狗策略（23 维）
│   ├── policy_team_attacker_actor.npz  # 我们训的旧 41 维（槽位顺序不同，见 §6）
│   └── leg_kick.py / gait_kick*.py / kick_*.py   # 踢腿标定独立脚本（主程序已不 import）
├── resource/
│   └── pkg_soccer                   # ★ 空文件，ament_index 标记，缺了 ros2 run 找不到包
├── package.xml
├── setup.cfg
├── setup.py                         # package_data 里必须列上要安装的 .npz
├── train/                           # 训练套件（不安装，留作记录）
├── runs/                            # 历史实验权重（不安装）
└── 训练指南.md
```

> ⚠️ 本地 `机器狗足球/pkg_soccer/` 是**摊平的编辑视图**（`setup.py` 和 `soccer.py` 在同一层），
> 直接 `scp` 到狗上会缺 `pkg_soccer/` 这一层模块目录，`colcon build` 会报
> `Multiple top-level packages` 或 `No module named pkg_soccer.soccer`。
> **部署请用仓库里的 `src/pkg_soccer/`（即 `_dd/src/pkg_soccer/`）。**

---

## 2. 上狗部署

### 2.1 放到狗上

狗上的工作区（SSH 进去 `ls` 能看到 `build/ install/ log/ src/` 的那个目录，
从截图看是 `~/lizj`）：

> 路径**区分大小写**，现场是小写 `~/lizj`。写成 `~/LIZJ/...` 会报
> `cp: cannot create regular file '/home/mi/LIZJ/start.sh': No such file or directory`。
> `start.sh` 会自动按"脚本所在目录 → 当前目录 → ~/lizj → ~/LIZJ → ~/A_ws"找
> `install/setup.bash`，所以只要脚本放对了，从哪个目录启动都能 source 到。

```
~/lizj/
├── build/  install/  log/
└── src/
    ├── myinterface_pkg/      # 别的包，不要动
    └── pkg_soccer/           # ← 换成新的
```

**方式 A：狗上直接 clone（能上外网时最省事）**

```bash
cd ~/lizj
mv src/pkg_soccer src/pkg_soccer.old.$(date +%m%d)    # 先备份旧包
git clone --depth 1 https://github.com/AmdirLeo/DynamicDog.git /tmp/dd
cp -r /tmp/dd/src/pkg_soccer src/
cp /tmp/dd/start.sh ~/lizj/
```

**方式 B：从 Windows 用 scp 传**

```bash
# 在 Windows 的 Git Bash 里执行
cd "E:/机器狗程序调试区"
scp -r _dd/src/pkg_soccer mi@<狗IP>:~/lizj/src/
scp _dd/start.sh mi@<狗IP>:~/lizj/
```

只传必要文件（省时间，不传 runs/train）：

```bash
scp -r _dd/src/pkg_soccer/pkg_soccer _dd/src/pkg_soccer/resource \
       _dd/src/pkg_soccer/package.xml _dd/src/pkg_soccer/setup.py \
       _dd/src/pkg_soccer/setup.cfg mi@<狗IP>:~/lizj/src/pkg_soccer/
```

> 老版本里没有 `attack.py` 和 `policy_2v2_shared_actor.npz`，这两个必须带上，
> 否则 `rl_policy:=shared_2v2` 启动时会 `FileNotFoundError`。

### 2.2 编译

```bash
cd ~/lizj
source /opt/ros2/galactic/setup.bash     # 本机 ROS 装在 /opt/ros2/galactic（不是 /opt/ros）
colcon build --packages-select pkg_soccer        # 首次编译约 10 s
source install/setup.bash
```

> 本机 Python 是 **3.6**（`/opt/ros2/galactic/lib/python3.6/...`）。代码里不能用
> 海象运算符 `:=`、内建泛型注解 `list[...]` 等 3.8+ 语法，否则节点在 import 阶段
> 直接 `SyntaxError`。`rl_control.py` 已按此改过。

编译成功的标志是这两条都有输出：

```bash
ros2 pkg prefix pkg_soccer                      # → /home/mi/lizj/install/pkg_soccer
ros2 pkg executables pkg_soccer                 # → pkg_soccer soccer
```

### 2.3 改 `start.sh`（三处，缺一不可）

```bash
DOG_NAME=mi_desktop_48_b0_2d_5f_bd_98   # ★ 本机运控命名空间，查：ros2 topic list | grep motion_status
TEAM_A_1=/vrpn/xll_dog/pose             # ★ 队友的狗（setup 里第一个）
TEAM_A_2=/vrpn/Lizj_dog/pose            # ★ 我的机器狗（setup 里第二个），查：ros2 topic list | grep vrpn
TEAM_B_1=/vrpn/team_b_1/pose
TEAM_B_2=/vrpn/team_b_2/pose
```

- `DOG_NAME` 是**运控接口**的命名空间（`/<DOG_NAME>/motion_servo_cmd`），
  和动捕刚体名（`/vrpn/Lizj_dog/pose`）是**两件事**。默认值
  `mi_desktop_48_b0_2d_7b_03_fd` 只是占位，填错的话指令发到不存在的 topic，
  **狗一步不动且没有任何报错**。
- `start.sh` 的话题检查**按策略判定**，只查当前策略真正用到的通道（缺哪条会逐条列出来）：

  | `RL_POLICY` | 维度 | 必须有的通道 |
  |---|---|---|
  | `stage2` | 23 | 本狗 + 球 |
  | `attacker` / `defender` | 29 | 本狗 + 球 + 1 个对手（1v1） |
  | `shared_2v2` / `shared_2v2_large` | 41 | 本狗 + 球 + 队友 + 2 个对手（2v2） |

  所以现场只有两只狗时，用 `RL_POLICY=attacker ./start.sh A2` 就能跑，
  不必为了凑队友去填假话题。台架只接本狗时加 `SKIP_TOPIC_CHECK=1`。
- 两个**只告警不拦截**的结构性问题：队友/对手话题与本狗话题相同（策略会把自己当成别人）、
  `shared_2v2` 下两个对手话题相同（次近对手槽位重复同一只狗）。
- 也可以用环境变量临时覆盖，不改文件：
  `DOG_NAME=xxx TEAM_A_2=/vrpn/Lizj_dog/pose ./start.sh A2`
- 写法必须是 `${VAR:-默认值}`（冒号 + 减号）。写成 `${VAR:默认值}` 会被 bash 当成
  **子串截取**，配合脚本开头的 `set -u` 报 `VAR: unbound variable`，看着像未定义，
  其实是少了那个 `-`。

> ⚠️ `shared_2v2`（41 维）要求 **dog / ball / teammate / opponent / opponent2 五路都有数据**，
> 少一路就永久停在 `WAIT_POSE`（日志会打印缺哪一路）。只有两只狗时用
> `RL_POLICY=attacker`（29 维）；只有本狗 + 球时用
> `RL_POLICY=stage2 ENABLE_MOTION=false ./start.sh A2` 先把链路跑通。

### 2.4 启动顺序

```bash
# ① 先起动捕（另开一个终端，或在别的机器上）
ros2 launch vrpn_listener sync_entity_state.launch
ros2 topic list | grep vrpn                     # 确认 4 只狗 + 球的 pose 都在

# ② 空跑：只推理、不发运动指令（换场地/换话题必做）
cd ~/lizj && ENABLE_MOTION=false ./start.sh A2

# ③ 真跑
ENABLE_MOTION=true ./start.sh A2
```

`ENABLE_MOTION=false` 时节点照常算策略、打日志（`|v|`、状态、观测是否齐全），
只是不往 `motion_servo_cmd` 发东西。第一次到新场地，先这样核对：

```bash
ros2 topic echo /vrpn/Lizj_dog/pose --once      # 狗刚体有数据
ros2 topic echo /vrpn/ball/pose --once          # 球刚体有数据
```

日志里出现 `WAIT_POSE ... 等待同坐标系连续有效位姿` 就是上面某个话题没通。

### 2.5 四个角色

| 角色 | 本狗 | 队友 | 对手 | 进攻方向 | 谁用 |
|---|---|---|---|---|---|
| `A1` | TEAM_A_1 | TEAM_A_2 | B1 + B2 | `goal_right` | **队友**（setup 里第一个） |
| `A2` | TEAM_A_2 | TEAM_A_1 | B1 + B2 | `goal_right` | **我**（setup 里第二个，默认角色） |
| `B1` | TEAM_B_1 | TEAM_B_2 | A1 + A2 | `goal_left` | 对手 |
| `B2` | TEAM_B_2 | TEAM_B_1 | A1 + A2 | `goal_left` | 对手 |

约定：两人共用这份 setup 时，队友永远是 A1，我永远是 A2；我的动捕话题是 `TEAM_A_2`
（`/vrpn/Lizj_dog/pose`），队友的是 `TEAM_A_1`（`/vrpn/xll_dog/pose`）。
`./start.sh` 不带参数时默认就是 A2。两只狗必须用不同角色、队友话题互指；
两台机器都跑 A1 会把彼此当对手。

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
RL_SPEED_LIMIT=0.45 RL_SPEED_GAIN=1.0 ./start.sh A2
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
| `dog_pose_topic` | `/vrpn/Lizj_dog/pose` | 本狗刚体（A2；队友的 `/vrpn/xll_dog/pose` 只作 teammate） |
| `dog_center_offset_x` | `0` m | 刚体原点 → 实际质心的偏移（刚体系 +x 朝狗头），用原始动捕朝向旋转 |
| `dog_center_offset_y` | `0` m | 同上，刚体系 +y 朝左；只作用于本狗，不改队友/对手 |
| `dog_radius` | `0` m | 狗的等效碰撞半径；0=关闭。非 0 时两只狗中心距 < 2×半径+0.10 m 且正在靠近 → 去掉径向速度、保留绕行和转向 |
| `defense_enable` | `true` | 危险区固定防守总开关（仅 1v1/2v2）；球在我方半场近本方门且对手持球时覆盖网络权重冲对手，详见 `代码使用说明.md` §10.1 |
| `defense_fraction` | `0.333` | 危险距离 = 该比例×场长（默认 1/3） |
| `defense_hysteresis` | `0.40` | 滞回半带宽（m），防边缘反复切换 |
| `defense_carrier_dist` | `0.60` | 判"对手持球"的球-对手距离上限（m） |
| `defense_contact_dist` | `0.50` | 判"已贴身"的狗-对手距离，到达后停平移（m） |
| `teammate_pose_topic` | 空 | 队友刚体（2v2 必填） |
| `opponent_pose_topic` | 空 | 对手 1 刚体 |
| `opponent2_pose_topic` | 空 | 对手 2 刚体 |
| `attack_goal` | `goal_right` | `goal_right` / `goal_left`，半场换边要改 |
| `max_run_time` | 600 s | 单个半场足够，暂停时间不计入 |
| `service_point_tolerance` | 0.40 m | 判定"球回到发球点"的半径 |
| `resume_settle_time` | 0.40 s | 球静止多久算稳定 |
| `resume_timeout` | 25 s | 暂停兜底放行时间 |

参数都是**只读**启动参数，改要重启节点。

`start.sh` 中对应的环境变量：`DOG_CENTER_OFFSET_X` / `DOG_CENTER_OFFSET_Y` / `DOG_RADIUS`，
以及防守的 `DEFENSE_ENABLE` / `DEFENSE_FRACTION` / `DEFENSE_HYSTERESIS` / `DEFENSE_CARRIER_DIST` / `DEFENSE_CONTACT_DIST`。
标定质心偏移的办法：让狗原地朝 +x 站好，量动捕刚体原点到实际机身中心的水平距离；
向狗头偏为正 x，向狗左偏为正 y。装反了会看到狗绕球方向整体偏一个固定角度/距离。
`dog_radius` 取机身后半部到尾部的最大半径，CyberDog 一般 0.25~0.30 m；
避让只在 `control_mode:=rl` 且 2v2 队友话题有效时生效。

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
ros2 topic hz /vrpn/Lizj_dog/pose
ros2 topic echo /vrpn/ball/pose --once
```

| 现象 | 常见原因 |
|---|---|
| `RCLError: Couldn't parse parameter override rule: '-p rl_model_path:='` | 空值参数被 rcl 拒绝。`start.sh` 已改成"空值整条省略"；手写命令时别写 `-p rl_model_path:=`（留空就别传这个参数） |
| `install/setup.bash: line N: COLCON_TRACE: unbound variable` | 脚本在 `set -u` 下 source colcon 环境。用 4bb1b55 之后的 `start.sh`（source 前后 `set +u` / `set -u`）；手改的话就是把 `source` 那行包在 `set +u` 和 `set -u` 之间 |
| 节点起来但狗不动，日志 `WAIT_POSE` | 位姿话题断流、坐标系不一致（`frame_id` 不同）、球/狗刚体名写错；`shared_2v2` 还会要求队友和两个对手都有数据 |
| 日志 `等待连续动捕数据` | VRPN 刚起，等 3~5 帧即可；持续出现说明丢包严重 |
| 两只狗互相抢球 | 两个节点用了同一角色，检查 `teammate_pose_topic` |
| 进球/出界后不动 | 等裁判把球**移动过且静止 0.5 s** 才自动继续；球还在滚就继续等。`RL_CONTINUE_DEADZONE=true`（默认）时死区里会以 0.35 m/s 低速走位，不会完全站住 |
| 动作很猛/容易撞球出界 | `rl_speed_gain` 放大过多，降到 1.5~2.0 再试 |
| 狗在充电/刚上电不动 | `switch_status=14` 或未站稳，运控会拒绝指令（日志有 3028） |

---

## 9. 与旧版（≤ 2026-11-19）的差异

- 观测槽位顺序改为「队友 23-28 / 对手 29-34 / 对手 2 35-40」，权重不通用。
- 进球、出界、死区不再 `stop()`，改为暂停等裁判发球后自动继续。
- 最高速度通过 `rl_speed_gain` 标定到 `rl_speed_limit`（默认 1.2 m/s）。
- 短冲门控默认 52°（只在朝对方球门时触发）。
- 新增 `start.sh`，按角色选话题，避免手工拼 8 个参数。
