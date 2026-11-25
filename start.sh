#!/usr/bin/env bash
set -euo pipefail

# 2v2 shared-policy launcher.
#   1) 首次使用：改 DOG_NAME 和四个 VRPN 话题（或用同名环境变量覆盖）
#   2) 启动：./start.sh A1|A2|B1|B2
# 角色约定（两人共用一份 setup）：队友始终是 A1（第一个），我始终是 A2（第二个）。
# 因此本机默认角色是 A2，本狗话题走 TEAM_A_2；不要再用 A1 跑本机。
ROLE="${1:-A2}"
ENABLE_MOTION="${ENABLE_MOTION:-false}"
RL_POLICY="${RL_POLICY:-shared_2v2}"
RL_MODEL_PATH="${RL_MODEL_PATH:-}"
# 最高速度（m/s）：策略是 0.45 档，默认按此值把动作放大（0.45 × 2.667 = 1.2）。
RL_SPEED_LIMIT="${RL_SPEED_LIMIT:-1.2}"
# 0 = 按 RL_SPEED_LIMIT 自动标定；想复现策略原始 0.45 速度就设 1.0
RL_SPEED_GAIN="${RL_SPEED_GAIN:-0}"
# 球进死区时仍继续低速走位；设为 false 可恢复等待裁判时停住。
RL_CONTINUE_DEADZONE="${RL_CONTINUE_DEADZONE:-true}"
RL_DEADZONE_SPEED_LIMIT="${RL_DEADZONE_SPEED_LIMIT:-0.35}"

# 动捕刚体原点 → 狗实际质心的偏移（米，刚体系：+x 朝狗头、+y 朝左）。
# 例：刚体装在屁股上、质心在前方 8 cm、偏左 2 cm → 0.08 / 0.02。
DOG_CENTER_OFFSET_X="${DOG_CENTER_OFFSET_X:-0}"
DOG_CENTER_OFFSET_Y="${DOG_CENTER_OFFSET_Y:-0}"
# 狗的等效碰撞半径（米）；两只狗中心距 < 2×半径+0.10 m 时不再朝队友加速。
# 0 = 关闭避让；CyberDog 建议 0.25~0.30。
DOG_RADIUS="${DOG_RADIUS:-0}"

# 运动接口命名空间：/<DOG_NAME>/motion_servo_cmd 的前缀。
# 留空 = 用节点默认值 mi_desktop_48_b0_2d_7b_03_fd，很可能不是本机，务必核对：
#   ros2 topic list | grep motion_status
DOG_NAME="${DOG_NAME:-}"

# 话题默认值：DEF_A2 是本机已确认的真实刚体，其余三条仍是"待填占位"。
# 占位判定只针对尚未确认的话题（见 is_placeholder），不要把所有默认值都当占位。
DEF_A1="/vrpn/xll_dog/pose"            # 队友（setup 里的第一个），当天按真实刚体名填
DEF_A2="/vrpn/Lizj_dog/pose"           # 我的机器狗（setup 里的第二个），已确认
DEF_B1="/vrpn/team_b_1/pose"           # 对手，占位
DEF_B2="/vrpn/team_b_2/pose"           # 对手，占位

TEAM_A_1="${TEAM_A_1:-$DEF_A1}"
TEAM_A_2="${TEAM_A_2:-$DEF_A2}"
TEAM_B_1="${TEAM_B_1:-$DEF_B1}"
TEAM_B_2="${TEAM_B_2:-$DEF_B2}"

# ros2 按字面量推断参数类型："0" 是 INTEGER，"0.0" 才是 DOUBLE。
# 节点把这些参数声明为 double，传整数形式会报
#   InvalidParameterTypeException: Trying to set parameter 'rl_speed_gain'
#   to '0' of type 'INTEGER', expecting type 'DOUBLE'
# 所以统一格式化成带小数点的写法；布尔同理（传 1/0 会变成 INTEGER）。
as_double() {
  local v="$1"
  if [[ ! "$v" =~ ^-?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?$ ]]; then
    echo "参数必须是数字：$v" >&2
    exit 2
  fi
  printf '%.3f' "$v"
}
as_bool() {
  case "$1" in
    true|True|TRUE|1|yes|on) echo true ;;
    false|False|FALSE|0|no|off) echo false ;;
    *) echo "布尔参数只能是 true/false：$1" >&2; exit 2 ;;
  esac
}
RL_SPEED_LIMIT="$(as_double "$RL_SPEED_LIMIT")"
RL_SPEED_GAIN="$(as_double "$RL_SPEED_GAIN")"
RL_DEADZONE_SPEED_LIMIT="$(as_double "$RL_DEADZONE_SPEED_LIMIT")"
DOG_CENTER_OFFSET_X="$(as_double "$DOG_CENTER_OFFSET_X")"
DOG_CENTER_OFFSET_Y="$(as_double "$DOG_CENTER_OFFSET_Y")"
DOG_RADIUS="$(as_double "$DOG_RADIUS")"
RL_CONTINUE_DEADZONE="$(as_bool "$RL_CONTINUE_DEADZONE")"
ENABLE_MOTION="$(as_bool "$ENABLE_MOTION")"

case "$ROLE" in
  A1) DOG_TOPIC="$TEAM_A_1"; TEAMMATE_TOPIC="$TEAM_A_2"; OPPONENT_TOPIC="$TEAM_B_1"; OPPONENT2_TOPIC="$TEAM_B_2"; GOAL=goal_right ;;
  A2) DOG_TOPIC="$TEAM_A_2"; TEAMMATE_TOPIC="$TEAM_A_1"; OPPONENT_TOPIC="$TEAM_B_1"; OPPONENT2_TOPIC="$TEAM_B_2"; GOAL=goal_right ;;
  B1) DOG_TOPIC="$TEAM_B_1"; TEAMMATE_TOPIC="$TEAM_B_2"; OPPONENT_TOPIC="$TEAM_A_1"; OPPONENT2_TOPIC="$TEAM_A_2"; GOAL=goal_left ;;
  B2) DOG_TOPIC="$TEAM_B_2"; TEAMMATE_TOPIC="$TEAM_B_1"; OPPONENT_TOPIC="$TEAM_A_1"; OPPONENT2_TOPIC="$TEAM_A_2"; GOAL=goal_left ;;
  *) echo "用法：$0 A1|A2|B1|B2" >&2; exit 2 ;;
esac

# 占位话题判定：DEF_A2 是本机确认过的真实刚体，不算占位；
# 其余三条是待填占位，裸启动时应拦下提醒用户按当天动捕名覆盖。
is_placeholder() {
  case "$1" in
    "$DEF_A1"|"$DEF_B1"|"$DEF_B2") return 0 ;;
    *) return 1 ;;
  esac
}

# 只检查当前策略真正会用到的通道：
#   stage2            只要本狗 + 球（23 维）
#   attacker/defender 本狗 + 球 + 1 个对手（29 维，1v1）
#   shared_2v2*       本狗 + 球 + 队友 + 2 个对手（41 维，2v2）
missing=()
is_placeholder "$DOG_TOPIC" && missing+=("本狗 dog_pose_topic=$DOG_TOPIC")
if [[ "$RL_POLICY" != "stage2" ]]; then
  is_placeholder "$OPPONENT_TOPIC" && missing+=("对手 opponent_pose_topic=$OPPONENT_TOPIC")
fi
case "$RL_POLICY" in
  shared_2v2|shared_2v2_large)
    is_placeholder "$TEAMMATE_TOPIC" && missing+=("队友 teammate_pose_topic=$TEAMMATE_TOPIC")
    is_placeholder "$OPPONENT2_TOPIC" && missing+=("次近对手 opponent2_pose_topic=$OPPONENT2_TOPIC")
    ;;
esac

if [[ "${SKIP_TOPIC_CHECK:-0}" != "1" && ${#missing[@]} -gt 0 ]]; then
  echo "以下话题还是占位值（当前 rl_policy=$RL_POLICY，角色 $ROLE）：" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  echo "查法：ros2 topic list | grep vrpn" >&2
  echo "台架只测本狗时可加 SKIP_TOPIC_CHECK=1 跳过本检查。" >&2
  exit 2
fi
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "警告：SKIP_TOPIC_CHECK=1，忽略占位话题：" >&2
  printf '  - %s\n' "${missing[@]}" >&2
fi

# 话题重复：同一刚体喂给两个通道，策略会看到"假队友/假对手"
for pair in "TEAMMATE_TOPIC:$TEAMMATE_TOPIC" "OPPONENT_TOPIC:$OPPONENT_TOPIC" \
            "OPPONENT2_TOPIC:$OPPONENT2_TOPIC"; do
  if [[ "${pair#*:}" == "$DOG_TOPIC" ]]; then
    echo "警告：${pair%%:*} 与 dog_pose_topic 相同（$DOG_TOPIC），策略会把自己当成别人。" >&2
  fi
done
if [[ "$RL_POLICY" == shared_2v2* && "$OPPONENT_TOPIC" == "$OPPONENT2_TOPIC" ]]; then
  echo "警告：两个对手话题相同（$OPPONENT_TOPIC），次近对手槽位会重复同一只狗。" >&2
fi



# 工作区：优先 A_WS，其次脚本所在目录、当前目录，再退到 ~/lizj、~/LIZJ、~/A_ws
# （Linux 区分大小写，现场工作区是 ~/lizj，不要写成 ~/LIZJ）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${A_WS:-}"
if [[ -z "$WS" ]]; then
  for cand in "$SCRIPT_DIR" "$PWD" "$HOME/lizj" "$HOME/LIZJ" "$HOME/A_ws"; do
    if [[ -f "$cand/install/setup.bash" ]]; then WS="$cand"; break; fi
  done
fi
if [[ -n "$WS" && -f "$WS/install/setup.bash" ]]; then
  # colcon 生成的 setup.bash 会引用可能未定义的变量（如 COLCON_TRACE），
  # 与本脚本的 set -u 冲突，source 期间必须临时关掉 nounset。
  set +u
  # shellcheck disable=SC1090
  source "$WS/install/setup.bash"
  set -u
  echo "已 source $WS/install/setup.bash"
else
  echo "提示：没找到 install/setup.bash，请先 colcon build 并 source，或设 A_WS=<工作区>" >&2
fi

# 开运动前核对 dog_name 是否真有 motion_status（错了狗不会动，且没有报错）
if [[ "$ENABLE_MOTION" == "true" && -n "$DOG_NAME" ]]; then
  if ! ros2 topic list 2>/dev/null | grep -qx "/${DOG_NAME}/motion_status"; then
    echo "警告：/${DOG_NAME}/motion_status 不存在，DOG_NAME 可能填错。" >&2
  fi
fi

ARGS=(
  -p control_mode:=rl
  -p rl_policy:="$RL_POLICY"
  -p dog_pose_topic:="$DOG_TOPIC"
  -p attack_goal:="$GOAL"
  -p rl_speed_limit:="$RL_SPEED_LIMIT"
  -p rl_speed_gain:="$RL_SPEED_GAIN"
  -p rl_continue_deadzone:="$RL_CONTINUE_DEADZONE"
  -p rl_deadzone_speed_limit:="$RL_DEADZONE_SPEED_LIMIT"
  -p enable_motion:="$ENABLE_MOTION"
)

# 可选参数：rcl 不接受 `-p name:=` 这种空值（会报
# "Couldn't parse parameter override rule" 并直接退出），空值必须整条省略。
add_opt() {
  if [[ -n "$2" ]]; then ARGS+=(-p "$1:=$2"); fi
  return 0
}
if [[ -n "$RL_MODEL_PATH" && ! -f "$RL_MODEL_PATH" ]]; then
  echo "警告：RL_MODEL_PATH=$RL_MODEL_PATH 不存在，改用包内默认权重。" >&2
else
  add_opt rl_model_path "$RL_MODEL_PATH"
fi
add_opt dog_name "$DOG_NAME"
add_opt dog_center_offset_x "$DOG_CENTER_OFFSET_X"
add_opt dog_center_offset_y "$DOG_CENTER_OFFSET_Y"
add_opt dog_radius "$DOG_RADIUS"
add_opt teammate_pose_topic "$TEAMMATE_TOPIC"
add_opt opponent_pose_topic "$OPPONENT_TOPIC"
add_opt opponent2_pose_topic "$OPPONENT2_TOPIC"

exec ros2 run pkg_soccer soccer --ros-args "${ARGS[@]}"
