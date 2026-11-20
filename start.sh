#!/usr/bin/env bash
set -euo pipefail

# 2v2 shared-policy launcher.
#   1) 首次使用：改 DOG_NAME 和四个 VRPN 话题（或用同名环境变量覆盖）
#   2) 启动：./start.sh A1|A2|B1|B2
ROLE="${1:-A1}"
ENABLE_MOTION="${ENABLE_MOTION:-false}"
RL_POLICY="${RL_POLICY:-shared_2v2}"
RL_MODEL_PATH="${RL_MODEL_PATH:-}"
# 最高速度（m/s）：策略是 0.45 档，默认按此值把动作放大（0.45 × 2.667 = 1.2）。
RL_SPEED_LIMIT="${RL_SPEED_LIMIT:-1.2}"
# 0 = 按 RL_SPEED_LIMIT 自动标定；想复现策略原始 0.45 速度就设 1.0
RL_SPEED_GAIN="${RL_SPEED_GAIN:-0}"

# 运动接口命名空间：/<DOG_NAME>/motion_servo_cmd 的前缀。
# 留空 = 用节点默认值 mi_desktop_48_b0_2d_7b_03_fd，很可能不是本机，务必核对：
#   ros2 topic list | grep motion_status
DOG_NAME="${DOG_NAME:-}"

TEAM_A_1="${TEAM_A_1:-/vrpn/xll_dog/pose}"
TEAM_A_2="${TEAM_A_2:-/vrpn/team_a_2/pose}"
TEAM_B_1="${TEAM_B_1:-/vrpn/team_b_1/pose}"
TEAM_B_2="${TEAM_B_2:-/vrpn/team_b_2/pose}"

case "$ROLE" in
  A1) DOG_TOPIC="$TEAM_A_1"; TEAMMATE_TOPIC="$TEAM_A_2"; OPPONENT_TOPIC="$TEAM_B_1"; OPPONENT2_TOPIC="$TEAM_B_2"; GOAL=goal_right ;;
  A2) DOG_TOPIC="$TEAM_A_2"; TEAMMATE_TOPIC="$TEAM_A_1"; OPPONENT_TOPIC="$TEAM_B_1"; OPPONENT2_TOPIC="$TEAM_B_2"; GOAL=goal_right ;;
  B1) DOG_TOPIC="$TEAM_B_1"; TEAMMATE_TOPIC="$TEAM_B_2"; OPPONENT_TOPIC="$TEAM_A_1"; OPPONENT2_TOPIC="$TEAM_A_2"; GOAL=goal_left ;;
  B2) DOG_TOPIC="$TEAM_B_2"; TEAMMATE_TOPIC="$TEAM_B_1"; OPPONENT_TOPIC="$TEAM_A_1"; OPPONENT2_TOPIC="$TEAM_A_2"; GOAL=goal_left ;;
  *) echo "用法：$0 A1|A2|B1|B2" >&2; exit 2 ;;
esac

if [[ "$TEAM_A_1" == "/vrpn/xll_dog/pose" ||
      "$TEAM_A_2" == "/vrpn/team_a_2/pose" ||
      "$TEAM_B_1" == "/vrpn/team_b_1/pose" ||
      "$TEAM_B_2" == "/vrpn/team_b_2/pose" ]]; then
  echo "请先在 start.sh 顶部填写 TEAM_A_1/A_2、TEAM_B_1/B_2 的实际 VRPN 话题。" >&2
  echo "查法：ros2 topic list | grep vrpn" >&2
  exit 2
fi

# 工作区：优先 A_WS，其次当前目录，再退到 ~/LIZJ、~/A_ws
WS="${A_WS:-}"
if [[ -z "$WS" ]]; then
  for cand in "$PWD" "$HOME/LIZJ" "$HOME/A_ws"; do
    if [[ -f "$cand/install/setup.bash" ]]; then WS="$cand"; break; fi
  done
fi
if [[ -n "$WS" && -f "$WS/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "$WS/install/setup.bash"
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
  -p rl_model_path:="$RL_MODEL_PATH"
  -p dog_pose_topic:="$DOG_TOPIC"
  -p teammate_pose_topic:="$TEAMMATE_TOPIC"
  -p opponent_pose_topic:="$OPPONENT_TOPIC"
  -p opponent2_pose_topic:="$OPPONENT2_TOPIC"
  -p attack_goal:="$GOAL"
  -p rl_speed_limit:="$RL_SPEED_LIMIT"
  -p rl_speed_gain:="$RL_SPEED_GAIN"
  -p enable_motion:="$ENABLE_MOTION"
)
[[ -n "$DOG_NAME" ]] && ARGS+=(-p dog_name:="$DOG_NAME")

exec ros2 run pkg_soccer soccer --ros-args "${ARGS[@]}"
