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
# 球进死区时仍继续低速走位；设为 false 可恢复等待裁判时停住。
RL_CONTINUE_DEADZONE="${RL_CONTINUE_DEADZONE:-true}"
RL_DEADZONE_SPEED_LIMIT="${RL_DEADZONE_SPEED_LIMIT:-0.35}"

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

# 占位话题判定（只有这几条默认值算"没填"）
is_placeholder() {
  case "$1" in
    /vrpn/xll_dog/pose|/vrpn/team_a_2/pose|/vrpn/team_b_1/pose|/vrpn/team_b_2/pose) return 0 ;;
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
  -p rl_continue_deadzone:="$RL_CONTINUE_DEADZONE"
  -p rl_deadzone_speed_limit:="$RL_DEADZONE_SPEED_LIMIT"
  -p enable_motion:="$ENABLE_MOTION"
)
[[ -n "$DOG_NAME" ]] && ARGS+=(-p dog_name:="$DOG_NAME")

exec ros2 run pkg_soccer soccer --ros-args "${ARGS[@]}"
