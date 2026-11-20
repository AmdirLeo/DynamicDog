#!/usr/bin/env bash
set -euo pipefail

# 2v2 shared-policy launcher. Edit the four topic variables once for the
# current field, then run: ./start.sh A1 (or A2/B1/B2).
ROLE="${1:-A1}"
ENABLE_MOTION="${ENABLE_MOTION:-false}"
RL_POLICY="${RL_POLICY:-shared_2v2}"
RL_MODEL_PATH="${RL_MODEL_PATH:-}"

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

if [[ "$TEAM_A_2" == "/vrpn/team_a_2/pose" ||
      "$TEAM_B_1" == "/vrpn/team_b_1/pose" ||
      "$TEAM_B_2" == "/vrpn/team_b_2/pose" ]]; then
  echo "请先在 start.sh 顶部填写 TEAM_A_2、TEAM_B_1、TEAM_B_2 的实际 VRPN 话题。" >&2
  exit 2
fi

if [[ -f "${A_WS:-$HOME/A_ws}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${A_WS:-$HOME/A_ws}/install/setup.bash"
fi

exec ros2 run pkg_soccer soccer --ros-args \
  -p control_mode:=rl \
  -p rl_policy:="$RL_POLICY" \
  -p rl_model_path:="$RL_MODEL_PATH" \
  -p dog_pose_topic:="$DOG_TOPIC" \
  -p teammate_pose_topic:="$TEAMMATE_TOPIC" \
  -p opponent_pose_topic:="$OPPONENT_TOPIC" \
  -p opponent2_pose_topic:="$OPPONENT2_TOPIC" \
  -p attack_goal:="$GOAL" \
  -p enable_motion:="$ENABLE_MOTION"
