#!/usr/bin/env bash
set -euo pipefail

# 2v2 shared-policy launcher.
#   1) 首次使用：改 DOG_NAME 和四个 VRPN 话题（或用同名环境变量覆盖）
#   2) 启动：./start.sh A1|A2|B1|B2
#      默认 A1/B1=attack、A2/B2=assist；可用 TEAM_ROLE 覆盖。
ROLE="${1:-A1}"
ENABLE_MOTION="${ENABLE_MOTION:-false}"
RL_POLICY="${RL_POLICY:-shared_2v2}"
TEAM_ROLE="${TEAM_ROLE:-}"
RL_AUTO_ROLE_SWITCH="${RL_AUTO_ROLE_SWITCH:-true}"
RL_MODEL_PATH="${RL_MODEL_PATH:-}"
RL_ATTACK_MODEL_PATH="${RL_ATTACK_MODEL_PATH:-}"
RL_ASSIST_MODEL_PATH="${RL_ASSIST_MODEL_PATH:-}"
# 若实测胸部相对刚体中心有偏差，可按每只狗单独覆盖（单位米）；留空使用权重中的标定值。
CONTACT_OFFSET_X="${CONTACT_OFFSET_X:-}"
CONTACT_OFFSET_Y="${CONTACT_OFFSET_Y:-}"
# 当前实机保守速度（m/s）：先用 0.45，避免高速动捕偏移。
RL_SPEED_LIMIT="${RL_SPEED_LIMIT:-0.45}"
# 0 = 按 RL_SPEED_LIMIT 自动标定；设 1.0 可复现权重原始动作档。
RL_SPEED_GAIN="${RL_SPEED_GAIN:-0.0}"
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

# 默认每队一攻一支援；需要更换初始主攻狗时显式设 TEAM_ROLE=attack/assist。
if [[ -z "$TEAM_ROLE" ]]; then
  case "$ROLE" in
    A1|B1) TEAM_ROLE="attack" ;;
    A2|B2) TEAM_ROLE="assist" ;;
  esac
fi

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
  -p team_role:="$TEAM_ROLE"
  -p rl_auto_role_switch:="$RL_AUTO_ROLE_SWITCH"
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
add_opt rl_attack_model_path "$RL_ATTACK_MODEL_PATH"
add_opt rl_assist_model_path "$RL_ASSIST_MODEL_PATH"
add_opt ball_contact_offset_x "$CONTACT_OFFSET_X"
add_opt ball_contact_offset_y "$CONTACT_OFFSET_Y"
add_opt dog_name "$DOG_NAME"
add_opt teammate_pose_topic "$TEAMMATE_TOPIC"
add_opt opponent_pose_topic "$OPPONENT_TOPIC"
add_opt opponent2_pose_topic "$OPPONENT2_TOPIC"

exec ros2 run pkg_soccer soccer --ros-args "${ARGS[@]}"
