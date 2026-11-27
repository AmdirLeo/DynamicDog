#!/usr/bin/env bash
# 已筛选的 T11 双网络实测配置；对手由每场比赛的命令行指定。
set -eo pipefail

usage() {
  echo "用法：$0 A1|A2 /vrpn/对手1/pose /vrpn/对手2/pose [goal_right|goal_left]" >&2
  echo "A1=xll_dog，初始 attack；A2=Lizj_dog，初始 assist。默认开运动，ENABLE_MOTION=false 只推理。" >&2
  echo "默认全队自动攻防：进攻 attack+assist，防守双阻挡。TEAM_ROLE 可覆盖为固定 attack|assist|defense。" >&2
  echo "可用 DOG_NAME 指定本机运动命名空间，A_WS 指定工作区。" >&2
}
if [[ $# -lt 3 || $# -gt 4 ]]; then usage; exit 2; fi
ROLE="$1"
OPPONENT_TOPIC="$2"
OPPONENT2_TOPIC="$3"
GOAL="${4:-${ATTACK_GOAL:-goal_right}}"
ENABLE_MOTION="${ENABLE_MOTION:-true}"
DOG_NAME="${DOG_NAME:-}"
TEAM_A_1="${TEAM_A_1:-/vrpn/xll_dog/pose}"
TEAM_A_2="${TEAM_A_2:-/vrpn/Lizj_dog/pose}"
case "$ROLE" in
  A1) DOG_TOPIC="$TEAM_A_1"; TEAMMATE_TOPIC="$TEAM_A_2"; DEFAULT_ROLE=attack; SPEED_LIMIT=0.60 ;;
  A2) DOG_TOPIC="$TEAM_A_2"; TEAMMATE_TOPIC="$TEAM_A_1"; DEFAULT_ROLE=assist; SPEED_LIMIT=0.45 ;;
  *) usage; exit 2 ;;
esac
# 默认自动攻防；显式 TEAM_ROLE 保留上一版的固定角色用法。
AUTO_ROLE_SWITCH=true
AUTO_TEAM_PHASE=true
if [[ -n "${TEAM_ROLE:-}" ]]; then
  AUTO_ROLE_SWITCH=false
  AUTO_TEAM_PHASE=false
fi
TEAM_ROLE="${TEAM_ROLE:-$DEFAULT_ROLE}"
case "$TEAM_ROLE" in
  attack|assist|defense) ;;
  *) echo "TEAM_ROLE 只能是 attack、assist 或 defense" >&2; exit 2 ;;
esac
case "$GOAL" in goal_right|goal_left) ;; *) usage; exit 2 ;; esac
case "$ENABLE_MOTION" in true|false) ;; *) echo "ENABLE_MOTION 只能是 true 或 false" >&2; exit 2 ;; esac
TOPICS=("$DOG_TOPIC" "$TEAMMATE_TOPIC" "$OPPONENT_TOPIC" "$OPPONENT2_TOPIC")
for ((i=0; i<4; i++)); do
  if [[ ! "${TOPICS[i]}" =~ ^(/[A-Za-z_][A-Za-z0-9_]*)+$ ]]; then
    echo "无效动捕话题：${TOPICS[i]}（请填写 /vrpn/刚体名称/pose）" >&2; exit 2
  fi
  for ((j=0; j<i; j++)); do
    if [[ "${TOPICS[i]}" == "${TOPICS[j]}" ]]; then
      echo "四只狗必须使用不同话题：${TOPICS[i]} 重复" >&2; exit 2
    fi
  done
done

# 不启用 nounset，兼容 Galactic/colcon 的 setup.bash。
if ! command -v ros2 >/dev/null 2>&1; then
  for base in /opt/ros2/galactic/setup.bash /opt/ros/galactic/setup.bash; do
    if [[ -f "$base" ]]; then source "$base"; break; fi
  done
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${A_WS:-}"
if [[ -z "$WS" ]]; then
  for candidate in "$SCRIPT_DIR/../.." "$PWD" "$HOME/A_ws" "$HOME/lizj" "$HOME/LIZJ"; do
    if [[ -f "$candidate/install/setup.bash" ]]; then WS="$candidate"; break; fi
  done
fi
if [[ -n "$WS" ]]; then
  if [[ ! -f "$WS/install/setup.bash" ]]; then
    echo "找不到 $WS/install/setup.bash，请先编译 pkg_soccer" >&2; exit 2
  fi
  source "$WS/install/setup.bash"
fi
if ! command -v ros2 >/dev/null 2>&1; then
  echo "未找到 ros2，请先 source 机器狗的 ROS 2 环境" >&2; exit 2
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
POLICY_DIR="$(python3 -c 'from pathlib import Path; import pkg_soccer; print(Path(pkg_soccer.__file__).resolve().parent)')"
ATTACK_MODEL="$POLICY_DIR/policy_2v2_attack_actor.npz"
ASSIST_MODEL="$POLICY_DIR/policy_2v2_assist_actor.npz"
if [[ ! -f "$ATTACK_MODEL" || ! -f "$ASSIST_MODEL" ]]; then
  echo "安装目录缺少 T11 双网络权重，请重新 colcon build --packages-select pkg_soccer" >&2; exit 2
fi

# 多台狗共享 ROS 网络时不能猜测要控制哪台，必须明确 DOG_NAME。
if [[ -z "$DOG_NAME" && "$ENABLE_MOTION" == true ]]; then
  mapfile -t NAMES < <(timeout 8s ros2 topic list 2>/dev/null |
    sed -n 's#^/\(.*\)/motion_status$#\1#p' | sort -u)
  if [[ ${#NAMES[@]} -ne 1 ]]; then
    echo "无法唯一确定本机运动接口，请设置 DOG_NAME（motion_status 前面的命名空间）。" >&2
    printf '发现：%s\n' "${NAMES[@]}" >&2
    exit 2
  fi
  DOG_NAME="${NAMES[0]}"
fi
DOG_NAME="${DOG_NAME#/}"; DOG_NAME="${DOG_NAME%/}"
echo "T11 | $ROLE=$TEAM_ROLE | dog=$DOG_TOPIC | teammate=$TEAMMATE_TOPIC | 自动换位=$AUTO_ROLE_SWITCH | 全队自动攻防=$AUTO_TEAM_PHASE"
echo "对手=$OPPONENT_TOPIC,$OPPONENT2_TOPIC | 进攻=$GOAL | 速度=$SPEED_LIMIT m/s | 运动=$ENABLE_MOTION | 接口=$DOG_NAME"
echo "持续比赛=true：比赛事件不停车；硬件故障和定位失效仍停车"
echo "权重：$ATTACK_MODEL + $ASSIST_MODEL；队友预测=regression_velocity"
ARGS=(
  -p control_mode:=rl -p rl_policy:=shared_2v2
  -p "team_role:=$TEAM_ROLE" -p "rl_auto_role_switch:=$AUTO_ROLE_SWITCH"
  -p "rl_auto_team_phase:=$AUTO_TEAM_PHASE"
  -p "rl_attack_model_path:=$ATTACK_MODEL" -p "rl_assist_model_path:=$ASSIST_MODEL"
  -p rl_continuous_play:=true -p rl_fast_start:=true -p rl_tracking_fallback:=true
  -p rl_teammate_prediction_model:=regression_velocity
  -p "rl_speed_limit:=$SPEED_LIMIT" -p rl_speed_gain:=0.0
  -p rl_continue_deadzone:=true -p rl_deadzone_speed_limit:=0.35
  -p "dog_pose_topic:=$DOG_TOPIC" -p "teammate_pose_topic:=$TEAMMATE_TOPIC"
  -p "opponent_pose_topic:=$OPPONENT_TOPIC" -p "opponent2_pose_topic:=$OPPONENT2_TOPIC"
  -p "attack_goal:=$GOAL" -p "enable_motion:=$ENABLE_MOTION"
)
if [[ -n "$DOG_NAME" ]]; then ARGS+=(-p "dog_name:=$DOG_NAME"); fi
exec ros2 run pkg_soccer soccer --ros-args "${ARGS[@]}"
