# pkg_1106_slam

节点 `human_tracking` 通过 `protocol/action/Navigation` 启动小米机器狗二代的
人体跟随。收到反馈 `501` 后，节点从 `person` 话题读取人体检测框。连续两帧
只检测到一个人体框时，节点将最新的人体框提交给 `tracking_object_srv`，
由机器狗锁定目标并跟随。

## 编译和运行

将更新后的 `pkg_1106_slam` 复制到机器狗工作空间的 `src`，执行：

```bash
cd ~/lizj
colcon build --packages-select pkg_1106_slam
source install/setup.bash
ros2 action list -t | grep start_algo_task
```

如果先前为查看 `/image` 单独启动了 `ros2 run camera_test camera_server`，
在那个终端按 Ctrl+C 结束手动启动的实例。跟随 Action 会让机器狗的视觉模块
启动算法用的相机流。若在机器狗自带的相机服务上手动开启了预览，也可以先
通过同一个服务调用 `{command: 10}` 结束自己开启的预览。保留系统自带的节点。

```bash
ros2 run pkg_1106_slam human_tracking --ros-args \
  -p dog_name:=mi_desktop_48_b0_2d_5f_bd_93 \
  -p duration_sec:=60.0
```

把 `dog_name` 改成 `start_algo_task` 的实际命名空间。收到 `501` 后，
让一人留在 AI 交互相机视野内，等到 `502` 或 `503` 再走动。程序只为本次
Goal 提交一次人体框。若目标彻底丢失，可退出后重新运行。
`duration_sec:=0.0` 表示跟随至手动停止。按 Ctrl+C 会请求取消当前 Goal。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `dog_name` | `mi_desktop_48_b0_2d_5f_bd_93` | 设备命名空间 |
| `action_name` | `/<dog_name>/start_algo_task` | Action 路径，人体检测话题和目标选择服务使用相同命名空间 |
| `image_topic` | 空字符串 | 可选预览监视，例如 `/image` 或 `/<dog_name>/image` |
| `keep_distance` | `1.2` | Goal 发送的距离（米），部分固件使用服务端配置 |
| `duration_sec` | `60.0` | 从发送 Goal 到请求取消的时间 |
| `feedback_period` | `2.0` | 等待人体检测、预览状态的日志间隔（秒） |

## 排查停帧与跟随状态

跟随时，`vision_manager` 从 AI 相机的算法流读取共享内存；`/image` 是同一
相机可单独开启的 ROS 预览。跟随定位还使用 RealSense 深度数据。可在第二个
终端直接观察人体检测：

```bash
ros2 topic list -t | grep -E 'person|image'
ros2 topic echo /实际机器狗名称/person --qos-reliability best_effort
```

| 输出 | 含义 |
| --- | --- |
| `code=500` | 正在启动视觉跟随和依赖节点 |
| `code=501` | 服务端就绪，等待人体检测框 |
| `code=502`、`code=503` | 已锁定目标、正在跟随 |
| `code=506`、`code=507` | 依赖启动失败、目标锁定失败 |
| `result=5` | 跟随任务取消，查看此前是否达到设定时长或手动停止 |

官方相机实现添加算法流时会暂停已有的重复采集请求。新增输出或恢复重复采集
失败时，预览可能持续停帧。若只看到 `500`，且 `person` 一直没有消息，可在
机器狗上检查 `ros2 node list`、`ros2 service list -t`，并查看
`camera_server`、`vision_manager`、`algorithm_manager` 的同期日志。

- [小米机器狗二代开发者手册](https://github.com/MiRoboticsLab/blogs/blob/rolling/docs/cn/developer_guide.md)
- [视觉跟随服务端](https://github.com/MiRoboticsLab/cyberdog_nav2/blob/rolling/algorithm_manager/src/executor_vision_tracking.cpp)
- [AI 相机采集请求实现](https://github.com/MiRoboticsLab/cyberdog_camera/blob/rolling/camera_service/src/argus_camera_context.cpp)
