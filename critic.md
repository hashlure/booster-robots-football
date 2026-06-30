你现在的问题非常关键：**仿真里“看着能走”只是第一层，真正要做 sim2real，需要把策略从“视频好看”变成“数据可比较”。**

你可以把评价分成两层：

```text
第一层：安全门槛
能不能上真机前测试？

第二层：性能排序
三个看起来差不多的策略，哪个更稳、更省力、更像真机可执行？
```

不要只看训练 reward。reward 是训练信号，不一定是评测指标。评测指标应该独立出来。

---

# 1. 先建立一个固定 replay benchmark

你以后每个策略都用同一套 replay 测试。

比如：

```text
策略 A
策略 B
策略 C
```

都跑同样的测试集：

```text
1. 原地站立 20 s
2. 低速前进 vx = 0.2 m/s
3. 中速前进 vx = 0.5 m/s
4. 快速前进 vx = 0.8 m/s
5. 左右横移 vy = ±0.2 m/s
6. 原地转向 yaw_rate = ±0.5 rad/s
7. 前进 + 转向
8. 轻微随机推扰
9. 地面摩擦随机
10. 质量 / COM 小随机
```

每个场景跑：

```text
num_envs = 128 或 256
episode_length = 10~20 s
固定随机种子
```

最后得到一个表，而不是只看视频。

---

# 2. 最重要的第一类指标：稳定性

这是最优先的。人形机器人上真机之前，稳定性是硬门槛。

## 2.1 Fall rate / 摔倒率

定义：

```text
fall_rate = 摔倒 episode 数 / 总 episode 数
```

摔倒判据可以是：

```text
base_height < 阈值
abs(base_roll) > 阈值
abs(base_pitch) > 阈值
机器人非脚部 link 接触地面
```

比如：

```text
base_height < 0.55 m → fall
abs(roll) > 0.7 rad → fall
abs(pitch) > 0.7 rad → fall
```

三个策略首先看这个：

| 策略 | fall rate |
| -- | --------: |
| A  |        0% |
| B  |        3% |
| C  |       12% |

这种情况下，哪怕 C 走路姿态看起来不错，也不能优先。

---

## 2.2 Survival time / 平均存活时间

如果有些策略会摔，看它平均能坚持多久：

```text
survival_time = episode 结束前未摔倒的持续时间
```

例如：

| 策略 | survival time |
| -- | ------------: |
| A  |        20.0 s |
| B  |        18.7 s |
| C  |        11.4 s |

这个指标特别适合比较不稳定策略。

---

## 2.3 Base attitude RMS / 身体姿态稳定性

看身体 roll、pitch 抖不抖：

```text
roll_rms  = sqrt(mean(roll^2))
pitch_rms = sqrt(mean(pitch^2))
```

roll/pitch 越小，身体越稳。

也可以看角速度：

```text
base_ang_vel_rms = sqrt(mean(wx^2 + wy^2 + wz^2))
```

如果一个策略走路看起来不错，但是 base angular velocity 很大，说明身体在高频晃，上真机容易炸。

---

# 3. 第二类指标：命令跟踪能力

如果你训练的是 command-based locomotion，policy 应该按照给定速度走。

## 3.1 线速度跟踪误差

比如命令是：

```text
vx_cmd = 0.5 m/s
vy_cmd = 0.0 m/s
```

实际机器人 base velocity 是：

```text
vx_actual
vy_actual
```

指标：

```text
vx_error = mean(abs(vx_actual - vx_cmd))
vy_error = mean(abs(vy_actual - vy_cmd))
```

或者用 RMSE：

```text
velocity_tracking_rmse = sqrt(mean((vx - vx_cmd)^2 + (vy - vy_cmd)^2))
```

三个策略可能视频都像在走，但一个实际 0.5 m/s，一个实际 0.35 m/s，一个实际 0.65 m/s，这就不一样了。

---

## 3.2 角速度跟踪误差

转向任务看：

```text
yaw_rate_error = mean(abs(yaw_rate_actual - yaw_rate_cmd))
```

人形机器人转向很容易露馅。
有些策略直走很好，但一转向就脚滑、身体晃、腿交叉。

---

## 3.3 速度 overshoot / 超调

如果命令从 0 切到 0.5 m/s，好的策略应该平滑加速，而不是突然冲出去。

可以看：

```text
overshoot = max(vx_actual - vx_cmd)
```

太大的 overshoot 真机上危险。

---

# 4. 第三类指标：执行器安全性

这个和 sim2real 最直接相关。

很多策略仿真里看起来差不多，但电机负载完全不同。上真机时，电机受不了的策略会发热、限流、抖动、摔倒。

---

## 4.1 Torque RMS / 平均力矩

每个关节都有 torque：

```text
τ_i(t)
```

计算：

```text
torque_rms = sqrt(mean(τ_i(t)^2))
```

可以分别统计：

```text
hip torque rms
knee torque rms
ankle torque rms
arm torque rms
```

如果策略 A 和 B 都能走，但 A 的膝关节力矩 RMS 是 B 的 1.5 倍，A 上真机风险更高。

---

## 4.2 Torque saturation ratio / 力矩饱和比例

这个非常重要。

定义：

```text
saturation_ratio = abs(τ_actual) / τ_limit
```

统计超过某个比例的时间：

```text
sat_80 = mean(abs(τ) > 0.8 * τ_limit)
sat_95 = mean(abs(τ) > 0.95 * τ_limit)
```

如果一个策略经常接近最大力矩，说明它在“压榨电机”。

判断建议：

```text
sat_95 < 1%      很好
sat_95 1%~5%     勉强
sat_95 > 5%      真机风险较高
```

---

## 4.3 Velocity saturation ratio / 速度饱和比例

同理：

```text
vel_sat = mean(abs(qd) > 0.8 * qd_limit)
```

如果关节速度经常接近极限，真机电机跟踪会变差。

---

## 4.4 Power / 功率

机械功率：

```text
power_i = τ_i * qd_i
```

常用统计：

```text
mean_abs_power = mean(abs(τ * qd))
positive_power = mean(max(τ * qd, 0))
```

功率越大，电机越容易发热。
sim2real 时很关键。

---

## 4.5 Cost of Transport / 运输代价

这个是腿足机器人很常用的能耗指标：

```text
CoT = energy / (mass * g * distance)
```

应用上你只需要知道：

```text
CoT 越低，说明走同样距离用的能量越少
```

如果三个策略都能走，CoT 更低的通常更自然、更省电、更适合真机。

---

# 5. 第四类指标：动作平滑性

很多仿真策略看起来能走，但是动作抖。真机上高频抖动非常危险。

---

## 5.1 Action rate / 动作变化率

如果 policy 输出 action：

```text
a_t
```

计算：

```text
action_rate = mean(||a_t - a_{t-1}||^2)
```

越大说明 policy 输出变化越剧烈。

---

## 5.2 Joint acceleration / 关节加速度

```text
joint_acc = (qd_t - qd_{t-1}) / dt
```

统计：

```text
joint_acc_rms
joint_acc_max
```

如果关节加速度很大，说明动作冲击强，容易激发真机机械振动。

---

## 5.3 Jerk / 加加速度

更严格可以看 jerk：

```text
jerk = (qdd_t - qdd_{t-1}) / dt
```

应用上不一定一开始就做 jerk，先做 action rate 和 joint acceleration 就够了。

---

# 6. 第五类指标：脚底接触质量

人形机器人走路好不好，脚底接触很关键。

---

## 6.1 Foot slip / 脚底打滑

当脚在接触地面时，它的水平速度应该接近 0。

定义：

```text
如果 foot_contact = True:
    foot_slip += ||foot_xy_velocity||
```

指标：

```text
mean_foot_slip = 接触阶段脚底水平速度平均值
```

如果脚接触地面时还在滑，说明策略依赖了仿真摩擦，真机风险高。

这是非常重要的 sim2real 指标。

---

## 6.2 Contact force peak / 接触力峰值

落脚冲击太大，真机容易震、摔、伤机械结构。

看：

```text
max_foot_contact_force
mean_foot_contact_force
contact_force_impulse
```

如果一个策略落脚时接触力峰值特别大，即使视频好看，也不是好策略。

---

## 6.3 Duty factor / 支撑相比例

对每只脚：

```text
duty_factor = 接触地面的时间 / 总时间
```

正常走路时左右脚 duty factor 应该比较稳定，而且左右差不多。

如果出现：

```text
左脚 80% 接触
右脚 20% 接触
```

说明步态不对称。

---

## 6.4 Double support ratio / 双脚支撑比例

统计双脚同时接触的时间比例：

```text
double_support_ratio = 双脚都接触的时间 / 总时间
```

太高：可能拖着走，不灵活。
太低：可能跑跳式，风险大。
是否好取决于你的目标步态。

---

## 6.5 Foot clearance / 摆动脚离地高度

摆动脚不能太低，不然真机容易绊地；也不能太高，不然耗能大。

统计：

```text
swing_foot_clearance_mean
swing_foot_clearance_min
```

应用判断：

```text
clearance 太小 → 容易绊脚
clearance 太大 → 浪费能量，动作夸张
```

---

# 7. 第六类指标：关节运动是否合理

这个主要用于排除“看起来能走但动作畸形”的策略。

---

## 7.1 Joint limit margin / 关节限位余量

看关节是否经常接近极限：

```text
joint_limit_margin = min(q - q_min, q_max - q)
```

如果某些关节频繁贴近 limit，真机风险很高。

---

## 7.2 Joint position RMS / 关节摆幅

比较不同策略关节活动幅度：

```text
joint_pos_rms
joint_vel_rms
```

比如一个策略手臂乱甩，视频可能不明显，但 arm joint RMS 会暴露。

---

## 7.3 左右对称性

对于直走任务，左右腿应该大致对称。

可以比较：

```text
left_hip_pitch vs right_hip_pitch
left_knee vs right_knee
left_ankle vs right_ankle
```

不是要求完全一样，而是周期、幅值、相位要合理。

---

# 8. 第七类指标：鲁棒性

sim2real 最终看的是鲁棒性，不是单一理想环境下跑得多漂亮。

你可以做 stress test：

```text
1. mass ±5%
2. COM ±2 cm
3. friction 0.5~1.2
4. motor strength 0.8~1.1
5. action delay 0~2 steps
6. observation noise
7. push disturbance
```

然后看：

```text
fall_rate
velocity_error
torque_sat
foot_slip
```

真正好的策略是：
**标称环境不是最漂亮，但随机扰动下仍然稳定。**

---

# 9. 实际排序时怎么做？

我建议你用这个顺序。

## 第一步：先过安全门槛

如果一个策略满足：

```text
fall_rate < 1%
sat_95 < 5%
foot_slip 不明显
base_roll/pitch 不大
joint_limit_violation = 0
```

才进入下一轮比较。

不满足的策略，直接淘汰。

---

## 第二步：比较综合分数

你可以定义一个简单 score：

```text
score =
  0.30 * velocity_tracking_score
+ 0.20 * stability_score
+ 0.20 * energy_score
+ 0.15 * smoothness_score
+ 0.15 * contact_score
```

不用一开始搞很复杂。重点是统一标准。

更简单一点，可以直接用排序：

```text
1. 摔倒率最低
2. 命令跟踪误差最低
3. 力矩饱和最少
4. 脚底打滑最少
5. action rate 最小
6. CoT 最低
```

---

# 10. 你可以直接用的评价表

以后每个策略 replay 完，生成这种表：

| 指标                  |  策略 A |  策略 B |  策略 C | 越小/越大越好 |
| ------------------- | ----: | ----: | ----: | ------- |
| fall rate           |    0% |    2% |    0% | 越小越好    |
| survival time       | 20.0s | 18.5s | 20.0s | 越大越好    |
| vx tracking RMSE    |  0.06 |  0.04 |  0.09 | 越小越好    |
| yaw tracking RMSE   |  0.08 |  0.11 |  0.07 | 越小越好    |
| roll RMS            |  0.04 |  0.07 |  0.05 | 越小越好    |
| pitch RMS           |  0.05 |  0.08 |  0.06 | 越小越好    |
| torque RMS          |  31.2 |  42.7 |  28.5 | 越小越好    |
| torque sat > 95%    |  0.3% |  4.8% |  0.7% | 越小越好    |
| joint vel sat > 90% |  1.2% |  3.5% |  0.9% | 越小越好    |
| mean foot slip      |  0.03 |  0.08 |  0.04 | 越小越好    |
| contact force peak  |  620N |  890N |  700N | 越小越好    |
| action rate         |  0.12 |  0.26 |  0.15 | 越小越好    |
| CoT                 |  0.78 |  0.92 |  0.73 | 越小越好    |

比如这个表里：

```text
策略 B 命令跟踪最好，但摔倒率、力矩、脚滑、动作抖都偏大。
策略 C 能耗最低，稳定性也不错，但速度跟踪差一点。
策略 A 比较均衡。
```

如果要上真机，我可能优先选 A 或 C，而不是 B。

---

# 11. 你 replay 时最应该 log 哪些量？

最小够用版：

```text
base_pos
base_quat
base_lin_vel
base_ang_vel

projected_gravity
roll / pitch / yaw

joint_pos
joint_vel
joint_torque
action
target_joint_pos

foot_contact
foot_contact_force
foot_pos
foot_vel

command_vx
command_vy
command_yaw_rate

episode_done
fall_flag
```

如果你能拿到这些，前面大多数指标都能算。

---

# 12. 最小够用的 replay 评价指标

如果你不想一开始做太复杂，就先做这 10 个：

```text
1. fall_rate
2. survival_time
3. velocity_tracking_rmse
4. yaw_tracking_rmse
5. roll_pitch_rms
6. torque_rms
7. torque_saturation_ratio
8. action_rate
9. foot_slip
10. contact_force_peak
```

这 10 个足够你区分大部分“看起来差不多”的策略。

---

# 13. 为什么不能只看姿态好不好？

因为有些策略视频看着很好，但背后可能是：

```text
电机力矩长期接近饱和
脚底轻微打滑
身体高频小幅振动
关节速度接近极限
action 输出剧烈抖动
落脚冲击很大
对摩擦变化很敏感
```

这些在视频里不一定明显，但上真机就会出问题。

你现在要从“看视频调 reward”升级成：

```text
看视频 + 看曲线 + 看统计表 + 看鲁棒性测试
```

这才是真正的 sim2real 工程流程。

---

# 14. 你可以怎么落地到 Isaac Lab replay？

你的 replay 脚本里每一步做：

```python
obs = env.get_observations()
action = policy(obs)
obs, reward, done, info = env.step(action)

logger.add({
    "base_lin_vel": robot.data.root_lin_vel_b,
    "base_ang_vel": robot.data.root_ang_vel_b,
    "projected_gravity": robot.data.projected_gravity_b,
    "joint_pos": robot.data.joint_pos,
    "joint_vel": robot.data.joint_vel,
    "joint_torque": robot.data.applied_torque,
    "action": action,
    "target_joint_pos": target_joint_pos,
    "contact_forces": contact_sensor.data.net_forces_w,
    "commands": commands,
    "done": done,
})
```

最后 episode 结束统一算指标。

注意：不同 Isaac Lab 版本字段名可能略有差异，但思路就是这个。

---

# 15. 推荐你现在的工作方式

你以后每训练出一个策略，不要直接说“这个看着不错”。

改成：

```text
1. 保存视频
2. 保存 replay log
3. 输出 metrics.csv
4. 输出 summary.json
5. 输出关键曲线图
```

关键曲线图包括：

```text
vx_cmd vs vx_actual
yaw_cmd vs yaw_actual
base roll/pitch
left/right foot contact
joint torque
torque saturation
action rate
foot slip
```

这样你和别人讨论策略时，就不是说：

> “我感觉这个策略更稳。”

而是说：

> “这个策略在 256 个 episode 上 fall rate 为 0%，vx RMSE 是 0.05 m/s，torque 95% saturation 只有 0.7%，foot slip 比 baseline 下降 40%，所以我认为它更适合上真机。”

这就是定量评价。

---

一句话总结：

**三个策略看起来差不多时，不要比视频，直接用固定 replay benchmark 比：摔倒率、速度跟踪误差、身体姿态 RMS、力矩/速度饱和、能耗、action 平滑性、脚底打滑、接触力峰值和扰动鲁棒性。对 sim2real 来说，最重要的不是“仿真里走得漂亮”，而是“电机不过载、脚不滑、身体不抖、动作平滑、扰动下不摔”。**
