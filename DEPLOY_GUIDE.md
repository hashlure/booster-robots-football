# Isaac Lab → booster_gym Deploy 问题与解决

## 背景

在 Isaac Lab 训练 T1 走路模型，通过 booster_gym deploy 部署到真机/Webots 时，机器人无法正常行走（爆炸/摔倒/静止）。原因是训练环境和部署环境存在多个隐性不匹配。

---

## 问题 1：关节顺序不一致（致命）

### 现象
Isaac Lab 仿真正常，部署后机器人动作完全错乱。

### 原因
Isaac Lab 内部使用 PhysX BFS 顺序排列关节，deploy SDK 使用串口顺序。两套顺序完全不同。

### 定位
通过 `replay_amp_txt.py` 的 `reorder()` 函数发现两套顺序的映射关系。

### 解决
在部署端 `policy_*.py` 的观测构建和动作输出处，加入双向重排：

```python
# 双向重排 (SDK serial ↔ Isaac Lab PhysX BFS)
# sdk_from_isaac[isaac_i] = sdk_i → 观测 SDK→Isaac: dof_pos[sdk_from_isaac]
sdk_from_isaac = [0,2,6,10,1,3,7,11,17,4,8,12,18,5,9,13,19,14,20,15,21,16,22]
# isaac_from_sdk[sdk_i] = isaac_i → 动作 Isaac→SDK: actions[isaac_from_sdk]
isaac_from_sdk = np.argsort(sdk_from_isaac)

# 观测: dof_pos[sdk_from_isaac], dof_vel[sdk_from_isaac]
# 动作: actions[isaac_from_sdk]
```

---

## 问题 2：PD 增益不匹配

### 现象
模型在仿真中行走正常，部署后关节动作幅度异常。

### 原因
Isaac Lab 训练的 PD 参数（BOOSTER_T1_CFG）和真机 firmware 参数（T1.yaml）不一致：

| 关节 | Isaac Lab 训练 | 真机 |
|------|:--:|:--:|
| 手臂 | 50/1.0 | **20/0.5** |
| 头部 | 10/1.0 | **20/0.2** |
| 踝关节 | 50/1.0 | **50/0**（平行机构纯P力矩） |

### 解决
在 `env_cfg.py` 中覆盖 PD 增益为真机值：

```python
DEPLOY_PD = {
    "head":  {"stiffness": 20, "damping": 0.2},
    "arms":  {"stiffness": 20, "damping": 0.5},
    "waist": {"stiffness": 200, "damping": 5},
    "legs":  {"stiffness": 200, "damping": 5},
    "feet":  {"stiffness": 50, "damping": 0},
}
```

---

## 问题 3：default_qpos 不一致

### 现象
默认站立姿态和模型学习的姿态有偏差，导致初始动作跳变。

### 原因
`BOOSTER_T1_CFG.init_state.joint_pos` 和真机 `T1.yaml` 的 `default_qpos` 不同：
- 肩 Roll: ±1.3 vs ±1.35
- 踝 Pitch: -0.2 vs -0.25

### 解决
`env_cfg.py` 中覆盖为真机值：

```python
DEPLOY_QPOS = {
    ".*_Shoulder_Pitch": 0.2,
    "Left_Shoulder_Roll": -1.35,
    "Right_Shoulder_Roll": 1.35,
    ".*_Ankle_Pitch": -0.25,
    ...
}
```

---

## 问题 4：Empirical Normalization

### 现象
部署时观测数值分布和训练时不同，模型输出异常。

### 原因
Isaac Lab 默认使用 `empirical_normalization=True`，运行时统计的均值/方差。部署时没有这个 normalizer，观测裸值进入模型。

### 解决
两种方案：
- **方案A**：训练时关掉 `empirical_normalization = False`，模型直接学习裸值分布
- **方案B**：导出时 bake normalizer 进 JIT 模型

walk_deploy2 使用方案A。

---

## 问题 5：use_default_offset

### 现象
策略输出 0 动作时，机器人不是保持站姿而是全关节拉直。

### 原因
K1 的 `ActionsCfg` 使用 `use_default_offset=False`，0 动作 = 关节角为 0。T1 的站立姿态需要特定关节角（Hip=-0.2, Knee=0.4）。

### 解决
设置 `use_default_offset = True`，0 动作 = 保持 default_joint_pos。

---

## 问题 6：踝关节平行机构力矩控制

### 现象
仿真正常，真机踝关节振荡剧烈，快速摔倒。

### 原因
真机踝关节通过 parallel_mech 力矩控制（纯 P，无阻尼）。Isaac Lab 使用 PD 位置控制（有阻尼）。模型在仿真中学习了依赖阻尼稳定。

### 解决
Isaac Lab 训练时设置踝关节 `damping = 0`，匹配真机力矩模式。

---

## 文件结构

```
t1/walk_deploy2/           ← 部署对齐版训练任务
├── env_cfg.py             PD + default_qpos 覆盖
├── ppo_cfg.py             关 empirical normalization
├── tracking_env_cfg.py    完全继承 walk2run_amp
└── __init__.py

booster_gym/deploy/
├── deploy_fullbody.py     全身版部署脚本
├── utils/policy_deploy2.py 部署端 policy（关节重排 + 78维观测）
└── configs/T1_fullbody.yaml 部署配置
```

## 训练命令

```bash
python scripts/rsl_rl/train.py --task=Booster-Walk-Deploy2-T1-v0 --num_envs=4096 --device=cuda:0 --headless
```

## 导出命令

```bash
python scripts/export_deploy.py --checkpoint logs/rsl_rl/walk_deploy2_t1/<run>/model_xxx.pt
```
