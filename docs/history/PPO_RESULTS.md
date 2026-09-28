> Historical experiment report. For the public release workflow and current limits, see [README.md](../../README.md) and [docs/RESULTS.md](../RESULTS.md).

# G1 0.8 m 跳高：PPO 状态反馈与跨引擎结果（2026-09-24）

本次按用户要求新增真正的强化学习状态反馈策略。最终视频是在 Isaac Sim 4.5 / PhysX 中逐步读取当前状态、运行冻结网络、施加关节电机力矩并同步渲染得到的；不写回参考根轨迹，不是 qpos 动画回放。策略在 MuJoCo-Warp 中训练，转移到 PhysX 后标定了一个反馈增益，因此这是**经过标定的 sim2sim 成功回合，不是零样本迁移或实机成功**。

## 学到了什么

旧 CEM 时间曲线用于生成一个动力学可执行示范、部署前馈参考及固定目标对照。新策略用 PPO 和模仿跟踪奖励训练：2048 并行环境、500 次迭代、32,768,000 环境步，actor 为 256/256/128 ELU MLP。网络读取参考关节状态、当前编码器 q/dq、骨盆 IMU 重力方向/角速度和时钟相位，共 152 维，输出 29 个关节目标角（50 Hz）；关节 PD 与 DC 转矩–转速限制在 400 Hz 运行。实际世界位置/线速度/接触真值只供训练 critic、奖励和验收，不能进入 actor。

使用差分残差结构：同一网络的实际状态输出减去参考状态输出；在参考状态上修正为零。目标是参考前馈加学到的状态修正，仍需要本地参考文件。它不是从零学出的无参考平衡策略，也不是官方 Mimic 模型。详细观测契约见 METHOD.md / docs/RL_METHOD.md。

## 有效反馈证据与限制

以下为 native MuJoCo 的独立初始扰动测试，成功包含不碰杆、全身越杆、落垫/背部支撑顺序及关节范围和速度检查。每组 50 个扰动回合不包含干净 nominal；固定目标对照仍保留底层 PD。

| 网络部署配置 | 独立种子 | 小扰动 ±0.01 rad | 较大扰动 ±0.025 rad |
|---|---|---:|---:|
| PPO 原始反馈增益 1.0 | 101–150 | 29/50 | 10/50 |
| 同组固定目标 | 101–150 | 15/50 | 4/50 |
| 高层状态替换为参考的消融，增益 1.0 | 101–150 | 15/50 | 未运行 |
| **最终跨引擎版本，增益 0.5** | 201–250 | **19/50** | **3/50** |
| 同组固定目标 | 201–250 | 13/50 | 4/50 |

扰动分别施加到初始关节角及 roll/pitch，初始水平速度范围为上述幅度的三倍（±0.03 或 ±0.075 m/s）。这些结果表明原始模型确有有用的高层反馈，但鲁棒性有限；最终 0.5 增益版本在较大扰动下没有优于基线，不能把原始增益的 29/50 冒充最终版本成绩。普通 PPO 的早期试验未显示有效纠错，保留为诊断。

## PhysX 视频回合

冻结同一 PPO 权重，仅比较反馈增益 0.5、1、2、4 后采用 0.5。原始增益 1 的试验存在横杆接触，已拒收。最终回合及录像复核：

- 杆中心高度 0.800 m，半径 0.012 m，固定但可碰撞；任何借杆支撑都不接受。垫顶 0.18 m，PhysX 中使用刚性碰撞盒。
- 全身在 1.3875 s 越杆；右手在 1.3925 s 开始落垫缓冲，躯干约 1.49 s 落垫，头部更晚接触。
- 开启 PhysX 实际接触报告，7827 条报告中没有横杆接触；最小碰撞几何杆面间隙 **1.8945 mm**，余量很小。
- 1048 个真实 PhysX 子步的胶囊/球精确几何验收通过；不是用 MuJoCo 重模拟产生成功。原有保守 AABB 包围盒测试会误判这条 PhysX 轨迹，因此另给出精确切片与有限半径杆距检查、独立 PhysX 接触证据。精确切片有解析/密集采样单测。
- 最大关节限位数值超出 0.000512 rad，最大速度/额定速度 0.9386；满足既定数值容差，但没有验证真实冲击安全。
- 有摄像机录像与无摄像机运行的全部子步 qpos 完全一致。视频 1280×720、50 FPS、131 帧、2.62 s，为仿真时间正常速度。
- ONNX 与 TorchScript 在两引擎共 262 帧输入上的最大差约 1.2e-7 rad；导出网络目标与录像实际执行目标匹配。

动作是原地双脚起跳、背部朝下越杆落垫；**尚不是助跑单脚起跳的标准田径背越式**。只验证了选定初始状态的一条 PhysX 成功轨迹，尚无 PhysX 扰动统计，也没有完成质量/摩擦/传感器噪声/延迟随机化及实机测试。不能保证平衡、恢复或直接部署 G1。

## 文件与复现

远端项目：`<project-root>`。
最终包：`artifacts/final-rl-sim2sim/`：

- `g1-ppo-physx-online.mp4`：最终跨引擎在线视频。
- `g1-ppo-mujoco-online.mp4`：相同增益模型的 native MuJoCo 在线视频。
- `policy.onnx` / `policy.torchscript`：29 关节绝对目标角模型。
- `commands.npz`、`contract.json`：参考命令、关节顺序、缩放及电机参数。
- `training-checkpoint.pt`、`evaluation-reference.npz`、`scene/`：训练权重与审计资产。
- `physx/`、`native/`、`validation/`：录像对应实际状态、网络输入输出、接触和独立评估证据。
- `export-verification.json`、`manifest.json`、`SHA256SUMS`：导出一致性和文件校验。

训练与 native 验证：

```bash
cd <project-root>
./scripts/runtime.sh scripts/prepare_rl.py
HIGHJUMP_GPU=7 ./scripts/runtime.sh scripts/train_rl.py --name reproduce-feedback --num-envs 2048 --iterations 500 --perturb .025 --feedback-only
./scripts/runtime.sh scripts/evaluate_rl.py --help
./scripts/runtime.sh tools/validation/audit_physx_geometry.py --run artifacts/rl-isaac/gain0.5-online-video
./scripts/runtime.sh -m unittest discover -s tests -v
```

PhysX 在线录像使用现有 Isaac 环境（不修改系统驱动），运行：

```bash
./scripts/isaac_runtime.sh scripts/isaac/evaluate_isaac.py --headless --enable_cameras --device cuda:1 --physics-device cpu --video --policy artifacts/final-rl-sim2sim/policy.torchscript --output artifacts/rl-isaac/reproduce-video
./scripts/runtime.sh tools/validation/audit_physx_geometry.py --run artifacts/rl-isaac/reproduce-video
```

PhysX 模型由 `scripts/isaac/prepare_isaac_model.py`、`scripts/isaac/isaac_convert.py` 转换得到，并保留在 `artifacts/rl-isaac/`。该脚本依赖本项目资产及既有训练环境，不是独立实机启动器。
