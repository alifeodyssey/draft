# Galileo X 四关节轮足概念模型

这是用户指定拓扑的概念模型，不是商业 Galileo X 的尺寸复刻。旧球足模型的生成
目录继续保留；当前配置为 `J1 pitch -> J2 局部 yaw -> J3 pitch -> J4 连续轮轴`。

## 运行与编辑

在 Conda `joint` 环境和 Draft 仓库目录运行：

```powershell
python -X utf8 scripts/generate_robot.py --robot galileo_x
python -X utf8 scripts/robot_tuner.py
python -X utf8 scripts/inspect_galileo_x.py generated/galileo_x_<timestamp> --render
```

编辑器选择 `galileo_x`，从 `New from defaults` 新建当前轮足设计，或加载本轮保存
的新设计。`Generate` 会更新当前输出目录；要保留旧球足或旧参数，先新建/另存设计。
旧生成目录不会自动升级。迁移旧参数时只带入独立输入，使用当前配置中的派生表达式，
不要让旧 `foot_sphere_r` 或球足高度表达式覆盖轮足高度。

`Reset pose` 恢复原生 `standing` keyframe，`Zero pose` 检查机械零位。停止物理
运行也恢复站姿。轮轴显示为 `Continuous hinge`，四个角速度滑块可独立正反向预览，
范围取生成模型的速度能力元数据。原始角度连续累计，界面只对显示相位取模。
`Stop wheel preview` 停止全部轮速；加载、重置、进入物理运行都会清零预览速度。
预览直接更新运动学姿态，不是地面驾驶控制器。

## 结构、轴向和质量

采用 `+X` 向前、`+Y` 向左、`+Z` 向上：

```text
torso + J1 定子（躯干前后 x 端面）
  J1 -y：转子 + 同轴外展连杆 + J2 定子
    J2 左 -z / 右 +z：转子 + 轴向连接件 + 固定斜接大腿 + J3 定子
      J3 +y：外侧输出连接件 + 外侧小腿 + 末端内侧 J4 定子
        J4 +y：内侧车轮，连续旋转
```

前后腿结构同向，左右装配镜像；J1/J3/J4 的轴正方向前后左右一致。J1 正转向机头
摆腿，J2 正转外展，J3 负转屈膝。J4 正转在站姿下对应向 `+X` 滚动。
J4 始终平行于 **J3 的局部 pitch 轴**；J2 转动会带着膝轴和轮轴一起转向，不能
将其理解为任意姿态下始终沿躯干或世界 Y。固定大腿斜角不改变 J2 轴线。

J3 轴心保留在大腿末端，小腿中心线向外偏移
`J3_length/2 + clearance + shank_radius`。轮毂中心向小腿内侧偏移
`shank_radius + clearance + wheel_width/2`，保证轮侧面和小腿有净空。
小腿长度定义为 J3 与 J4 平行轴之间的垂直距离；轴向偏置不增加这段结构长度。
连接件与小腿共用原有 `shank` 结构趋势质量，按现有长度分配规则计重。

每腿四个电机整体包络各计一次质量，归属上游刚体；J4 电机随小腿运动，净车轮
质量随 J4 转动。沿用 Draft 的整体执行器与反射惯量简化，没有实测定转子拆分数据。
轮胎采用带碰撞的圆柱和净质量等效惯量；转动标记无质量、无碰撞。

## 当前默认参数与来源

| 参数 | 当前默认值 | 来源 |
|---|---|---|
| 机体长宽高 | 0.5 / 0.30 / 0.15 m | 用户最新保存设计 `galileo_x_20261123_135703` |
| 大腿 / 小腿长 | 0.24 / 0.23 m | 用户保存设计 |
| 大腿固定斜角 | 30 deg | 用户保存设计，显式转为 rad |
| 站姿 J1 / J2 / J3 | -20 / 0 / -100 deg | J3 默认 `2*(standing_j1_deg-thigh_cant_deg)`，也可独立调整 |
| J1 / J2 / J3 行程 | [-2.5,2.5] / [-pi/2,pi/2] / [-2.8,pi/2] rad | 保留原概念验证行程 |
| J4 | 无角度限位 | 用户指定连续旋转 |
| J1–J4 电机 | L 类，150 N·m、50 rad/s、减速比 5 | 沿用的假想示例；包络、质量、armature 由趋势估算 |
| 装配净空 / 胎厚 | 5 / 15 mm | 可调示例 |
| 轮半径 / 轮宽 | 约 94.5 / 70.9 mm | `J4_radius+clearance+tire_thickness` / `J4_length+2*clearance` |
| 净车轮质量 | 0.5 kg/个 | 可调示例，不含电机 |

`standing_height` 是机械零位根高度，`standing_pose_height` 是站姿根高度，分别
按对应 FK 与轮半径计算；J4 的零位和站姿初值均为零。修改角度和杆长后重新生成
会更新高度和关键帧，站姿超限明确报错。

四个原生 MuJoCo `<motor>` 驱动器名为 `fl_j4_drive`、`fr_j4_drive`、
`rl_j4_drive`、`rr_j4_drive`。`data.ctrl[actuator_id]` 输入输出轴力矩 N·m，
传动系数为 1，控制及输出力矩受限于 ±150 N·m。减速比不在驱动器上重复相乘。
50 rad/s 是无负载速度元数据，并非理想 torque motor 的硬速度限制。J1–J3 继续
由消费方提供控制器；本轮不含站姿保持、地面驾驶、RL、优化器或 CAD 导出。

## 坐下检查

运行 `inspect_galileo_x.py`，每次修改参数后重新检查。工具将机体保持水平，从
站姿降至 `torso_lz/2`，J3 插值至 -145°；前段用接地 FK 求 J1，躯干离地最后
50 mm 内平滑增加折叠量，终点 J1=-140°，允许轮子抬起，保留原髋部安装高度。
这些检查角度是可调示例，不是新的关节限位或控制轨迹。

检查步长不超过 0.5°/1 mm，直接计算关键实体之间的距离，绕过默认父子/祖孙
碰撞排除；只列明豁免实际法兰、刚接处和电机嵌套轮毂代理。其余实体保持 5 mm
净空，轮胎和躯干允许接地，其他硬件不允许穿地。无解、超限或干涉返回非零退出码，
并保存错误报告；不截断角度、不移动安装点。

当前设计终点四轮离地约 43.7 mm；J1 电机离地仅约 0.49 mm，报告会保留这个
很小的余量。输出 `inspection/sitting_clearance.json` 记录最小间隙、几何对、
对应姿态和模型/参数哈希，`sitting_path.csv` 保存完整采样。`--render` 另输出
站姿侧视图、三维图、关节轴图、转轮 GIF 和坐下路径 GIF。

这些是采样几何检查，不证明接触稳定性、承载能力或真实硬件可执行该动作。
L 类示例仍保留 `allow_hypothetical: true` 和原趋势警告，不修改拟合系数或关闭装配门槛。

## 回归测试

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:MPLBACKEND = 'Agg'
python -X utf8 -m pytest -q
```

覆盖连续旋转与独立力矩映射、内外侧装配、质量归属、解析 FK、四腿对称、站姿
与零位、编辑器重新生成/预览重置、坐下采样及失败诊断；原 humanoid/quadruped
和论文变体质量断言继续作为回归检查。
