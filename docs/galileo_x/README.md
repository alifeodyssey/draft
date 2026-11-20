# Galileo X 三关节球足概念模型

这是用户指定拓扑的独立概念模型，不是商业 Galileo X 的尺寸复刻。每腿为
`J1 pitch -> J2 局部 yaw -> J3 膝 pitch`，末端暂用球足。

## 运行

在本地 `joint` Conda 环境、Draft 仓库目录运行：

```powershell
conda activate joint
python -X utf8 scripts/generate_robot.py --robot galileo_x
python -X utf8 scripts/robot_tuner.py
```

网页编辑器的机器人列表中选择 `galileo_x`。修改 `thigh_cant_deg`、
`upper_leg_link_length`、`lower_leg_link_length` 或机体尺寸即可重新生成。
`Generate` 更新当前输出目录；`Save` 保存为另一个设计。源配置不被覆盖。
已有设计继续保留原参数；使用 `New from defaults` 加载当前的前后端安装布局。

## 连接与坐标系

采用右手坐标系：`+x` 向前，`+y` 向左，`+z` 向上。

```text
torso / J1 定子
  J1 pitch：转子带动同轴外展连杆和 J2 定子
    J2 yaw：转子带动轴向连接件、斜接大腿和末端 J3 定子
      J3 pitch：转子带动小腿和球足
```

J2 的轴随 J1 运动。大腿的固定斜角放在 J2 之后的刚体变换中，不改变 J2
本身的轴；膝轴随 J2、大腿运动。各轴正方向定义在关节所在的局部刚体中：

| 腿 | J1 | J2 | J3 | 大腿零位朝向 |
|---|---|---|---|---|
| fl 前左 | +y | +z | -y | 前倾 |
| fr 前右 | +y | -z | -y | 前倾 |
| rl 后左 | -y | -z | +y | 后倾 |
| rr 后右 | -y | +z | +y | 后倾 |

四腿相同命令产生对称动作：J1 正转向中部收腿，J2 正转向外展开，J3 负转
屈膝。四腿显式生成，不依赖上游 `mirror_of` 自动改变关节轴。

## 参数含义与来源

| 输入 | 默认值 | 定义及来源 |
|---|---|---|
| `torso_lx / ly / lz` | 0.25 / 0.22 / 0.11 m | 示例机体包络，沿用 Draft 默认量级 |
| `upper_leg_link_length` | 0.22 m | 示例大腿杆长，固定斜接起点至膝轴 |
| `lower_leg_link_length` | 0.22 m | 示例小腿长，膝轴至足心 |
| `thigh_cant_deg` | 30 deg | 示例安装角，相对 J2 转轴方向线；显式转为 rad |
| `assembly_clearance` | 0.005 m | 示例装配净空 |
| `j1_range / j2_range / j3_range` | [-2.5, 2.5] / [-pi/2, pi/2] / [-2.8, 0] rad | 概念验证范围，不代表真实硬件行程 |
| `j1_mot / j2_mot / j3_mot` | L / L / L | 用户指定三电机拓扑；初始选用 Draft 默认 L 类示例 |
| L 类工作点 | 150 N*m, 50 rad/s, reduction 5 | 继承的示例输入，不是关节任务所需扭矩或实际选型 |

电机质量、包络、反射惯量和结构质量由现有 Draft 趋势估算；真实零件尺寸、
定转子各自质量、负载能力和热性能未知。模型编译成功不等于载荷或运动能力经过验证。

外展杆与 J1 同轴，其中心距为 `J1_length/2 + J2_radius + clearance`。
J1 位于躯干沿 x 方向的前后两个端面之外，前后各两个，中心与躯干中心等高。
中心坐标为 `[sx*j1_mount_x, sy*j1_mount_y, 0]`，其中
`j1_mount_x = torso_lx/2 + J1_radius + clearance`，
`j1_mount_y = torso_ly/2 - J1_length/2 - clearance`。
J1 的径向包络与前后端面留出净空，轴向包络收在机体宽度内；J1 转轴仍沿 y。
J2 的轴向连接件长度为
`J2_length/2 + drawn_thigh_radius*abs(sin(cant)) + clearance`，使斜杆边缘避开
J2 输出端面。所有这些都是包络派生的概念尺寸。

大腿杆长不等于 J2 到膝的直线距离：后者还包括轴向连接件，由
`thigh_joint_span` 保存，并用于结构质量审计。根高度由斜角、连接件、杆长和足球
半径派生，零位四个足球与地面相切。固定安装角和动态关节角分别保存。

每个电机整体包络只计一次质量，并归属于上游刚体，这是 Draft 的简化约定；
没有声称已拆分真实定子和转子质量。膝电机位于大腿末端，不使用原 quadruped
的髋部膝电机和无质量皮带滑轮表示。

## 可行性与验证边界

默认 L 类示例超过目录趋势边界，因此显式保留 `allow_hypothetical: true` 及
报告警告。电机相互装配和躯干重叠检查保持严格，无全局豁免。
示例机体较小，报告也可能提示电机/躯干质量比例和结构有效密度异常；
这些提示用于后续设计审查，不通过修改拟合系数来消除。

生成目录包含 `galileo_x.xml`、`scene.xml`、`parameters_resolved.yaml`、
`feasibility_report.yaml` 和躯干网格/惯性文件。MJCF 沿用上游约定，仅保存关节
力矩和速度能力，不添加控制器的 `<actuator>`。

自动验证覆盖独立解析 FK、J2 圆周轨迹、膝中心不动、四腿命令对称、足球高度、
前后端安装净空、左右包络收纳、结构轴间距、质量归属和编辑器修改后再生成。
修改机体长宽后会重新派生安装中心。原 humanoid/quadruped 和论文变体
的质量基线继续作为回归检查。本轮不包含轮轴、RL、优化器或 CAD 导出。

Windows 上运行完整快速测试时，让趋势重拟合测试启动的 Python 子进程也继承
UTF-8 模式（单独的 `-X utf8` 仅作用于当前解释器）：

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
python -X utf8 -m pytest -q
```
