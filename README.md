# Practice1 — 2025 MCM Problem B：Managing Sustainable Tourism in Juneau

> 2027 美赛备赛真题演练（2025 B 题）。**当前权威模型版本：v7.0**。
> 新接手者请先读 👉 **[HANDOFF_交接必读.md](HANDOFF_交接必读.md)**（含防幻觉铁律、权威文件、待办与运行方式）。

## 快速导航

| 你要做什么 | 看哪里 |
|---|---|
| 了解项目现状 / 下一步 / 运行代码 | [HANDOFF_交接必读.md](HANDOFF_交接必读.md) |
| 读题目原文 | [Problem/](Problem/) |
| 查参考文献 / 优秀论文 | [Reference/](Reference/)（索引见 [00_文献清单.md](Reference/00_文献清单.md)） |
| 读模型与问题分析 | [Docs/](Docs/)：模型建立_v7.0.pdf（★权威）、问题分析.pdf、修改点清单、权重讨论 |
| 查参数（唯一参数源） | [Docs/权威参数表_v7.0.md](Docs/权威参数表_v7.0.md) |
| 看版本演进 / 旧版为何作废 | [Docs/CHANGELOG_模型迭代记录.md](Docs/CHANGELOG_模型迭代记录.md) |
| 跑 M1 / 看结果 | [Code/M1_equilibrium/](Code/M1_equilibrium/)（代码+数据+图一体） |
| 收入下限核查 | [Code/M1_revenue_audit/](Code/M1_revenue_audit/) |
| 跑 M2 资金账本 / 反馈 / 情景 | [Code/M2_budget/](Code/M2_budget/)（英文图，自包含） |
| 跑 M3 敏感性 / 压力 / 偏好稳健 | [Code/M3_robustness/](Code/M3_robustness/)（英文图，自包含） |
| 跑 M4 威尼斯迁移 | [Code/M4_migration/](Code/M4_migration/)（AHP-熵权 TOPSIS，英文图，自包含） |
| 跑 M5 空间均衡 | [Code/M5_spatial/](Code/M5_spatial/)（4 节点引力/最大流，英文图，自包含） |
| 原始数据 | [Code/data_raw/juno_raw.xlsx](Code/data_raw/) |
| 写论文（本次不写） | [Paper/论文现状与写作要点.md](Paper/论文现状与写作要点.md) |
| 旧版本（作废勿用） | [archive/](archive/) |

## 当前进度

- ✅ **M1 闭环均衡**：需求内生均衡 + 饱和投资 + sigmoid 接受度 + 资金账本，已通过验证（推荐 147.2 万人 / τ=14.72% / m=01110）。
- ✅ **M1 收入下限专项核查**：确认 147.2 为软下限折点解。
- ✅ **M2 资金账本**：6 情景 + 总收入/可用资金双口径闭合账本 + 直接反馈（英文图 2 张）。
- ✅ **M3 敏感性/压力/偏好稳健**：48 敏感性、600 压力（245 全过）、18 权重 Minimax Regret（最大后悔 0.118102→0.110780，英文图 10 张）。
- ✅ **M4 威尼斯迁移**：7 维特征 AHP-熵权 TOPSIS（CR=0.0134）+ 阈值重校准，52 项独立检查（英文图 4 张）。
- ✅ **M5 4 节点空间均衡**：引力分配 + 500 万冷点投资，冰川压力 0.95→0.797；最大流割集校验；冷热转移矩阵；41 项独立检查（英文图 4 张）。
- ✅ 全部模型（M1–M5）图均已为英文 300 dpi；**美赛论文按用户要求本次不写**。

## 运行环境

- Windows / Python 3.13（`C:\Users\Eternal Sunshine\AppData\Local\Programs\Python\Python313\python.exe`）；依赖 numpy、pandas、scipy、matplotlib、pymoo、openpyxl。
- 代码路径基于脚本自身，整体移动目录不影响运行；命令见 HANDOFF 第 4 节。

## 重要约定

- 参数以 `Code/M1_equilibrium/config.json` 与《权威参数表_v7.0.md》为准，**PDF 表 2 有 3 处笔误**（见 HANDOFF 第 5 节）。
- `archive/` 为作废旧版，禁止引用其中任何数值。
- 不同权重下的 J 不可跨权重比较；假设参数须如实标注，不得写成拟合结果。
