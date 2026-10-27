# HANDOFF — 交接必读（给接手的 AI Agent / 编程手）

> 本文件是项目**唯一入口**。开始任何工作前，必须先读完本文件。
> 题目：**2025 MCM Problem B — Managing Sustainable Tourism in Juneau（朱诺可持续旅游管理）**
> 整理日期：2026-10-26（v7.0 权威版）；2026-10-27 新增 M2、M3 自包含模块与英文图（原混杂整合包 `Model2_Model3_整合` 已拆分删除）。本目录已剔除全部旧版本，只保留当前权威成果。

---

## 0. 最高优先级铁律（违反即出错）

1. **唯一权威模型是《Docs/模型建立_v7.0.pdf》**（原文件名"模型建立改版3.0"）。它推翻了此前所有版本（v6.0 / v6.1 / 改版前的 v1–v6）。
2. **参数一律以代码 `Code/M1_equilibrium/config.json` 为准**，不是 PDF 表 2（PDF 有 3 处笔误，见本文第 5 节）。
3. **`archive/` 目录里全部是已作废的旧版本，禁止引用其中任何参数、公式、结果、图、文字**。它只用于追溯，不得作为建模或写论文依据。
4. 项目里曾出现"v3.0"两个含义（旧 CHANGELOG 的 v3.0 与本次"改版3.0"）。现统一把当前权威版命名为 **v7.0**（接续 v6.1），**不要再使用"改版3.0"这个会撞号的名字**。
5. 不同模型版本、不同压力函数下的指数**不能直接相减算"改善率"**；不同权重下的目标值 J **不能跨权重比较**。
6. 所有假设参数（K、L、η、S0、维护预留、成本系数等）是**情景假设，不是实测拟合**；产出与论文必须如实标注，不得包装成回归结果。

---

## 1. 目录结构（按 problem → reference → 分析建模 → 代码/数据/图 的逻辑组织）

```
Practice1_FINAL/
├── HANDOFF_交接必读.md            ← 你正在读（唯一入口）
├── README.md                      ← 项目总导航
├── .gitignore
├── Problem/                       ← 1. 题目原文（2 张图）
├── Reference/                     ← 2. 参考文献
│   ├── data_and_reports/          ←    题面数据、McKinley 报告、居民调查、FY24 预算
│   ├── methods/                   ←    方法学文献（STMO、AHP、sustainability）
│   └── award_papers_B2025/        ←    6 篇 2025 B 题优秀论文（图表/排版标杆）
├── Docs/                          ← 3. 分析与建模文档
│   ├── 问题分析.pdf               ←    题面解析（数据口径，有效）
│   ├── 模型建立_v7.0.pdf          ←    ★权威模型（M1–M5 公式定稿）
│   ├── 模型建立_修改点清单.pdf     ←    本版相对旧版的 23 条修改
│   ├── 权重向量与偏好空间讨论.md   ←    M3 Minimax Regret 设计
│   ├── 权威参数表_v7.0.md         ←    ★唯一参数源（含 PDF 勘误）
│   └── CHANGELOG_模型迭代记录.md  ←    版本演进与"旧版为何作废"
├── Code/                          ← 4. 代码（按模型分目录，代码+数据+图一体，各自自包含可运行）
│   ├── data_raw/juno_raw.xlsx     ←    真实原始数据底稿（4 个 sheet，非空）
│   ├── M1_equilibrium/            ←    M1 闭环均衡（已完成，可运行）
│   ├── M1_revenue_audit/          ←    M1 收入下限专项核查（已完成）
│   ├── M2_budget/                 ←    M2 资金账本/反馈/情景（已完成，英文图）
│   └── M3_robustness/             ←    M3 敏感性/压力/偏好Minimax（已完成，英文图）
├── Paper/                         ← 5. 论文（现状说明与写作要点）
└── archive/                       ← 【作废勿用】旧 v6.x 文档/代码/论文/历史 prompt
```

---

## 2. 题目（题面口径，不得改动）

朱诺（Juneau, Alaska）：常住人口约 **3 万**；2023 年邮轮旅客 **160 万人次**；最繁忙日停靠 **7 艘**大型邮轮、日游客约 **2 万**；旅游收入约 **3.75 亿美元**；门登霍尔冰川自 2007 年退缩约 8 个足球场。
五个任务：① 建可持续旅游模型（游客量/收入/措施，明确优化目标与约束，支出反馈，敏感性分析）；② 迁移到另一目的地；③ 给旅游委员会的一页备忘录（memo）；全文 ≤25 页；含 Summary Sheet、目录、References、AI Use Report。

> 参考 PDF 中的 167 万乘客、4.9 亿总支出、3850 就业仅作**校准补充**，主模型严格用题面 160 万 / 3.75 亿，不得混口径。

---

## 3. 当前权威模型 v7.0（与旧版的根本区别）

- 游客量 **x 不再是自由决策变量**，而由需求方程（M1-3）内生均衡求解（Brent 法）。
- 日峰值是派生量：**xd = x / 80**（= 2.675·x / 214），不再是独立决策变量。
- 投资是**边际递减饱和函数** g_k = η_k L_k (1 − e^(−u_k/K_k))，不再是可把压力抵成负数的线性抵扣。
- 居民接受度 S 用 **logit-sigmoid 平滑**（γ=2，S0=0.7）；营销措施 m5 不再"免费"加 S，杜绝同一笔投入双重计费。
- 预算为**总预算制**：先扣基础维护预留（0.6419 / 0.6134 百万美元），再分三类新增投入，支出 ≤ 0.8B。
- 约束：**14 个硬约束 + 3 个软约束**。R_min=3.45、S_min=0.70、社会压力 H_max=1.0 均为**软约束**（罚项）；生态 E_max=1.0 等为硬约束。

### 推荐解（M1 主情景，已复现验证）

| 指标 | 值 |
|---|---|
| 年游客量 x | **147.2 万人** |
| 日峰值 xd | 1.84 万人/日 |
| 旅游收入 R | **3.45 亿美元** |
| 综合税费 τ | **14.7188%** |
| 措施组合 m=(m1..m5) | **(0,1,1,1,0)**（限流0、酒类1、岸电1、交通1、营销0） |
| 新增 基建/生态/社区 投入 | 6.087 / 18.576 / 9.464 百万美元/年 |
| 居民接受度 S | 0.8639 |
| 环境/社会/基建压力 C_env/C_soc/C_inf | 0.3822 / 0.4995 / 0.7752 |
| 预算 | B=50.780 百万、0.8B=40.624 百万，使用率 100%（绑定） |
| 最大硬约束违反 / 需求残差 | 6.86e-15 / 2.84e-14 |
| 权重 w=(w_R,w_E,w_S,w_I) | 0.3165168 / 0.2750766 / 0.2042066 / 0.2042000 |

> **重要**：147.2 万是"收入软下限 R_min=3.45 亿"作用下的**折点解**，不是"客观最佳承载量"。取消收入罚项后约 127.69 万 / 2.99 亿。论证见 `Code/M1_revenue_audit/results/收入下限专项核查报告.md`。

### 各模块在 v7.0 中的定义（与旧版完全不同，勿混淆）

| 模块 | v7.0 定义 | 状态 |
|---|---|---|
| M1 | 闭环均衡：政策→资金→三类投资→容量/压力/接受度→需求均衡→预算 | ✅ 已完成 |
| M2 | **资金账本核算**（总预算制、维护预留、措施成本、未配置收入）+ 6 情景对照 + 直接投资反馈 | ✅ 已完成（`Code/M2_budget/`，英文图） |
| M3 | 12 参数 ±10% + 600 压力情景 + 政策地形 + 收入核查 + **权重空间 Minimax Regret（18 点）** | ✅ 已完成（`Code/M3_robustness/`，英文图） |
| M4 | **威尼斯迁移**：7 维特征 AHP-熵权 TOPSIS（半定量） | ❌ 无代码 |
| M5 | **4 节点引力模型**（冷点投资 500 万、压力 0.95→0.80、转移矩阵），不是 5 节点 MILP | ❌ 无代码 |

---

## 4. 如何运行（Windows / Python 3.13）

Python 解释器：`C:\Users\Eternal Sunshine\AppData\Local\Programs\Python\Python313\python.exe`
依赖：numpy、pandas、scipy、matplotlib、pymoo、openpyxl（见各目录 requirements.txt）。
代码路径全部基于脚本自身（`Path(__file__).parent`），**整体移动目录不影响运行**。

```powershell
$py = "C:\Users\Eternal Sunshine\AppData\Local\Programs\Python\Python313\python.exe"
# M1 主实验（耗时长：NSGA-II 3 种子×180×160 + SLSQP 精修）
& $py Code\M1_equilibrium\run_round2.py
# M1 单元测试
& $py -m unittest discover -s Code\M1_equilibrium -v
# M1 结果校验（应全部 PASS）
& $py Code\M1_equilibrium\verify_results.py
# 从已有 CSV 重画图（不需重跑）
& $py Code\M1_equilibrium\redraw.py
# 收入下限专项核查
& $py Code\M1_revenue_audit\audit.py
& $py Code\M1_revenue_audit\verify.py
# M2 资金账本（重跑 / 英文图 / 校验）
& $py Code\M2_budget\run_m2.py
& $py Code\M2_budget\render_m2.py
& $py Code\M2_budget\verify_m2.py
# M3 敏感性/压力（重跑 / 英文图 / 校验）
& $py Code\M3_robustness\run_m3.py
& $py Code\M3_robustness\render_m3.py
& $py Code\M3_robustness\verify_m3.py
# M3 偏好 Minimax（耗时长，可 --resume；英文图 / 校验）
& $py Code\M3_robustness\run_preferences.py
& $py Code\M3_robustness\render_preferences.py
& $py Code\M3_robustness\verify_preferences.py
```

已复现结果：M1 `validation.json` = PASS（479 可行政策、48 敏感性行、600 蒙卡、15 图）；收入核查 `verification.json` = PASS（902 政策回代、最大残差 2.56e-10，**未声明全局最优**）。
M2 `independent_checks.json` = PASS（78 项公式/账本检查、6 情景、2 英文图）；M3 `independent_checks.json` = PASS（48 敏感性、600 压力、245 可行、5 英文图）；M3 偏好 `verification.json` = PASS（18 权重点、20 政策、360 后悔单元、455 菜单点、600 压力、顶点凸性已验证）。

---

## 5. 《模型建立_v7.0.pdf》已知 3 处笔误（以代码为准，建模手待修 PDF）

1. **表 2 维护预留单位错误（差 100 倍）**：PDF 写 m_I=0.6419、m_C=0.6134「亿美元」；实际为 **0.006419 / 0.006134 亿美元 = 0.6419 / 0.6134 百万美元**（config：`maintenance_I=0.006419`、`maintenance_C=0.006134`；修改点清单第 11 条正确）。
2. **表 2 S_min 错误**：PDF 写「S_min=0.90」，主情景实际 **S_min=0.70**（config `S_min=0.7`）；0.90 仅为"严格情景"。若误用 0.90，推荐解 S=0.8639 会触发社会罚项，结论改变。
3. **页数不符**：修改点清单第 23 条称成品 33 页，当前 PDF 仅 11 页（缺正文/图表/附录的完整论文版可能未提交），需建模手确认。

---

## 6. 下一步任务（按优先级）

1. **M4**：威尼斯 7 维 AHP-熵权 TOPSIS（v7.0 PDF 第 4 章已给威尼斯 7 维打分表与判断矩阵）。
2. **M5**：4 节点引力模型 x_i = A_i(1+τ_i)^(−ε)·x/Σ，冷点投资 500 万、压力 0.95→0.80，给冷热转移矩阵；税率离散化（防双决策变量相乘）按补充文档细则。
3. **M1 英文论文图**：M2/M3 共 12 张图已是英文 300 dpi；但 **M1 的 15+4 张图仍为中文**（`Code/M1_equilibrium/results/figures`、`Code/M1_revenue_audit/results/figures`），需改绘图脚本中文字符串为英文后用 redraw 重画（CSV 已在，无需重跑 NSGA-II），统一 300 dpi、与 `Reference/award_papers_B2025/` 风格对齐。
4. **论文**：`archive/旧论文_main_v61.pdf` 是旧版（仅写到 M1，且用 3.6 亿硬下限、xd 独立等旧设定），须按 v7.0 重写 M1 并补 M2–M5。M2/M3 的英文图表与核心结论已就绪（账本、40.83% 可行率、R0/peak_factor 最敏感、Minimax 0.118102→0.110780），详见 `Paper/论文现状与写作要点.md` 与各模块 README。

> 每个新模块完成后，必须像 M1 一样提供：可复现脚本 + 独立 verify 回代 + 关键 CSV + 英文图（PNG+SVG）+ 诚实的边界声明（不夸大、不把假设写成拟合）。

---

## 7. 数据说明

- `Code/data_raw/juno_raw.xlsx`：**真实原始数据**（4 sheet：历年客流、2023 消费、居民调查、题面基准），非空。
- v7.0 的 M1 只依赖题面参数 + `config.json`，**不读取旧 preprocessed 目录**（旧的 S01–S14 情景、旧参数区间、旧 5 区域数据均已随旧代码归档）。
- 新增官方来源（FY24 预算、岸电、Crossing Guard）见 `Reference/data_and_reports/FY24及新增来源说明.md`。
