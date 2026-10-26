# Practice1 — 2027 MCM 第一次真题演练（2025 Problem B: Sustainable Tourism, Juneau）

> ## ⚠ 当前基准版本：**v6.1**（2026-10-23 定稿）
> **唯一权威模型文档：`Docs/模型建立6.1.pdf` + `Docs/模型建立6.1（补充）.pdf`**
> 权威参数表与演进历史：`Docs/CHANGELOG.md`（唯一参数源 single source of truth）
> 模型版本迭代说明见 `Docs/迭代记录.md`；旧版模型文档与参数已从工作区删除（完整历史见 git log）。
> 所有代码、图表、论文写作必须以此版为准，禁止引用旧版参数。

## 目录结构

```
Practice1/
├── README.md                 # 本文件：项目说明与约定
├── Problem/                  # 赛题原文（2025B）
├── Reference/                # 文献库（part1 主文献 / part2 补充 / O奖范文）
├── Docs/                     # 建模组文档（模型建立6.1 + 补充手册 / CHANGELOG / 迭代记录）
├── Prompts/                  # 编程 AI 提示词包（按阶段编号，可直接粘贴给编程 AI）
├── Code/juno_project/        # 代码工程（唯一代码工作区）
│   ├── data/                 # 原始数据（juno_raw.xlsx，4 个 sheet，真实数据）
│   ├── preprocessed/         # 预处理产物（CSV/MAT/JSON + figures/ EDA 图），代码直接读取
│   ├── results/              # 模型结果输出，按模型分类（见下）
│   ├── m1_solver.py          # M1 基准模型（MINLP 全枚举 + 不动点）
│   ├── m2_robust.py          # M2 鲁棒优化（蒙特卡洛反哺）
│   ├── m3_sensitivity.py     # M3 单因素敏感性分析
│   ├── m5_spatial.py         # M5 空间均衡（MILP，PuLP）
│   ├── figures_paper.py      # 8 张论文图生成脚本（fig01-fig08，英文版）
│   ├── preprocess.py         # 数据预处理（勿重写）
│   └── requirements.txt      # 依赖清单
└── main.pdf                  # 竞赛论文（论文手维护）
```

**results/ 结果分类（论文手直接按模型引用）**

```
results/
├── figures/                  # 跨模型总览图
│   ├── fig01_background.png  #   背景数据总览（乘客量/支出/居民态度）
│   ├── fig02_architecture.png#   模型架构图（M1-M5 联动 + M3 反馈回路）
│   ├── fig03_calibration.png #   历史数据 vs 模型 S00 校准
│   └── run_log.txt
├── m1/                       # M1 基准模型（税+投资配额枚举优化）
│   ├── fig04_pareto.png      #   Pareto 前沿（英文版）
│   ├── m1_results.csv        #   160 组合 × 全部指标
│   └── run_log.txt
├── m2/                       # M2 鲁棒反哺（y 投资组合选择）
│   ├── fig06_m2_robust.png   #   8 组合 z_worst + 最优 y 蒙卡分布
│   ├── m2_projects.csv / m2_scenarios.csv
│   └── run_log.txt
├── m3/                       # M3 单因素敏感性
│   ├── fig05_tau_lever.png   #   τ 杠杆曲线（J/R 双轴 + R_min 约束线）
│   ├── fig07_m3_elasticity.png#  弹性排序 + 15 情景
│   ├── m3_elasticity.csv / m3_elasticity_R.csv / m3_scenarios.csv / tau_lever_v61.csv
│   └── run_log.txt
└── m5/                       # M5 空间均衡（MILP）
    ├── fig08_m5_flow.png     #   热/冷 5 节点流量分配
    ├── m5_region_allocation_v61.csv / m5_objectives_v61.csv
    └── run_log.txt
```

## 约定

1. **参数唯一源**：一切公式/参数以 `Docs/CHANGELOG.md` 为准；改参数必须同步更新 CHANGELOG，禁止各文件重复维护参数表。
2. **结果实时落位**：每次跑完模型，结果表与图**直接写入 `results/<模型名>/`**（英文图），**禁止打包 zip 提交仓库**；交付包另行分发，不入库。
3. **论文用图仅限英文版 fig01-fig08**；旧中文/旧版本图已全部删除，如发现残留请报告。
4. **数据**：直接读取 `preprocessed/` 下 CSV；原始数据在 `data/juno_raw.xlsx`（已重建，4 sheets）。
5. **开发顺序**：M1 → M2 → M5 → M3 → M4（M3 敏感性依赖 M1 最优解）。

## 当前状态（2026-10-26）

- [x] **M1 基准模型（v6.1）** — 153/160 可行；最优 m=11111, n_s=5, τ=0.1483, δ=(0.8,0,0), J=0.0545
      ⚠ 待建模手确认：最优解 R=3.6000 贴 R_min 下限（"生态优先牺牲经济"政策含义）；
      C_env 地板 0.4 且 147/148 可行解 C_env=0.4（生态约束非紧，Pareto 横轴退化，见 fig04 说明）
- [x] **M2 鲁棒反哺（v6.1）** — 最优 y=(1,0,0)（只投基建），z_worst=1.1100
- [x] **M5 空间均衡（v6.1）** — Σ=160 万守恒 0% 误差；热 τ=0.12 / 冷 τ=0.15；冷点扩容 v=1
- [x] **M3 敏感性（v6.1 重跑）** — |S_J|>1 共 4 个（w_R=+7.22, w_S=-3.58, w_E=-2.20, τ=+1.34）
      ⚠ τ=0.15 处 R_min 硬约束激活不可行（模型真实结构，非 bug）；τ 杠杆最优 τ*=0.145（固定 δ 口径）
- [x] **8 张论文图（英文版，300dpi）** — fig01-fig08 全部生成并归位到各模型目录
- [x] **juno_raw.xlsx 重建** — 4 sheets 真实数据（2014-2023 乘客量 / 2023 支出 / 2023 调查 / 模型基准）
- [ ] **M4 跨目的地迁移（AHP-TOPSIS）** — 公式已在 模型建立6.1（补充）.pdf，待编程实现
- [ ] **论文 main.pdf** — 仅覆盖 M1，M2/M3/M4/M5 章节缺失；M1-5/M1-6 参数仍为 v6.0 旧值 2844/1137，需核对为 1422.72/0.6

> 历史：2026-10-20 结构重组（Refrence→Reference、Model/ 拆分 Docs/ 与 Code/、新增 Prompts/）。
> 2026-10-26 结果目录按模型分类（figures_upgrade 并入 m1/m2/m3/m5/figures），删除全部旧中文图与旧版产物。
