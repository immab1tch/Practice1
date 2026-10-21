# Practice1 — 2027 MCM 第一次真题演练（2025 Problem B: Sustainable Tourism, Juneau）

> 团队协作根目录。本文件说明目录结构与约定；开发路线、模型公式、输出规范以 `Docs/to编程手.pdf` 与 `Docs/模型建立.pdf` 为准。

## 目录结构

```
Practice1/
├── README.md                 # 本文件：项目说明与约定
├── Problem/                  # 赛题原文（2025B 图片）
├── Reference/                # 文献库（part1 主文献 / part2 补充）
├── Docs/                     # 建模组文档（问题分析 / 模块选择 / 模型建立 / 数据预处理 / to编程手交接）
├── Prompts/                  # 编程 AI 提示词包（按阶段编号，可直接粘贴给编程 AI）
└── Code/juno_project/        # 代码工程（唯一代码工作区）
    ├── data/                 # 原始数据（juno_raw.xlsx 当前为空，preprocess.py 内置演示数据兜底）
    ├── preprocessed/         # 预处理产物（CSV/MAT/JSON + figures/），勿手工修改，代码直接读取
    ├── results/              # 模型结果输出（m1/m2/m3/m5 子目录，供论文引用）
    ├── preprocess.py         # 已完成的数据预处理（勿重写）
    ├── main.py               # PyCharm 模板占位，可删除
    └── requirements.txt      # 依赖清单（当前为空，由编程手与 AI 补齐）
```

## 约定

1. **路径**：所有脚本使用相对路径（基于 `__file__`），禁止硬编码绝对路径。
   `preprocess.py` 的 `D:\CQU\...` 旧路径已在 2026-10-21 改为 `Path(__file__)` 相对路径（清洗逻辑未动）。
2. **数据**：直接读取 `preprocessed/` 下 CSV（列名→模型符号映射见 `Docs/to编程手.pdf` 第 1 节），不重写预处理。
3. **输出**：按 `Docs/to编程手.pdf` 第 3 节规范，结果表存 CSV/JSON，图存 PNG（300 dpi，中文标签），统一写入 `results/<模型名>/`。
4. **开发顺序**：M1 → M2 → M3 → M5 → M4（后续模型依赖前者输出）。

## 当前进度与待办（2026-10-21 更新）

- [x] 选题 / 文献 / 问题分析 / 模块选择 / 模型建立 v1.0 / 数据预处理
- [x] **M1 基准模型**（MINLP，全枚举 32 组合 + 不动点迭代）— 15/15 情景可解，结果表 + 2 张 Pareto 图已产出
      ⚠ 待建模手确认：S00 优化解 R=3.4543 亿（交接说明期望 ≈3.75 亿），解落在收入下限 R_min=3.45 上
      ⚠ 参数冲突标记：E_max=1.0 超出 CSV eco_threshold(0.6–0.95)、S_min=0.9 超出 social_accept_min(0.4–0.7)
- [x] **M1/M3 v2.0 重跑（2026-10-21 完成）** — 裁定落地：R_min=3.6、E_max=0.95、S_min=0.70、lambda_R=10、
      M1-2'（环境吸引力 γ=0.1）/ M1-4'（C_env≥0.5 地板）；S00 验收通过（R=3.645≥3.6、τ=0.15↑、C_env=0.500、
      x=155.5 万、15/15 可解）；旧结果备份于 results/archive_v1/
      ⚠ 遗留待办：C_env 目标 Sobol 指数数值爆炸（C_env 恒钉地板 0.5、方差≈0，ST=15.9/333 无意义），
      论文需剔除该目标或改用单因素弹性/约束活跃性说明；τ_max 现为弹性最高参数（税收上限是政策瓶颈），需建模手复核
- [ ] **M3 敏感性分析** — 弹性表 ✅（15 情景×11 参数，含 τ_max 与权重）；Sobol ✅（N=128 并行，E_max 弹性=0 符合 v2.0 预期）；仅 C_env 目标数值退化待处理
- [ ] M2 闭环反馈（鲁棒优化 + 蒙特卡洛）
- [ ] M5 空间均衡（MILP）
- [ ] M4 跨目的地迁移（AHP-TOPSIS）
- [x] 补齐 `requirements.txt`（numpy/pandas/matplotlib/scipy/pymoo/SALib/pulp）
- [ ] 确认 `data/juno_raw.xlsx` 是否需真实数据（当前为空，依赖演示数据）

> 2026-10-20 结构重组记录：`Refrence`→`Reference`（拼写修正）；`Model/` 拆分为 `Docs/`（建模文档）与 `Code/`（代码工程）；新增 `Prompts/` 与 `results/`。
