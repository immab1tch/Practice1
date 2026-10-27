# M2 资金账本、反馈与情景对照（Budget Ledger & Feedback）

接续 M1（v7.0）冻结方程与参数，给出 6 个情景下的税费收入去向、直接投资反馈与消融对照。单位：游客万人、代码金额亿美元、报告支出百万美元/年；综合评分 J 越大越好（higher is better）。

## 运行

在本目录下（Python 3.13，依赖见 `requirements-lock.txt`）：

```bash
python run_m2.py        # 重跑 6 情景优化（每情景 32 组合 × 3 起点）
python render_m2.py     # 从 CSV 重绘英文 300 dpi 图（无需重跑）
python verify_m2.py     # 独立公式与账本校验（不调用 model.evaluate）
```

## 关键结论（推荐方案，147.2 万人）

- 税费收入 **50.780**；新增改善合计 **34.127**（基建 6.087 / 生态 18.576 / 社区 9.464）；措施年化 **5.242**；基础维护 **1.2553**；未配置 **10.156**。
- 支出已用满可分配的 **80%**；其余 20% 是模型保留额，**不是**可继续随意花的预算。
- 仅新增预算情景（Incremental）：总收入 53.314，但新增可用资金仅 12.064，其余 41.250 为原有预算（维持原用途）。
- 推荐均衡下：基建容量 **+11.68%**；生态/社区原始压力分别削减 **51.13% / 33.30%**。
- 严格接受度情景 S=0.8815 仍未达到 0.9 软目标；维护归零仅为机制对照，不代表现实中应取消维护。

## 文件

- `model.py`：冻结的 M1 核心模型（内容哈希见 `inputs/source_manifest.json`）
- `inputs/`：冻结的 M1 输入（推荐解、权重、参数、来源清单、收入核查数据）
- `results/`：
  - `M2_scenarios.csv`：6 情景完整状态
  - `M2_budget_ledger.csv`：税费收入账本（总收入 B 与可用资金 A 双口径闭合）
  - `M2_feedback.csv`：同一均衡下的直接投资反馈
  - `M2_fixed_policy_ablations.csv`：固定政策、只重求均衡的消融
  - `M2_solver_diagnostics.csv`：求解日志
  - `verification.json` / `independent_checks.json`：运行与独立校验
- `figures/`：2 张英文 300 dpi PNG+SVG（索引 `results/figure_index_en.csv`）
  - `M2_01_Budget_breakdown`：税费收入去向与可用资金口径
  - `M2_02_Feedback_decomposition`：直接投资反馈分解

## 声明

响应效率、饱和尺度、部分措施成本为假设，未由实测数据识别；账本数字是模型响应，不是实测减排率或满意率；未声称全局最优。
