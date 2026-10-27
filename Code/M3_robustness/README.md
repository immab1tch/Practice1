# M3 敏感性、压力测试与偏好稳健（Sensitivity / Stress / Minimax Regret）

接续 M1（v7.0）冻结方程与参数，包含两部分：

1. **基础 M3**：12 参数 ±10% 敏感性、600 个假设压力情景、5 个权重偏好、政策地形、资金转移、收入软目标核查。
2. **偏好空间 Minimax Regret**：18 组权重（15 结构点 + MAIN/AHP/ENT 3 命名点）扫描、政策菜单、跨权重后悔矩阵、菜单 Minimax 选择与连续稳健精修。

## 运行

在本目录下（Python 3.13，依赖见 `requirements-lock.txt`）：

```bash
python run_m3.py                          # 重跑基础 M3（复用 M1 决策，重新求值均衡）
python render_m3.py                        # 从 CSV 重绘 5 张基础英文图
python verify_m3.py                        # 独立校验基础 M3
python run_m3.py --with-preferences        # 再跑偏好 Minimax（18 权重优化，耗时较长）
python run_preferences.py --resume         # 仅偏好模块：中断后按哈希续跑
python render_preferences.py               # 重绘 5 张偏好英文图
python verify_preferences.py               # 独立校验偏好 Minimax
```

## 关键结论

- **敏感性**：收入基准 R0（校准量）的评分变化最大，其次是旺季集中程度（peak_factor）；该排序是指定扰动范围内的响应，不是因果重要性，也不是 Sobol 指数。
- **600 压力情景**：全部硬约束通过 **245/600（40.83%）**；预算违反 51.50%，日峰值与船舶运力各违反 12.50%（二者上限相同，不可相加）。占比是情景覆盖，**不是现实成功概率**。
- **偏好 Minimax**：18 权重点中 13/18 游客量约 147.2 万，范围 147.2–160 万。原固定权重方案最大后悔值 **0.118102**，连续稳健方案 **0.110780**（下降 6.20%）；两者措施组合均为 **01110**（日限流、酒类管控、岸电、交通管理、营销）。稳健方案为备选，**未覆盖原推荐**。
- 最坏偏好为 **VE / VS / VI**（环境、社会、基建顶点）；收入仍为软目标。
- **偏好稳健不等于参数稳健**：稳健方案在 600 参数扰动复测中仍为 245/600，本轮 Minimax 不优化参数风险。

## 文件

- `model.py`：冻结的 M1 核心模型（内容哈希见 `inputs/source_manifest.json`）
- `inputs/`：冻结 M1 输入，含 `preference_robustness/`（AHP/熵权来源）
- `results/`：
  - `M3_sensitivity.csv` / `M3_sensitivity_ranking.csv`：48 条敏感性与排序
  - `M3_stress_samples.csv` / `M3_constraint_violations.csv`：600 压力情景与违约统计
  - `M3_preference_scenarios.csv`、`M3_policy_landscape.csv`、`M3_marginal_transfers.csv`、`M3_search_stage_gaps.csv`
  - `M3_income_*.csv`：收入软目标核查（复用 M1 专项核查）
  - `preference_robustness/`：偏好 Minimax 的 `weights / policy_menu / score_matrix / regret_matrix / regret_ranking / summary / verification` 等，及 `checkpoints/`（18 权重求解快照）
- `figures/`：5 张基础英文 300 dpi PNG+SVG；`figures/preference_robustness/`：5 张偏好英文图
- 中文分析底稿：`results/M2_M3_简版结果_中文底稿.md`、`results/preference_robustness/偏好空间与Minimax_Regret_结果说明_中文底稿.md`（论文图以英文目录为准）

## 声明

所有最优基准均为数值搜索找到的最好值，**未证明全局最优**；后悔值在同一偏好列内作差，不直接比较不同权重的总分；13/18 是已选点占比，不是委员会偏好概率。
