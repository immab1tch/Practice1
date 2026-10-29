# 147.2万人与收入下限：专项核查

这是针对第二轮结论的独立诊断，不覆盖第一、第二轮源码或结果。冻结原模型快照、参数和权重，仅变动收入软下限或缺口罚系数。阅读 `results/收入下限专项核查报告.md`。

当前电脑在工作区目录运行：

```bash
Model1_v1_实现/.venv/bin/python Model1_收入下限专项核查/audit.py
Model1_v1_实现/.venv/bin/python Model1_收入下限专项核查/verify.py
```

换电脑可使用与前两轮相同的依赖（requirements.txt），Python 3.9+。单独重绘：

```bash
python report_en.py
```

实验：27个收入目标、16个罚系数、14个固定游客量剖面点；措施组合枚举加多起点局部优化；两组固定优胜措施分支的差分进化交叉核验。只检查这个问题，不属于新的全模型升级。

收入是R=(3.75/160)x，仍是题面口径的经济活动代理，不是政府收入。R_min为软目标，而非收入硬约束。消除收入罚项后，经济收益仍保留在四目标评分内。

输出：counterfactuals.csv（取消罚项/取消下限）、floor_sweep.csv、penalty_sweep.csv、visitor_profile.csv、independent_optimizer.csv、all_combo_solutions.csv、solver_diagnostics.csv、findings.json、verification.json。图表含PNG与SVG；所有重要数字可在CSV复核。

结果仅对当前结构和假设成立。有限多起点搜索与独立算法对照不构成全局最优证明。取消保障后的解不等于实际应采用的方案，也不能代替对居民生计和就业的保障目标。
