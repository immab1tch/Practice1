"""M3 independent checks: sensitivity, stress, rankings and delivered figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
sens=pd.read_csv(OUT/'M3_sensitivity.csv')
assert len(sens)==48 and not sens.duplicated(['parameter','factor','method']).any()
assert sens.groupby(['parameter','method']).size().eq(2).all()
stress=pd.read_csv(OUT/'M3_stress_samples.csv');assert len(stress)==600 and stress['sample'].is_unique
margins=stress[[k for k in stress if k.startswith('margin_')]]
passed=(margins.min(axis=1)>=-1e-7)&(stress.demand_residual.abs()<1e-7)
assert passed.eq(stress.feasible).all() and passed.sum()==245
viol=pd.read_csv(OUT/'M3_constraint_violations.csv')
for r in viol.itertuples():assert r.violations==int((stress['margin_'+r.constraint]<-1e-7).sum())
rank=pd.read_csv(OUT/'M3_sensitivity_ranking.csv');assert rank.parameter.iloc[0]=='R0' and rank.parameter.iloc[1]=='peak_factor'
seeds=pd.read_csv(OUT/'M3_search_stage_gaps.csv');assert np.allclose(seeds.absolute_score_gap,seeds.refined_reference_J-seeds.best_J)
figures=pd.read_csv(OUT/'figure_index_en.csv');assert len(figures)==5
for name in figures.file:
    for ext in ['png','svg']:assert (ROOT/'figures'/f'{name}.{ext}').stat().st_size>1000
result={'status':'passed','M3_sensitivity_rows':48,'M3_stress_rows':600,'M3_stress_feasible':int(passed.sum()),'figure_pairs':5,'scope':'Checks formulas and delivered evidence; scenario coverage is not a real-world success probability.'}
(OUT/'independent_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
