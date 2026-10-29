"""Re-evaluate frozen-model policies and check the claimed threshold diagnosis."""
import json,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from model_snapshot import Params,evaluate,vector
root=Path(__file__).resolve().parent;out=root/'results'
p=Params(**json.loads((root/'inputs/parameters_used.json').read_text()))
w=np.array(json.loads((root/'inputs/weights_used.json').read_text())['weights'])
assert hashlib.sha256((root/'model_snapshot.py').read_bytes()).hexdigest()==(root/'inputs/original_model.sha256').read_text().strip()
count=0;max_residual=0.
for name in ['counterfactuals','floor_sweep','penalty_sweep','visitor_profile','independent_optimizer','all_combo_solutions']:
 df=pd.read_csv(out/(name+'.csv'))
 for _,row in df.iterrows():
  q=replace(p,R_min=row.R_min,penalty_R=row.penalty_R)
  r,f,g=evaluate(vector(row),q)
  assert r['feasible'],(name,row.get('case'),r['max_violation'])
  for key in ['x','R','S','C_env','C_soc','C_inf']:
   assert np.isclose(row[key],r[key],atol=2e-7,rtol=1e-8),(name,key)
  gross=-w@f;income_pen=q.penalty_R*max(0,q.R_min-r['R'])/q.R0
  assert abs(row.J-(gross-r['soft_penalty']))<1e-7
  assert abs(row.income_penalty-income_pen)<1e-7
  if name=='visitor_profile':assert abs(row.x-row.target_x)<1e-6
  max_residual=max(max_residual,abs(r['demand_residual']));count+=1
cf=pd.read_csv(out/'counterfactuals.csv').set_index('case')
assert abs(cf.loc['default','x']-p.R_min*p.x0/p.R0)<1e-6
assert cf.loc['income_penalty_removed','x']<cf.loc['default','x']-1
assert abs(cf.loc['income_penalty_removed','J']-cf.loc['income_floor_removed','J'])<1e-7
assert abs(cf.loc['income_penalty_removed','x']-cf.loc['income_floor_removed','x'])<1e-4
floor=pd.read_csv(out/'floor_sweep.csv');assert len(floor)==27
for rmin in [3.4,3.425,3.44,3.45,3.46,3.475,3.5]:
 r=floor.loc[np.isclose(floor.R_min,rmin)].iloc[0]
 assert abs(r.x-rmin*p.x0/p.R0)<1e-5
assert (floor.x<=160+1e-6).all()
pen=pd.read_csv(out/'penalty_sweep.csv');assert len(pen)==16
# Numerical maximizers must obey weak monotonicity in increasing penalty on income shortfall.
assert (np.diff(pen.sort_values('penalty_R').income_gap)<=1e-6).all()
profile=pd.read_csv(out/'visitor_profile.csv');assert len(profile)==14
peak=profile.loc[profile.J.idxmax()]
assert abs(peak.target_x-147.2)<1e-8
assert abs(pd.read_csv(out/'score_decomposition_en.csv').delta_J.sum()-json.loads((out/'findings.json').read_text())['score_decomposition_total'])<1e-9
for name in pd.read_csv(out/'figure_index_en.csv').name:
 for ext in ['png','svg']:assert (out/'figures'/f'{name}.{ext}').stat().st_size>1000
result=dict(status='PASS',policy_rows_recomputed=count,largest_demand_residual=max_residual,threshold_points=27,penalty_points=16,profile_points=14,frozen_model_verified=True,global_optimality_proven=False)
(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
