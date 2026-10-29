"""Independent delivery audit: saved policies, arithmetic, constraints and figures."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from model import Params,evaluate,vector
root=Path(__file__).resolve().parent;out=root/'results'
p=Params(**json.loads((out/'parameters_used.json').read_text()))
w=np.array(json.loads((out/'weights_used.json').read_text())['weights'])
assert json.loads((root/'config.json').read_text())==json.loads((out/'parameters_used.json').read_text())
count=0
for name in ['recommended','pareto','measure_combinations']:
 frame=pd.read_csv(out/(name+'.csv'))
 for _,row in frame.iterrows():
  r,f,g=evaluate(vector(row),p)
  assert r['feasible'],(name,r['max_violation'])
  for key in ['x','R','S','C_env','C_soc','C_inf','spend_I','spend_E','spend_C']:
   assert np.isclose(row[key],r[key],atol=1e-7,rtol=1e-7),(name,key)
  assert abs(row.J-(-w@f-r['soft_penalty']))<1e-7
  used=sum(row[k] for k in ['spend_I','spend_E','spend_C','measure_cost','maintenance_charged'])
  assert used<=p.delta_total*row.available+1e-7
  count+=1
sc=pd.read_csv(out/'scenarios.csv').set_index('case');b=sc.loc['default'];nm=sc.loc['no_maintenance']
assert nm.J>=b.J-1e-7
# Positive discretionary investment must remain without any maintenance minimum.
assert (nm[['spend_I','spend_E','spend_C']].astype(float)>1e-6).all()
assert abs(nm.maintenance_charged)<1e-10
assert np.allclose(sc.loc['no_investment_effects',['spend_I','spend_E','spend_C']].astype(float),0)
assert abs(sc.loc['no_community_effect','spend_C'])<1e-10
s=pd.read_csv(out/'sensitivity.csv');assert len(s)==48 and s.parameter.nunique()==12
for _,row in s.iterrows():
 q=replace(p,**{row.parameter:getattr(p,row.parameter)*row.factor})
 r,f,g=evaluate(vector(row),q)
 assert bool(row.feasible)==bool(r['feasible'])
 assert abs(row.J-(-w@f-r['soft_penalty']))<1e-7
stress=pd.read_csv(out/'stress_samples.csv');assert len(stress)==600
for _,row in stress.iloc[::24].iterrows():
 q=replace(p,**{k[6:]:row[k] for k in stress.columns if k.startswith('input_')})
 r,_,_=evaluate(vector(row),q);assert bool(row.feasible)==bool(r['feasible'])
index=pd.read_csv(out/'figure_index_en.csv');assert len(index)==15
for name in index.file:
 for ext in ['png','svg']:assert (out/'figures'/f'{name}.{ext}').stat().st_size>1000
assert abs(pd.read_csv(out/'budget_ledger.csv').million_usd.sum()-b.available*100)<1e-6
result=dict(status='PASS',recomputed_feasible_policies=count,sensitivity_rows=48,stress_samples=600,figures=15,no_maintenance_all_discretionary_positive=True)
(out/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
