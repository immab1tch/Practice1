"""M2 independent checks: formulas and budget accounting, no model.evaluate calls."""
from pathlib import Path
import json, math
import pandas as pd
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
p=json.loads((ROOT/'inputs/parameters_used.json').read_text());w=json.loads((ROOT/'inputs/weights_used.json').read_text())['weights']
sc=pd.read_csv(OUT/'M2_scenarios.csv');ledger=pd.read_csv(OUT/'M2_budget_ledger.csv')
assert len(sc)==6 and sc.case.is_unique and sc.feasible.all()
checks=0
for r in sc.to_dict('records'):
    q=p.copy()
    if r['case']=='no_maintenance':q['maintenance_multiplier']=0
    if r['case']=='incremental':q['budget_mode']='incremental'
    if r['case']=='no_community_effect':q['eta_C']=0
    if r['case']=='no_investment_effects':q.update(eta_I=0,eta_E=0,eta_C=0)
    if r['case']=='strict_acceptance':q['S_min']=.9
    R=q['R0']*r['x']/q['x0'];B=r['tau']*R
    A=B if q['budget_mode']=='total' else max(B-q['tau0']*q['R0'],0)
    u=[r['delta_'+k]*A for k in 'IEC'];g=[q['eta_'+k]*q['L_'+k]*(1-math.exp(-v/q['K_'+k])) for k,v in zip('IEC',u)]
    peak=q['peak_factor']*r['x']/q['season_days']/q['peak0']
    env=(q['e1']*r['x']/q['x0']+q['e2']*peak*(1-q['power_effect']*r['m3']))*(1-g[1])
    soc=(q['s_h']*(r['x']/q['x0'])**2+q['s_n']*peak*(1-q['traffic_effect']*r['m4'])+q['s_noise']*(1-q['alcohol_effect']*r['m2']))*(1-g[2])
    I=q['I0']*(1+g[0]);S=1/(1+math.exp(-(math.log(q['S0']/(1-q['S0']))-q['acceptance_slope']*(soc-1))))
    infra=q['i_p']*(r['x']/I)**2+q['i_d']*peak
    demand=q['x0']*(r['tau']/q['tau0'])**q['epsilon']*(I/q['I0'])**q['alpha']*(S/q['S0'])**q['beta']*math.prod(1+q['kappa'][i]*r['m'+str(i+1)] for i in range(5))
    penalty=q['penalty_S']*max(q['S_min']-S,0)+q['penalty_R']*max((q['R_min']-R)/q['R0'],0)+q['penalty_H']*max(soc-q['H_max'],0)
    J=w[0]*R/q['R0']-w[1]*env-w[2]*soc-w[3]*infra-penalty
    expected={'R':R,'B':B,'available':A,'I':I,'S':S,'C_env':env,'C_soc':soc,'C_inf':infra,'J':J,'x':demand}
    for k,v in expected.items():assert abs(r[k]-v)<1e-7,(r['case'],k,r[k],v)
    subset=ledger[ledger.case==r['case']]
    assert abs(subset.million_usd.sum()-B*100)<1e-7
    assert abs(subset[subset.category!='baseline_retained'].million_usd.sum()-A*100)<1e-7
    assert r['budget_used']<=q['delta_total']*A+1e-7
    checks+=len(expected)+3
figures=pd.read_csv(OUT/'figure_index_en.csv');assert len(figures)==2
for name in figures.file:
    for ext in ['png','svg']:assert (ROOT/'figures'/f'{name}.{ext}').stat().st_size>1000
result={'status':'passed','M2_independent_formula_and_accounting_checks':checks,'scenarios':6,'figure_pairs':2,'scope':'Checks formulas/accounting and delivered evidence, not empirical calibration or global optimality.'}
(OUT/'independent_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
