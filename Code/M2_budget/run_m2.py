"""M2: budget ledger, feedback and scenario comparisons, frozen round-2 equations."""
from pathlib import Path
from dataclasses import replace
import argparse, json, hashlib, platform
import numpy as np
import pandas as pd
from model import Params, validate, evaluate, refine, vector

ROOT=Path(__file__).resolve().parent
IN=ROOT/'inputs'; OUT=ROOT/'results'
P=Params(**json.loads((IN/'parameters_used.json').read_text()))
W=np.asarray(json.loads((IN/'weights_used.json').read_text())['weights'])
BEST=pd.read_csv(IN/'recommended.csv').iloc[0].to_dict()
CHANGES={'default':{},'no_maintenance':{'maintenance_multiplier':0},'incremental':{'budget_mode':'incremental'},'no_community_effect':{'eta_C':0},'no_investment_effects':{'eta_I':0,'eta_E':0,'eta_C':0},'strict_acceptance':{'S_min':.9}}

def save(rows,name):
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')

def score(v,p,w=W):
    r,f,g=evaluate(v,p);r['J']=float(-np.asarray(w)@f-r['soft_penalty']);return r

def optimize(q,old):
    starts=[vector(BEST),vector(old),np.array([.14,.2,.25,.2])]
    candidates,logs=refine(q,W,[v[:4] for v in starts])
    for v in starts[:2]:
        r=score(v,q)
        if r['feasible']:candidates.append(r)
    if not candidates:raise RuntimeError('No feasible candidate found; this does not prove infeasibility.')
    best=max(candidates,key=lambda r:r['J']);v=vector(best)
    for i,k in enumerate('IEC',1):
        if getattr(q,'eta_'+k)==0:v[i]=0
    canonical=score(v,q)
    if canonical['feasible'] and canonical['J']>=best['J']-1e-9:best=canonical
    return best,logs

def m2(reuse=False):
    old=pd.read_csv(IN/'scenarios.csv').set_index('case')
    rows=[]; logs=[]; ledgers=[]; feedback=[];fixed=[]
    for case,change in CHANGES.items():
        q=replace(P,**change);validate(q)
        if reuse:r=score(vector(old.loc[case]),q);lg=[]
        else:r,lg=optimize(q,old.loc[case])
        assert r['feasible'] and r['J']>=old.loc[case,'J']-1e-7
        rows.append(dict(case=case,**r,previous_J=old.loc[case,'J'],J_change_from_previous=r['J']-old.loc[case,'J']))
        logs.extend(dict(case=case,**l) for l in lg)
        baseline_earmarked=r['B']-r['available']
        residual=r['available']-r['budget_used']
        parts={'infrastructure_new':r['spend_I'],'ecology_new':r['spend_E'],'community_new':r['spend_C'],'measures':r['measure_cost'],'maintenance_charged':r['maintenance_charged'],'available_unallocated':residual,'baseline_retained':baseline_earmarked}
        assert abs(sum(parts.values())-r['B'])<1e-9
        assert abs(sum(v for k,v in parts.items() if k!='baseline_retained')-r['available'])<1e-9
        for category,amount in parts.items():ledgers.append(dict(case=case,budget_mode=q.budget_mode,category=category,million_usd=100*amount,share_of_total_tax=amount/r['B'],total_tax_million=100*r['B'],available_million=100*r['available']))
        for channel,raw,after,gain in [('infrastructure',q.I0,r['I'],r['capacity_gain']),('ecology',r['C_env_raw'],r['C_env'],r['eco_reduction']),('community',r['C_soc_raw'],r['C_soc'],r['community_reduction'])]:
            actual=(after/raw-1) if channel=='infrastructure' else (1-after/raw)
            assert abs(actual-gain)<1e-12
            feedback.append(dict(case=case,channel=channel,before_investment=raw,after_investment=after,relative_effect=gain,interpretation='Within the same equilibrium; not a separately solved no-investment policy'))
        if case in ['default','no_community_effect','no_investment_effects']:
            fixed.append(dict(case=case,**score(vector(BEST),q)))
        print(f'M2 {case}: x={r["x"]:.6f}, J={r["J"]:.9f}',flush=True)
    save(rows,'M2_scenarios');save(logs,'M2_solver_diagnostics');save(ledgers,'M2_budget_ledger');save(feedback,'M2_feedback');save(fixed,'M2_fixed_policy_ablations')
    return {'scenario_count':len(rows),'optimization_rerun':not reuse,'solver_runs':len(logs),'max_score_change_from_previous':max(abs(r['J_change_from_previous']) for r in rows)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse',action='store_true',help='Recompute saved M2 decisions without searching');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    for path,meta in json.loads((IN/'source_manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==meta['sha256'],f'Frozen input changed: {path}'
    info={'python':platform.python_version(),'M2':m2(args.reuse)}
    (OUT/'verification.json').write_text(json.dumps(info,indent=2,ensure_ascii=False))
    print('M2 complete',flush=True)

if __name__=='__main__':main()
