"""M3: sensitivity, stress tests, preference scenarios, policy landscape and income checks."""
from pathlib import Path
import argparse, json, hashlib, platform
import numpy as np
import pandas as pd
from model import Params, evaluate, vector

ROOT=Path(__file__).resolve().parent
IN=ROOT/'inputs'; OUT=ROOT/'results'
P=Params(**json.loads((IN/'parameters_used.json').read_text()))
W=np.asarray(json.loads((IN/'weights_used.json').read_text())['weights'])
BEST=pd.read_csv(IN/'recommended.csv').iloc[0].to_dict()

def save(rows,name):
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')

def score(v,p,w=W):
    r,f,g=evaluate(v,p);r['J']=float(-np.asarray(w)@f-r['soft_penalty']);return r

COMPARE=['x','xd','R','B','I','S','C_env','C_soc','C_inf','J','budget_used','max_violation']
def compare(r,old):
    err=max(abs(float(r[k])-float(old[k])) for k in COMPARE)
    assert err<1e-7,(err,r['x'],old['x'])
    assert bool(r['feasible'])==bool(old['feasible'])
    return err

def m3():
    from dataclasses import replace
    sensitivity=[];errors=[]
    for old in pd.read_csv(IN/'sensitivity.csv').to_dict('records'):
        q=replace(P,**{old['parameter']:getattr(P,old['parameter'])*old['factor']})
        r=score(vector(old),q);errors.append(compare(r,old))
        sensitivity.append(dict(parameter=old['parameter'],factor=old['factor'],method=old['method'],delta_J=r['J']-BEST['J'],**r))
    sens=pd.DataFrame(sensitivity);save(sens,'M3_sensitivity')
    ranking=[]
    for key,sub in sens.groupby('parameter'):
        opt=sub[sub.method=='reoptimized'];fx=sub[sub.method=='fixed']
        ranking.append(dict(parameter=key,max_abs_delta_J=float(opt.delta_J.abs().max()),delta_J_minus=float(opt[opt.factor==.9].delta_J.iloc[0]),delta_J_plus=float(opt[opt.factor==1.1].delta_J.iloc[0]),fixed_policy_feasible_cases=int(fx.feasible.sum()),cases=2))
    save(pd.DataFrame(ranking).sort_values('max_abs_delta_J',ascending=False),'M3_sensitivity_ranking')
    rng=np.random.default_rng(20261026);stress=[];oldstress=pd.read_csv(IN/'stress_samples.csv')
    keys=['epsilon','eta_I','eta_E','eta_C','K_I','K_E','K_C','peak_factor','measure_cost_multiplier']
    for i in range(600):
        q=replace(P,epsilon=rng.uniform(-.8,-.2),eta_I=rng.uniform(.5,.9),eta_E=rng.uniform(.6,.95),eta_C=rng.uniform(.4,.8),K_I=P.K_I*rng.uniform(.7,1.3),K_E=P.K_E*rng.uniform(.7,1.3),K_C=P.K_C*rng.uniform(.7,1.3),peak_factor=P.peak_factor*rng.uniform(.9,1.1),measure_cost_multiplier=rng.uniform(.8,1.2))
        r=score(vector(BEST),q);old=oldstress.iloc[i]
        for k in keys:assert abs(getattr(q,k)-old['input_'+k])<1e-12
        errors.append(compare(r,old));stress.append(dict(sample=i,**{'input_'+k:getattr(q,k) for k in keys},**r))
    stress=pd.DataFrame(stress);save(stress,'M3_stress_samples')
    violations=[dict(constraint=k[7:],violations=int((stress[k]<-1e-7).sum()),sample_count=len(stress),violation_fraction=float((stress[k]<-1e-7).mean())) for k in stress if k.startswith('margin_')]
    save(pd.DataFrame(violations).sort_values('violations',ascending=False),'M3_constraint_violations')
    weights=[]
    for old in pd.read_csv(IN/'scenarios.csv').to_dict('records'):
        if old['case'] in ['default','economic_priority','ecological_priority','community_priority','new_entropy_weights']:
            w=[old['weight_'+k] for k in ['R','E','S','I']];r=score(vector(old),P,w);errors.append(compare(r,old))
            weights.append(dict(case=old['case'],**{'weight_'+k:v for k,v in zip(['R','E','S','I'],w)},**r))
    save(weights,'M3_preference_scenarios')
    seed=pd.read_csv(IN/'seed_comparison.csv');seed['refined_reference_J']=BEST['J'];seed['absolute_score_gap']=BEST['J']-seed.best_J
    assert (seed.absolute_score_gap>=0).all()
    save(seed,'M3_search_stage_gaps')
    margins=[]
    for old in pd.read_csv(IN/'marginal_transfers.csv').to_dict('records'):
        r=score(vector(old),P);errors.append(compare(r,old));margins.append(dict(origin=old['origin'],destination=old['destination'],nominal_transfer_million=old['nominal_transfer_million'],J_change=r['J']-BEST['J'],**r))
    save(margins,'M3_marginal_transfers')
    surface=[]
    for old in pd.read_csv(IN/'policy_landscape.csv').to_dict('records'):
        r=score(vector(old),P);errors.append(compare(r,old));surface.append(dict(transfer=old['transfer'],**r))
    save(surface,'M3_policy_landscape')
    for n in ['counterfactuals','floor_sweep','penalty_sweep']:
        save(pd.read_csv(IN/('income_'+n+'.csv')),'M3_income_'+n)
    info={'sensitivity_rows_recomputed':len(sens),'stress_equilibria_rerun':len(stress),'landscape_rows_recomputed':len(surface),'max_replay_absolute_error':max(errors),'stress_feasible_count':int(stress.feasible.sum()),'stress_sample_count':len(stress),'stress_feasible_fraction':float(stress.feasible.mean()),'optimization_search_reused':True,'stress_is_empirical_probability':False,'global_optimum_proven':False}
    print('M3:',info,flush=True);return info

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--with-preferences',action='store_true',help='Also run the preference-space and minimax-regret extension');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    for path,meta in json.loads((IN/'source_manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==meta['sha256'],f'Frozen input changed: {path}'
    info={'python':platform.python_version(),'M3':m3()}
    (OUT/'verification.json').write_text(json.dumps(info,indent=2,ensure_ascii=False))
    if args.with_preferences:
        from run_preferences import main as preference_main
        preference_main([])
    print('M3 complete',flush=True)

if __name__=='__main__':main()
