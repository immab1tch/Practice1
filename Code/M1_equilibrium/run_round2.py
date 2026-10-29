"""Reproducible round-two experiment. All scenario changes are explicit."""
import argparse,json,hashlib,platform
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import pandas as pd
from model import Params,validate,evaluate,state_at,response,solve,refine,vector,weights_and_rank

ROOT=Path(__file__).resolve().parent

def score(v,p,w):
    r,f,_=evaluate(v,p);r['J']=float(-np.asarray(w)@f-r['soft_penalty']);return r

def optimize(p,w,starts):
    candidates,logs=refine(p,w,[np.asarray(v)[:4] for v in starts])
    # Include the previous feasible decision to avoid a worse result from a failed
    # local solve. Reoptimization never discards a feasible warm start.
    for v in starts:
        if len(v)==9:
            r=score(v,p,w)
            if r['feasible']:candidates.append(r)
    best=max(candidates,key=lambda r:r['J']) if candidates else None
    # When an investment channel has exactly zero effect, prefer zero spending
    # among objective-equivalent policies; never report arbitrary idle spending.
    if best is not None:
        v=vector(best)
        for i,k in enumerate('IEC',start=1):
            if getattr(p,'eta_'+k)==0:v[i]=0.
        canonical=score(v,p,w)
        if canonical['feasible'] and canonical['J']>=best['J']-1e-9:best=canonical
    return best,candidates,logs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=ROOT/'config.json');ap.add_argument('--output',type=Path,default=ROOT/'results')
    ap.add_argument('--population',type=int,default=180);ap.add_argument('--generations',type=int,default=160)
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,73,109]);args=ap.parse_args()
    if args.population<32 or args.generations<1:raise ValueError('Population >= 32, generations >= 1 required')
    p=Params(**json.loads(args.config.read_text()));validate(p)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    def save(rows,name):pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
    old=pd.read_csv(ROOT/'baseline_round1/recommended.csv').iloc[0].to_dict()
    # Freeze first-round preferences to isolate equation and accounting changes.
    w=json.loads((ROOT/'baseline_round1/weights.json').read_text())['combined']
    (out/'weights_used.json').write_text(json.dumps(dict(weights=w,source='Frozen round-one combined weights; no cross-round preference drift'),indent=2))
    (out/'parameters_used.json').write_text(json.dumps(asdict(p),indent=2))
    candidates=[];seed_results=[]
    for seed in args.seeds:
        rows=solve(p,seed,args.population,args.generations)
        for r in rows:r.update(seed=seed,J=score(vector(r),p,w)['J'])
        candidates+=rows
        seed_results.append(dict(seed=seed,feasible_count=len(rows),best_J=max([r['J'] for r in rows],default=np.nan)))
        print(f'NSGA-II seed {seed}: {len(rows)} feasible candidates',flush=True)
    if not candidates:raise RuntimeError('No feasible candidates found; no infeasibility proof')
    candidates.sort(key=lambda r:r['J'],reverse=True)
    starts=[vector(candidates[0])[:4],[.14,.2,.25,.2],[.15,.1,.3,.2],[.12,.2,.2,.2]]
    refined,logs=refine(p,w,[np.asarray(v)[:4] for v in starts])
    all_rows=candidates+refined
    best=max(all_rows,key=lambda r:r['J'])
    # Entropy/AHP recomputation is reported as an alternative, not silently applied.
    front,derived=weights_and_rank([dict(r) for r in all_rows],p)
    for r in front:r['J']=score(vector(r),p,w)['J']
    save(front,'pareto');save(refined,'measure_combinations');save(logs,'solver_diagnostics');save(seed_results,'seed_comparison');save([best],'recommended')
    (out/'derived_weights_alternative.json').write_text(json.dumps(derived,indent=2))
    old_now=score(vector(old),p,w);baseline=score([p.tau0,0,0,0,0,0,0,0,0],p,w)
    save([dict(case='baseline',**baseline),dict(case='round1_policy_in_round2',**old_now),dict(case='round2',**best)],'comparable_policies')
    save([dict(constraint=k[7:],margin=v,passed=v>=-1e-7) for k,v in best.items() if k.startswith('margin_')],'constraint_audit')
    print('Main recommendation:',{k:best[k] for k in ['x','tau','spend_I','spend_E','spend_C','S','J']},flush=True)
    scenarios=[]
    configurations=[('default',p,w),('no_maintenance',replace(p,maintenance_multiplier=0),w),('incremental',replace(p,budget_mode='incremental'),w),
        ('no_community_effect',replace(p,eta_C=0),w),('no_investment_effects',replace(p,eta_I=0,eta_E=0,eta_C=0),w),
        ('economic_priority',p,[.55,.15,.15,.15]),('ecological_priority',p,[.2,.5,.15,.15]),('community_priority',p,[.2,.2,.45,.15]),('new_entropy_weights',p,derived['combined']),('strict_acceptance',replace(p,S_min=.9),w)]
    for name,q,ww in configurations:
        validate(q)
        if name=='default':r=best
        else:
            r,_,_=optimize(q,ww,[vector(best),[.14,.2,.25,.2],[.15,.05,.2,.05]])
        if r is None:scenarios.append(dict(case=name,feasible=False))
        else:scenarios.append(dict(case=name,weight_R=ww[0],weight_E=ww[1],weight_S=ww[2],weight_I=ww[3],**r))
        print('Scenario:',name,flush=True)
    save(scenarios,'scenarios')
    sensitivity=[]
    for key in ['epsilon','eta_I','eta_E','eta_C','K_I','K_E','K_C','acceptance_slope','peak_factor','measure_cost_multiplier','R0','S_min']:
        for factor in [.9,1.1]:
            q=replace(p,**{key:getattr(p,key)*factor});validate(q)
            fixed=score(vector(best),q,w)
            sensitivity.append(dict(parameter=key,factor=factor,method='fixed',**fixed))
            r,_,_=optimize(q,w,[vector(best),[.14,.15,.3,.15]])
            sensitivity.append(dict(parameter=key,factor=factor,method='reoptimized',**(r or dict(feasible=False))))
        print('Sensitivity:',key,flush=True)
    save(sensitivity,'sensitivity')
    # Fixed-policy stress test: exploratory independent uniforms, no empirical probabilities.
    rng=np.random.default_rng(20261026);stress=[]
    for i in range(600):
        q=replace(p,epsilon=rng.uniform(-.8,-.2),eta_I=rng.uniform(.5,.9),eta_E=rng.uniform(.6,.95),eta_C=rng.uniform(.4,.8),
            K_I=p.K_I*rng.uniform(.7,1.3),K_E=p.K_E*rng.uniform(.7,1.3),K_C=p.K_C*rng.uniform(.7,1.3),
            peak_factor=p.peak_factor*rng.uniform(.9,1.1),measure_cost_multiplier=rng.uniform(.8,1.2))
        r=score(vector(best),q,w)
        stress.append(dict(sample=i,**{f'input_{k}':getattr(q,k) for k in ['epsilon','eta_I','eta_E','eta_C','K_I','K_E','K_C','peak_factor','measure_cost_multiplier']},**r))
    save(stress,'stress_samples')
    curves=[]
    for spend in np.linspace(0,.5,101):
        for k in ['I','E','C']:
            eta=getattr(p,'eta_'+k);L=getattr(p,'L_'+k);K=getattr(p,'K_'+k)
            curves.append(dict(category=k,spend_million=spend*100,response=response(spend,eta,L,K),marginal_per_million=eta*L/K*np.exp(-spend/K)/100))
    save(curves,'response_curves')
    # Reallocate ecology to infrastructure/community with constant total shares.
    landscape=[];v=vector(best)
    for tax in np.linspace(p.tau_min,p.tau_max,49):
        for transfer in np.linspace(-.12,.12,49):
            z=v.copy();z[0]=tax;z[1]+=transfer;z[2]-=transfer
            if min(z[1:4])<0:continue
            r=score(z,p,w);landscape.append(dict(transfer=transfer,**r))
    save(landscape,'policy_landscape')
    # Budget-preserving marginal transfers. Both directions are kept for audit.
    marginal=[]
    for origin in range(3):
        for dest in range(3):
            if origin==dest:continue
            z=vector(best);ds=.001/best['available'] # $100,000 nominal at reference
            z[1+origin]-=ds;z[1+dest]+=ds
            if z[1+origin]<0:continue
            r=score(z,p,w);marginal.append(dict(origin='IEC'[origin],destination='IEC'[dest],nominal_transfer_million=.1,J_change=r['J']-best['J'],**r))
    save(marginal,'marginal_transfers')
    manifest=dict(seeds=args.seeds,population=args.population,generations=args.generations,python=platform.python_version(),stress_seed=20261026,stress_n=600,global_optimum_proven=False,
        hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/'model.py',ROOT/'run_round2.py',args.config,ROOT/'baseline_round1/recommended.csv']})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    from figures_en import create_figures
    create_figures(out,ROOT,p)
    print('Finished; figures and interpretation saved.',flush=True)

if __name__=='__main__':main()
