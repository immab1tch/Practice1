"""Targeted identification audit of R_min and penalty_R; round-two model frozen."""
import json,itertools,hashlib,platform
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.optimize import minimize,differential_evolution,NonlinearConstraint
from model_snapshot import Params,validate,evaluate,refine,vector
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
P=Params(**json.loads((ROOT/'inputs/parameters_used.json').read_text()))
W=np.array(json.loads((ROOT/'inputs/weights_used.json').read_text())['weights'])
OLD=pd.read_csv(ROOT/'inputs/recommended.csv').iloc[0]
BASE_STARTS=[vector(OLD)[:4],[.15,.15,.3,.2],[.11,.25,.25,.15],[.135,.05,.4,.15]]
LOGS=[];COMBOS=[];CACHE={}

def save(rows,name):pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')

def enrich(r,p):
    r=dict(r);raw,f,_=evaluate(vector(r),p)
    r.update(raw)
    r['J_gross']=float(-W@f)
    r['income_gap']=max(0.,p.R_min-r['R'])
    r['income_penalty']=p.penalty_R*r['income_gap']/p.R0
    r['other_penalty']=r['soft_penalty']-r['income_penalty']
    r['J_without_income_penalty']=r['J_gross']-r['other_penalty']
    r['J']=r['J_gross']-r['soft_penalty']
    r['floor_x']=p.R_min*p.x0/p.R0
    r['revenue_margin']=r['R']-p.R_min
    r['R_min']=p.R_min;r['penalty_R']=p.penalty_R
    r['at_income_kink']=abs(r['revenue_margin'])<1e-6
    r['combo']=''.join(str(int(r['m'+str(i)])) for i in range(1,6))
    return r

def solve_case(floor,penalty,warm=None):
    key=(float(floor),float(penalty))
    if key in CACHE:return CACHE[key]
    p=replace(P,R_min=floor,penalty_R=penalty);validate(p)
    starts=list(BASE_STARTS)
    if warm is not None:starts.append(vector(warm)[:4])
    rows,logs=refine(p,W,starts)
    if not rows:raise RuntimeError(f'No feasible candidate for {key}; this is not proof of infeasibility')
    for row in rows:COMBOS.append(dict(run_type='sweep',**enrich(row,p)))
    for l in logs:LOGS.append(dict(run_type='sweep',R_min=floor,penalty_R=penalty,**l))
    best=enrich(rows[0],p)
    # Preserve every valid warm policy even if a local solver reports failure.
    if warm is not None:
        rr=enrich(warm,p)
        if rr['feasible'] and rr['J']>best['J']:best=rr
    CACHE[key]=best
    return best

def profile(target,starts):
    """Fix visitor volume, reoptimize tau/allocations and enumerate all 32 bits.
    Income penalty is disabled while computing the upper envelope, then added
    analytically at the fixed x. A soft deficit epigraph handles other penalties.
    """
    p=replace(P,penalty_R=0.)
    candidates=[]
    for combo in itertools.product([0,1],repeat=5):
        # Analytically impossible daily-cap combinations need no solver.
        if p.peak_factor*target/p.season_days>p.peak0-p.cap_reduction*combo[0]+1e-9:continue
        def ev(z):return evaluate(np.r_[z[:4],combo],p)
        def gaps(r):return np.array([p.S_min-r['S'],r['C_soc']-p.H_max])
        def obj(z):r,f,g=ev(z);return float(W@f+p.penalty_S*z[4]+p.penalty_H*z[5])
        def ine(z):r,f,g=ev(z);return np.r_[-g,z[4:]-gaps(r)]
        def eq(z):return (ev(z)[0]['x']-target)/p.x0
        for start in starts:
            start=np.asarray(start)[:4];r,_,_=ev(start)
            z=np.r_[start,np.maximum(gaps(r),0)+1e-7]
            opt=minimize(obj,z,method='SLSQP',bounds=[(p.tau_min,p.tau_max)]+[(0,1)]*3+[(0,None)]*2,
                constraints=[{'type':'ineq','fun':ine},{'type':'eq','fun':eq}],options={'maxiter':250,'ftol':1e-11})
            r,_,_=ev(opt.x)
            LOGS.append(dict(run_type='profile',target_x=target,combo=''.join(map(str,combo)),success=bool(opt.success),message=str(opt.message),max_violation=r['max_violation'],profile_residual=r['x']-target))
            if opt.success and r['feasible'] and abs(r['x']-target)<1e-6:
                rr=enrich(r,P);rr['target_x']=target;candidates.append(rr)
    if not candidates:raise RuntimeError(f'Profile could not resolve x={target}')
    return max(candidates,key=lambda r:r['J_without_income_penalty'])

def main():
    baseline=solve_case(P.R_min,P.penalty_R)
    nop=solve_case(P.R_min,0.,baseline)
    removed=solve_case(0.,P.penalty_R,nop)
    save([dict(case='default',**baseline),dict(case='income_penalty_removed',**nop),dict(case='income_floor_removed',**removed)],'counterfactuals')
    print('COUNTERFACTUAL',[(x['x'],x['R'],x['combo']) for x in [baseline,nop,removed]],flush=True)
    floors=[0,2.9,2.99,3.,3.005,3.01,3.02,3.05,3.1,3.2,3.3,3.35,3.39,3.4,3.425,3.44,3.45,3.46,3.475,3.5,3.55,3.6,3.65,3.7,3.75,3.8,4.]
    rows=[];warm=nop
    for floor in floors:
        r=solve_case(floor,P.penalty_R,warm);rows.append(r);warm=r
        print(f'FLOOR {floor:.3f}: x={r["x"]:.6f} R={r["R"]:.6f} kink={r["at_income_kink"]} m={r["combo"]}',flush=True)
    save(rows,'floor_sweep')
    penalties=[0,.005,.01,.02,.03,.04,.05,.075,.1,.15,.2,.5,1,2,5,10]
    rows=[];warm=nop
    for penalty in penalties:
        r=solve_case(P.R_min,penalty,warm);rows.append(r);warm=r
        print(f'PENALTY {penalty}: x={r["x"]:.6f} gap={r["income_gap"]:.6f}',flush=True)
    save(rows,'penalty_sweep')
    # Targeted fixed-x profile around the suspected kink; all other decisions free.
    targets=[144.,145.,146.,147.,147.1,147.19,147.2,147.21,147.3,147.4,148.,149.,150.,152.]
    profiles=[];starts=[vector(baseline)[:4],[.15,.2,.3,.15]]
    for target in targets:
        r=profile(target,starts);profiles.append(r);starts=[vector(r)[:4],vector(baseline)[:4]]
        print(f'PROFILE x={target:.2f}: gross-other={r["J_without_income_penalty"]:.8f} net={r["J"]:.8f}',flush=True)
    save(profiles,'visitor_profile')
    # Independent stochastic optimizer cross-checks the selected discrete branch.
    cross=[]
    for name,p,reference in [('default',P,baseline),('no_income_penalty',replace(P,penalty_R=0),nop)]:
        combo=[reference['m'+str(i)] for i in range(1,6)]
        def ev(z):return evaluate(np.r_[z,combo],p)
        def obj(z):r,f,g=ev(z);return float(W@f+r['soft_penalty'])
        con=NonlinearConstraint(lambda z:ev(z)[2],-np.inf,0.)
        res=differential_evolution(obj,[(p.tau_min,p.tau_max)]+[(0,1)]*3,constraints=con,
            seed=451,popsize=18,maxiter=450,tol=1e-9,polish=False)
        r=enrich(ev(res.x)[0],p)
        cross.append(dict(case=name,optimizer_success=bool(res.success),optimizer_message=str(res.message),reference_J=reference['J'],score_gap=reference['J']-r['J'],**r))
        print('CROSSCHECK',name,r['x'],reference['J']-r['J'],flush=True)
    save(cross,'independent_optimizer');save(COMBOS,'all_combo_solutions');save(LOGS,'solver_diagnostics')
    manifest=dict(model_sha256=hashlib.sha256((ROOT/'model_snapshot.py').read_bytes()).hexdigest(),original_model_sha256=(ROOT/'inputs/original_model.sha256').read_text().strip(),python=platform.python_version(),weights=W.tolist(),floor_grid=floors,penalty_grid=penalties,profile_grid=targets,global_optimality_proven=False,independent_seed=451)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    from report import render
    render(ROOT)
    print('AUDIT COMPLETE',flush=True)

if __name__=='__main__':main()
