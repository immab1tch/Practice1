"""Add preference-simplex scanning and minimax regret without modifying M1/M2.

Scores are evaluated under a common weight within each regret column. The
continuous reference optima are numerical best-known values, not proven global
optima. Both the finite policy-menu choice and continuous minimax refinement
are retained, so their decision domains cannot be confused.
"""
from pathlib import Path
from dataclasses import asdict, replace
import argparse, hashlib, itertools, json, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from model import Params, validate, evaluate, refine, vector

ROOT=Path(__file__).resolve().parent; IN=ROOT/'inputs'; OUT=ROOT/'results/preference_robustness'
KEYS=['tau','delta_I','delta_E','delta_C','m1','m2','m3','m4','m5']
AXES=['R','E','S','I']; PN=1e-8

def save(rows,name):
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')

def weight_points(main,derived):
    rows=[]
    def add(wid,label,kind,w):rows.append(dict(weight_id=wid,label=label,kind=kind,**dict(zip(['w_'+k for k in AXES],np.asarray(w,float)))))
    names=['收入','环境','社会','基建']
    for i in range(4):add('V'+AXES[i],names[i]+'顶点','vertex',np.eye(4)[i])
    for i,j in itertools.combinations(range(4),2):
        w=np.zeros(4);w[[i,j]]=.5;add('E'+AXES[i]+AXES[j],names[i]+'—'+names[j]+'边中点','edge',w)
    for i in range(4):
        w=np.ones(4)/3;w[i]=0;add('F'+AXES[i],'不含'+names[i]+'的面心','face',w)
    add('C','等权中心','center',[.25]*4)
    add('MAIN','原推荐权重','named',main);add('AHP','原AHP权重','named',derived['ahp']);add('ENT','第二轮熵权','named',derived['entropy'])
    df=pd.DataFrame(rows);W=df[['w_'+k for k in AXES]].to_numpy()
    assert len(df)==18 and np.all(W>=0) and np.allclose(W.sum(1),1)
    return df,W

def state(v,p):return evaluate(v,p)[0]
def matrix(rows,W,p):
    f=np.array([[r['R']/p.R0,-r['C_env'],-r['C_soc'],-r['C_inf']] for r in rows])
    if not np.isfinite(f).all() or not np.isfinite(W).all():raise ValueError('Nonfinite utility or weight')
    return np.einsum('ij,kj->ik',f,W,optimize=False)-np.array([r['soft_penalty'] for r in rows])[:,None]
def scored(v,p,w):
    r,f,g=evaluate(v,p);r['J']=float(-w@f-r['soft_penalty']);return r

def pool_best(rows,W,main,p):
    S=matrix(rows,W,p);idx=[];tie_scores=matrix(rows,np.asarray(main)[None,:],p)[:,0]
    for j in range(len(W)):
        candidates=np.flatnonzero(S[:,j]>=S[:,j].max()-1e-9)
        idx.append(int(candidates[np.argmax(tie_scores[candidates])]))
    return idx,S.max(0),S

def continuous_minimax(p,W,reference,starts):
    """Epigraph t bounds regret; three auxiliary deficits linearize max penalties."""
    penalties=np.array([p.penalty_S,p.penalty_R,p.penalty_H]);rows=[];logs=[]
    for combo in itertools.product([0,1],repeat=5):
        cache_z=None;cache_rfg=None
        def ev(z):
            nonlocal cache_z,cache_rfg
            if cache_z is None or not np.array_equal(z[:4],cache_z):
                cache_z=z[:4].copy();cache_rfg=evaluate(np.r_[z[:4],combo],p)
            return cache_rfg
        def gaps(r):return np.array([p.S_min-r['S'],(p.R_min-r['R'])/p.R0,r['C_soc']-p.H_max])
        def constraints(z):
            r,f,g=ev(z)
            return np.r_[-g,z[4:7]-gaps(r),z[7]-reference-W@f-penalties@z[4:7]]
        for si,v in enumerate(starts):
            r,f,g=ev(v);slack=np.maximum(gaps(r),0)+1e-6
            t=max(0.,float(np.max(reference+W@f+penalties@slack)))+1e-6
            z0=np.r_[v[:4],slack,t]
            opt=minimize(lambda z:z[7],z0,jac=lambda z:np.array([0.]*7+[1.]),method='SLSQP',bounds=[(p.tau_min,p.tau_max)]+[(0,1)]*3+[(0,None)]*4,constraints={'type':'ineq','fun':constraints},options={'maxiter':350,'ftol':1e-11})
            r,f,g=ev(opt.x);regret=reference+W@f+r['soft_penalty']
            violation=max(0.,-float(np.min(constraints(opt.x))))
            logs.append(dict(combo=''.join(map(str,combo)),start_index=si,success=bool(opt.success),message=str(opt.message),epigraph_violation=violation,max_violation=r['max_violation'],worst_regret=float(regret.max())))
            if opt.success and r['feasible'] and violation<1e-7:rows.append(dict(r,solver_worst_regret=float(regret.max())))
    return rows,logs

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true',help='Reuse completed weight optimizations only when input hash matches');args=parser.parse_args(argv)
    OUT.mkdir(exist_ok=True);check=OUT/'checkpoints';check.mkdir(exist_ok=True)
    p=Params(**json.loads((IN/'parameters_used.json').read_text()));validate(p)
    mw=np.asarray(json.loads((IN/'weights_used.json').read_text())['weights'])
    derived=json.loads((IN/'preference_robustness/derived_weights_alternative.json').read_text())
    wp,W=weight_points(mw,derived);save(wp,'weights')
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/'model.py',ROOT/'run_preferences.py',IN/'parameters_used.json',IN/'weights_used.json',IN/'preference_robustness/derived_weights_alternative.json',IN/'recommended.csv',IN/'scenarios.csv']}
    signature=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    baseline=state(vector(pd.read_csv(IN/'recommended.csv').iloc[0]),p)
    # Only original-parameter policies enter this feasible pool. Modified-model
    # M2 scenarios must NOT be treated as feasible without reevaluation.
    pool=[dict(baseline,source='original_recommendation')]
    for old in pd.read_csv(IN/'scenarios.csv').to_dict('records'):
        if old['case'] in ['default','economic_priority','ecological_priority','community_priority','new_entropy_weights']:
            r=state(vector(old),p)
            if r['feasible']:pool.append(dict(r,source='existing_'+old['case']))
    audit=[];logs=[];t0=time.time()
    for i,weight in wp.iterrows():
        wid=weight.weight_id;file=check/(wid+'.json');w=W[i]
        if args.resume and file.exists():
            data=json.loads(file.read_text());assert data['signature']==signature,'Checkpoint inputs changed; rerun without --resume'
        else:
            current_scores=matrix(pool,w[None,:],p)[:,0];warm=pool[int(np.argmax(current_scores))]
            starts=[vector(warm)[:4],np.array([.14,.2,.25,.2]),np.array([.15,.05,.35,.2])]
            candidates,lg=refine(p,w,starts)
            if not candidates:raise RuntimeError(f'No locally solved feasible policy at {wid}; not an infeasibility proof')
            data={'signature':signature,'rows':candidates,'logs':lg};file.write_text(json.dumps(data,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
        for r in data['rows']:pool.append(dict(r,source='weight_'+wid))
        logs.extend(dict(weight_id=wid,**l) for l in data['logs'])
        selected,reference,scores=pool_best(pool,w[None,:],mw,p);r=pool[selected[0]]
        audit.append(dict(weight_id=wid,best_score_after_weight_solve=float(reference[0]),x=r['x'],combo=''.join(str(int(r['m'+str(j)])) for j in range(1,6)),successful_combinations=len(data['rows'])))
        print(f'Weight {i+1:02d}/18 {wid}: x={r["x"]:.4f}, J={reference[0]:.7f}, elapsed={time.time()-t0:.0f}s',flush=True)
    save(audit,'weight_search_audit');save(logs,'weight_solver_diagnostics')
    selected,reference,_=pool_best(pool,W,mw,p)
    # Reconcile cross-weight warm starts: improve each oracle from the best point
    # found anywhere in the shared feasible pool, retaining all previous results.
    polish_logs=[]
    for j,w in enumerate(W):
        warm=pool[selected[j]]
        if reference[j]>audit[j]['best_score_after_weight_solve']+1e-7:
            candidates,lg=refine(p,w,[vector(warm)[:4]])
            pool.extend(dict(r,source='cross_polish_'+wp.weight_id.iloc[j]) for r in candidates)
            polish_logs.extend(dict(weight_id=wp.weight_id.iloc[j],**l) for l in lg)
            print('Cross-weight improvement polished:',wp.weight_id.iloc[j],flush=True)
    save(polish_logs,'cross_weight_solver_diagnostics')
    selected,reference,_=pool_best(pool,W,mw,p)
    menu=[pool[idx] for idx in selected];menu_scores=matrix(menu,W,p);menu_worst=(reference-menu_scores).max(1)
    menu_index=int(np.argmin(menu_worst));menu_choice=menu[menu_index]
    print('Menu minimax:',wp.weight_id.iloc[menu_index],float(menu_worst[menu_index]),flush=True)
    robust_rows=[];robust_logs=[];reference_updates=[]
    # Finite-reference regret must be self-consistent if refinement discovers a
    # better benchmark. Rebuild references and rerun, rather than clipping negative
    # regret values or passing them off as exact global-oracle values.
    for iteration in range(3):
        selected,reference,_=pool_best(pool,W,mw,p);menu=[pool[idx] for idx in selected]
        menu_worst=(reference-matrix(menu,W,p)).max(1);menu_choice=menu[int(np.argmin(menu_worst))]
        previous=(min(robust_rows,key=lambda r:float((reference-matrix([r],W,p)[0]).max())) if robust_rows else baseline)
        starts=[vector(menu_choice),vector(previous),np.array([.14,.2,.25,.2])]
        rows,lg=continuous_minimax(p,W,reference,starts)
        if not rows:raise RuntimeError('Minimax refinement did not return a valid candidate')
        pool.extend(dict(r,source=f'minimax_{iteration}') for r in rows);robust_rows+=rows
        robust_logs.extend(dict(iteration=iteration,**l) for l in lg)
        _,newref,_=pool_best(pool,W,mw,p);growth=float(np.max(newref-reference));reference_updates.append(growth)
        print(f'Minimax refinement {iteration+1}: {len(rows)} feasible solutions, benchmark change={growth:.3g}',flush=True)
        if growth<=1e-7:break
    else:raise RuntimeError('Benchmarks still changing: do not publish unconverged regret estimates')
    selected,reference,all_scores=pool_best(pool,W,mw,p);menu=[pool[idx] for idx in selected]
    # Enumerated menu includes one representative for each weight (duplicates are
    # intentionally retained and identified, since they encode policy plateaus).
    records=[]
    for j,r in enumerate(menu):records.append(dict(r,policy_id='P_'+wp.weight_id.iloc[j],policy_label=wp.label.iloc[j],origin_weight=wp.weight_id.iloc[j],kind='weight_menu'))
    records.append(dict(baseline,policy_id='ORIGINAL',policy_label='原固定权重推荐',origin_weight='MAIN',kind='baseline'))
    menu_regret=reference-matrix(menu,W,p);mi=int(np.argmin(menu_regret.max(1)))
    # Search includes all already found feasible candidates as a fallback, so
    # continuous refinement can never be reported as worse than the finite menu.
    all_worst=(reference-all_scores).max(1);ri=int(np.argmin(all_worst));robust=pool[ri]
    records.append(dict(robust,policy_id='ROBUST',policy_label='连续精修稳健方案',origin_weight='',kind='continuous_minimax'))
    S=matrix(records,W,p);regrets=reference-S
    if regrets.min() < -1e-8:raise AssertionError('Regret reference is below an evaluated candidate')
    regrets[np.abs(regrets)<1e-12]=0
    ranking=[]
    for i,r in enumerate(records):
        worst=int(np.argmax(regrets[i]));ranking.append(dict(policy_id=r['policy_id'],label=r['policy_label'],max_regret=float(regrets[i,worst]),mean_regret=float(regrets[i].mean()),worst_weight=wp.weight_id.iloc[worst],x=r['x'],tau=r['tau'],income_penalty=p.penalty_R*max((p.R_min-r['R'])/p.R0,0)))
        r['J_main']=float(matrix([r],mw[None,:],p)[0,0]);r['max_regret']=ranking[-1]['max_regret']
        r.pop('J',None) # Do not retain a score computed under some earlier weight.
    save(records,'policy_menu')
    exported_pool=[{k:v for k,v in r.items() if k!='J'} for r in pool]
    save(exported_pool,'all_feasible_candidates')
    save(pd.DataFrame(S,index=[r['policy_id'] for r in records],columns=wp.weight_id).rename_axis('policy_id').reset_index(),'score_matrix')
    save(pd.DataFrame(regrets,index=[r['policy_id'] for r in records],columns=wp.weight_id).rename_axis('policy_id').reset_index(),'regret_matrix')
    save(pd.DataFrame(ranking).sort_values('max_regret',kind='stable'),'regret_ranking')
    save([dict(weight_id=wp.weight_id.iloc[j],reference_J=reference[j],best_policy_id='P_'+wp.weight_id.iloc[j],reference_gap_from_tie_break=reference[j]-S[j,j]) for j in range(len(W))],'reference_values')
    save(robust_logs,'minimax_solver_diagnostics')
    # Dense *menu evaluation*, not additional optimization: no implicit claim
    # that the few solved weights identify continuous policy-switch boundaries.
    dense=[];den=12
    for a in range(den+1):
        for b in range(den-a+1):
            for c in range(den-a-b+1):dense.append(np.array([a,b,c,den-a-b-c])/den)
    DW=np.array(dense);DS=matrix(records,DW,p);best_idx=np.argmax(DS,axis=0);best_score=DS.max(0);dense_regret=best_score[None,:]-DS
    save([dict(**dict(zip(['w_'+k for k in AXES],w)),menu_best_policy=records[k]['policy_id'],menu_best_x=records[k]['x'],menu_best_combo=''.join(str(int(records[k]['m'+str(j)])) for j in range(1,6)),robust_menu_relative_regret=float(dense_regret[-1,i])) for i,(w,k) in enumerate(zip(DW,best_idx))],'dense_menu_evaluation')
    summary={'weight_count':len(wp),'structured_points':15,'named_points':3,'all_feasible_candidates':len(pool),'menu_weight_rows':18,'menu_minimax_policy':records[mi]['policy_id'],'menu_minimax_max_regret':float(menu_regret[mi].max()),'original_max_regret':float(regrets[-2].max()),'robust_max_regret':float(regrets[-1].max()),'robust_worst_weights':[wp.weight_id.iloc[j] for j in np.flatnonzero(np.abs(regrets[-1]-regrets[-1].max())<1e-6)],'robust_x':robust['x'],'robust_R':robust['R'],'robust_tau':robust['tau'],'robust_spend_million':{k:robust['spend_'+k]*100 for k in 'IEC'},'robust_combo':''.join(str(int(robust['m'+str(j)])) for j in range(1,6)),'menu_x_min':min(r['x'] for r in menu),'menu_x_max':max(r['x'] for r in menu),'menu_plateau_at_147_2_count':sum(abs(r['x']-147.2)<1e-4 for r in menu),'dense_menu_points':len(DW),'dense_regret_max_at_vertices_verified':bool(np.all(dense_regret.max(1)<=regrets[:,:4].max(1)+1e-7)),'benchmark_updates':reference_updates,'global_optimum_proven':False,'benchmark_type':'best-known numerical optimum from pooled feasible candidates','elapsed_seconds':time.time()-t0,'hashes':hashes}
    assert summary['robust_max_regret']<=summary['menu_minimax_max_regret']+1e-7
    assert summary['robust_max_regret']<=summary['original_max_regret']+1e-7
    stress=[]
    for old in pd.read_csv(IN/'stress_samples.csv').to_dict('records'):
        changes={k[6:]:v for k,v in old.items() if k.startswith('input_')}
        q=replace(p,**changes);r=state(vector(robust),q)
        stress.append(dict(sample=old['sample'],original_feasible=bool(old['feasible']),**{'input_'+k:v for k,v in changes.items()},**r))
    save(stress,'robust_parameter_stress')
    summary['original_parameter_stress_passed']=sum(r['original_feasible'] for r in stress)
    summary['robust_parameter_stress_passed']=sum(bool(r['feasible']) for r in stress)
    summary['parameter_stress_n']=len(stress)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (OUT/'parameters_used.json').write_text(json.dumps(asdict(p),indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    from render_preferences import render
    render()
    from verify_preferences import verify
    verify()

if __name__=='__main__':main()
