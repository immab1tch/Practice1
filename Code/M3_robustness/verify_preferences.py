"""Numerical checks for preference-space and minimax-regret extension."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
import pandas as pd
from model import Params,evaluate,vector
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/preference_robustness'

def verify():
    summary=json.loads((OUT/'summary.json').read_text());p=Params(**json.loads((OUT/'parameters_used.json').read_text()))
    for name,digest in summary['hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,('changed input/code',name)
    # Frozen M1 equations must match the original integration input hash.
    manifest=json.loads((ROOT/'inputs/source_manifest.json').read_text())
    assert hashlib.sha256((ROOT/'model.py').read_bytes()).hexdigest()==manifest['model.py']['sha256']
    wp=pd.read_csv(OUT/'weights.csv');W=wp[['w_R','w_E','w_S','w_I']].to_numpy()
    assert W.shape==(18,4) and (W>=0).all() and np.allclose(W.sum(1),1)
    assert wp.kind.value_counts().to_dict()=={'edge':6,'vertex':4,'face':4,'named':3,'center':1}
    assert np.allclose(W[:4],np.eye(4)) and wp.weight_id.is_unique
    structured=[]
    for count in range(1,5):
        for selected in itertools.combinations(range(4),count):
            z=np.zeros(4);z[list(selected)]=1/count;structured.append(z)
    assert all(any(np.allclose(w,s) for s in W[:15]) for w in structured)
    policies=pd.read_csv(OUT/'policy_menu.csv');pool=pd.read_csv(OUT/'all_feasible_candidates.csv')
    score=pd.read_csv(OUT/'score_matrix.csv').set_index('policy_id');regret=pd.read_csv(OUT/'regret_matrix.csv').set_index('policy_id')
    ref=pd.read_csv(OUT/'reference_values.csv').set_index('weight_id').loc[wp.weight_id].reference_J.to_numpy()
    def independent_scores(df):
        # Recompute penalties directly, not from exported soft_penalty or J.
        penalty=p.penalty_S*np.maximum(p.S_min-df.S.to_numpy(),0)+p.penalty_R*np.maximum((p.R_min-df.R.to_numpy())/p.R0,0)+p.penalty_H*np.maximum(df.C_soc.to_numpy()-p.H_max,0)
        utility=np.column_stack([df.R/p.R0,-df.C_env,-df.C_soc,-df.C_inf])
        return (utility[:,None,:]*W[None,:,:]).sum(2)-penalty[:,None]
    computed=independent_scores(policies);expected_ref=independent_scores(pool).max(0)
    assert np.allclose(computed,score.loc[policies.policy_id,wp.weight_id],atol=1e-10,rtol=0)
    assert np.allclose(expected_ref,ref,atol=1e-10,rtol=0)
    expected_regret=ref-computed
    assert expected_regret.min()>-1e-9
    assert np.allclose(expected_regret,regret.loc[policies.policy_id,wp.weight_id],atol=1e-10,rtol=0)
    assert np.max(np.abs(np.diag(expected_regret[:18])))<2e-9
    # For linear utility and weight-independent penalties, all simplex weights
    # are convex combinations of vertices. Verify worst regret is at a vertex
    # for the common candidate-pool reference, not a per-column inconsistent set.
    assert np.all(ref<=W@ref[:4]+1e-9)
    assert np.allclose(expected_regret.max(1),expected_regret[:,:4].max(1),atol=1e-9,rtol=0)
    rank=pd.read_csv(OUT/'regret_ranking.csv').set_index('policy_id')
    for i,r in policies.iterrows():
        current,f,g=evaluate(vector(r),p)
        assert current['feasible'],r.policy_id
        for key in ['x','R','S','C_env','C_soc','C_inf','budget_used','spend_I','spend_E','spend_C']:
            assert abs(current[key]-r[key])<1e-7,(r.policy_id,key)
        assert abs(rank.loc[r.policy_id,'max_regret']-expected_regret[i].max())<1e-9
    assert rank.loc['ROBUST','max_regret']<=rank.loc['ORIGINAL','max_regret']+1e-9
    menu_min=expected_regret[:18].max(1).min()
    assert abs(summary['menu_minimax_max_regret']-menu_min)<1e-9
    assert rank.loc['ROBUST','max_regret']<=menu_min+1e-9
    assert abs(summary['robust_max_regret']-rank.loc['ROBUST','max_regret'])<1e-9
    original=pd.read_csv(ROOT/'inputs/recommended.csv').iloc[0]
    assert np.allclose(vector(policies[policies.policy_id=='ORIGINAL'].iloc[0]),vector(original))
    dense=pd.read_csv(OUT/'dense_menu_evaluation.csv');DW=dense[['w_R','w_E','w_S','w_I']].to_numpy()
    assert len(dense)==455 and np.allclose(DW.sum(1),1)
    assert (dense.robust_menu_relative_regret>=-1e-9).all()
    stress=pd.read_csv(OUT/'robust_parameter_stress.csv');margins=stress[[k for k in stress if k.startswith('margin_')]]
    passed=(margins.min(1)>=-1e-7)&(stress.demand_residual.abs()<1e-7)
    assert len(stress)==600 and passed.eq(stress.feasible).all()
    assert passed.sum()==summary['robust_parameter_stress_passed']
    # Small exact cases check regret orientation and invariance to weight-column
    # score offsets, including negative raw utility values.
    toy=np.array([[10.,0.],[6.,6.],[0.,10.]])
    toyreg=toy.max(0)-toy
    assert np.argmin(toyreg.max(1))==1 and toyreg[1].max()==4
    shifted=toy+np.array([-30.,2.]);assert np.allclose(shifted.max(0)-shifted,toyreg)
    fig=pd.read_csv(OUT/'figure_index_en.csv');assert len(fig)==5
    for name in fig.file:
        for ext in ['png','svg']:assert (ROOT/'figures/preference_robustness'/f'{name}.{ext}').stat().st_size>1000
    result={'status':'passed','weight_points':18,'policy_rows_checked':len(policies),'regret_cells_checked':expected_regret.size,'dense_menu_points':len(dense),'stress_rows':len(stress),'worst_case_vertex_property_verified':True,'minimax_toy_and_offset_checks':True,'original_model_hash_unchanged':True,'global_optimum_proven':False}
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
    return result
if __name__=='__main__':verify()
