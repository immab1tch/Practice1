"""M1 round 2: diminishing returns, direct community benefit, smooth acceptance.
Units: 10,000 visitors, 100 million USD, dimensionless pressure indices.
Repaired mode is a documented extension, not a verbatim transcription.
"""
from dataclasses import dataclass, asdict, replace
import itertools
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as evolutionary_minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

@dataclass
class Params:
    x0: float = 160.; R0: float = 3.75; P: float = 3.
    tau0: float = .11; tau_min: float = .09; tau_max: float = .15
    epsilon: float = -.5; alpha: float = .2; beta: float = .2
    I0: float = 160.; S0: float = .7; peak0: float = 2.
    season_days: float = 214.; peak_factor: float = 2.675
    eta_I: float = .7; eta_E: float = .8; eta_C: float = .6
    delta_total: float = .8; E_max: float = 1.; S_min: float = .7
    R_min: float = 3.45; H_max: float = 1.
    e1: float = .5; e2: float = .5
    s_h: float = .5; s_n: float = .3; s_noise: float = .2
    i_p: float = .6; i_d: float = .4
    power_effect: float = .3; traffic_effect: float = .4
    alcohol_effect: float = .2
    cap_reduction: float = .4
    passengers_per_ship_10k: float = .4; ship_limit: int = 5; physical_berths: int = 7
    kappa: tuple = (-.1, -.05, .02, .03, .1)
    # Annual-equivalent costs. Shore power derived from $65m midpoint / annuity;
    # other entries are explicitly hypothetical, NOT measured cost estimates.
    measure_costs: tuple = (.002, .001, .047828138, .00358825, .005)
    measure_cost_multiplier: float = 1.
    ahp_weights: tuple = (.35, .30, .20, .15)
    theta: float = .5
    penalty_S: float = 5.; penalty_R: float = 5.; penalty_H: float = 2.
    mode: str = 'round2'
    K_I: float = .15; K_E: float = .15; K_C: float = .08
    L_I: float = .5; L_E: float = .9; L_C: float = .8
    acceptance_slope: float = 2.
    maintenance_I: float = .006419
    maintenance_C: float = .006134
    maintenance_multiplier: float = 1.
    budget_mode: str = 'total'  # 'incremental': max(tau*R - tau0*R0, 0)


def validate(p):
    if p.mode != 'round2' or p.budget_mode not in ('total','incremental'):
        raise ValueError('Invalid model/budget mode')
    if not (0 < p.S0 < 1 and 0 < p.S_min < 1 and 0 < p.tau_min <= p.tau0 <= p.tau_max):
        raise ValueError('Invalid reference acceptance or tax interval')
    if min(p.K_I,p.K_E,p.K_C,p.I0,p.x0,p.R0,p.P,p.season_days,p.passengers_per_ship_10k)<=0:
        raise ValueError('Scales must be positive')
    if not (0<=p.eta_E*p.L_E<1 and 0<=p.eta_C*p.L_C<1 and p.eta_I>=0 and p.L_I>=0):
        raise ValueError('Response ceilings must leave positive residual pressure')
    if not (0<p.delta_total<=1 and p.peak_factor>=1 and p.maintenance_multiplier>=0):
        raise ValueError('Invalid budget or seasonal concentration')


def response(spend,eta,limit,scale):
    """Nonnegative, increasing, concave response; spend unit = 100m USD."""
    return eta*limit*(-np.expm1(-spend/scale))


def state_at(x,v,p):
    tau,dI,dE,dC=np.asarray(v[:4],float)
    m=(np.asarray(v[4:9])>=.5).astype(int)
    R=p.R0*x/p.x0; B=tau*R; B0=p.tau0*p.R0
    available=B if p.budget_mode=='total' else max(B-B0,0.)
    uI,uE,uC=np.array([dI,dE,dC])*available
    aI=response(uI,p.eta_I,p.L_I,p.K_I)
    aE=response(uE,p.eta_E,p.L_E,p.K_E)
    aC=response(uC,p.eta_C,p.L_C,p.K_C)
    I=p.I0*(1+aI)
    xd=p.peak_factor*x/p.season_days; peak=xd/p.peak0
    raw_env=p.e1*x/p.x0+p.e2*peak*(1-p.power_effect*m[2])
    raw_soc=p.s_h*(x/p.x0)**2+p.s_n*peak*(1-p.traffic_effect*m[3])+p.s_noise*(1-p.alcohol_effect*m[1])
    env=raw_env*(1-aE)
    soc=raw_soc*(1-aC)
    # Community affects social pressure directly. Acceptance inherits that channel;
    # no second additive credit for the same investment or free marketing uplift.
    latent=np.log(p.S0/(1-p.S0))-p.acceptance_slope*(soc-1.)
    S=float(expit(latent))
    infra=p.i_p*(x/I)**2+p.i_d*peak
    demand=p.x0*(tau/p.tau0)**p.epsilon*(I/p.I0)**p.alpha*(max(S,1e-12)/p.S0)**p.beta*np.prod(1+np.asarray(p.kappa)*m)
    maint_I=p.maintenance_I*p.maintenance_multiplier
    maint_C=p.maintenance_C*p.maintenance_multiplier
    cost=float(m@np.asarray(p.measure_costs))*p.measure_cost_multiplier
    # Total-budget case funds basic services inside its budget. Incremental case
    # assumes unchanged basic services remain funded by baseline revenues.
    maint_charged=maint_I+maint_C if p.budget_mode=='total' else 0.
    return dict(x=x,xd=xd,R=R,B=B,B_incremental=B-B0,available=available,I=I,S=S,
        C_env=env,C_soc=soc,C_inf=infra,C_env_raw=raw_env,C_soc_raw=raw_soc,demand=demand,
        spend_I=uI,spend_E=uE,spend_C=uC,measure_cost=cost,
        maintenance_I=maint_I,maintenance_C=maint_C,maintenance_charged=maint_charged,
        total_I=uI+maint_I,total_E=uE,total_C=uC+maint_C,
        capacity_gain=aI,eco_reduction=aE,community_reduction=aC)


def evaluate(v,p):
    """Solve demand equality; all hard constraint margins independently exported."""
    tau,dI,dE,dC=np.asarray(v[:4],float)
    m=(np.asarray(v[4:9])>=.5).astype(int)
    x=brentq(lambda x:x-state_at(x,v,p)['demand'],0.,10000.,xtol=1e-9)
    r=state_at(x,v,p)
    r.update(tau=tau,delta_I=dI,delta_E=dE,delta_C=dC,**{f'm{i+1}':int(b) for i,b in enumerate(m)})
    r['demand_residual']=x-r['demand']
    r['ships_required_proxy']=int(np.ceil(r['xd']/p.passengers_per_ship_10k-1e-10))
    used=r['spend_I']+r['spend_E']+r['spend_C']+r['measure_cost']+r['maintenance_charged']
    r['budget_used']=used;r['budget_limit']=p.delta_total*r['available']
    margins=dict(capacity=(r['I']-x)/p.x0,ecology=p.E_max-r['C_env'],
        daily_cap=p.peak0-p.cap_reduction*m[0]-r['xd'],
        ship_capacity=p.ship_limit*p.passengers_per_ship_10k-r['xd'],
        physical_berths=p.physical_berths-p.ship_limit,annual_daily_link=r['xd']-x/p.season_days,
        budget=(r['budget_limit']-used)/(p.tau0*p.R0),allocation=p.delta_total-dI-dE-dC,
        acceptance_lower=r['S'],acceptance_upper=1-r['S'],tax_lower=tau-p.tau_min,tax_upper=p.tau_max-tau,
        allocation_lower=min(dI,dE,dC),allocation_upper=1-max(dI,dE,dC))
    r['soft_penalty']=p.penalty_S*max(p.S_min-r['S'],0)+p.penalty_R*max((p.R_min-r['R'])/p.R0,0)+p.penalty_H*max(r['C_soc']-p.H_max,0)
    r['max_violation']=max(0.,-min(margins.values()))
    r['feasible']=r['max_violation']<=1e-7 and abs(r['demand_residual'])<1e-7
    r.update({f'margin_{k}':v for k,v in margins.items()})
    f=np.array([-r['R']/p.R0,r['C_env'],r['C_soc'],r['C_inf']])
    return r,f,-np.array(list(margins.values()))


class TourismProblem(ElementwiseProblem):
    def __init__(self,p):
        self.p=p
        super().__init__(n_var=9,n_obj=4,n_ieq_constr=14,xl=np.array([p.tau_min,0,0,0,0,0,0,0,0]),xu=np.array([p.tau_max,1,1,1,1,1,1,1,1]))
    def _evaluate(self,v,out,*args,**kwargs):
        r,f,g=evaluate(v,self.p); out['F']=f; out['G']=g

def solve(p,seed=42,pop=160,generations=120):
    rng=np.random.default_rng(seed)
    initial=rng.random((pop,9)); initial[:,0]=rng.uniform(p.tau_min,p.tau_max,pop)
    initial[:,1:4]=rng.dirichlet([1,1,1],pop)*p.delta_total*rng.uniform(.2,.8,(pop,1))
    for i in range(pop): initial[i,4:]=list(itertools.product([0,1],repeat=5))[i%32]
    result=evolutionary_minimize(TourismProblem(p),NSGA2(pop_size=pop,sampling=initial),('n_gen',generations),seed=seed,verbose=False)
    if result.X is None: return []
    rows=[]
    for v in np.atleast_2d(result.X):
        r,_,_=evaluate(v,p)
        if r['feasible']: rows.append(r)
    return rows

def vector(r): return np.array([r[k] for k in ['tau','delta_I','delta_E','delta_C','m1','m2','m3','m4','m5']])

def weights_and_rank(rows,p):
    f=np.array([[-r['R']/p.R0,r['C_env'],r['C_soc'],r['C_inf']] for r in rows])
    keep=NonDominatedSorting().do(f,only_non_dominated_front=True)
    rows=[rows[i] for i in keep]; f=f[keep]
    span=np.ptp(f,axis=0); active=span>1e-10
    benefit=np.zeros_like(f); benefit[:,active]=(f[:,active].max(axis=0)-f[:,active])/span[active]
    proportions=(benefit+1e-12)/(benefit+1e-12).sum(axis=0)
    entropy=-(proportions*np.log(proportions)).sum(axis=0)/np.log(max(len(rows),2))
    information=np.where(active,np.maximum(0,1-entropy),0)
    ew=information/information.sum() if information.sum()>1e-12 else np.full(4,.25)
    ahp=np.asarray(p.ahp_weights); ahp=ahp/ahp.sum()
    # A reciprocal consistent matrix constructed from declared preferences, NOT expert elicitation.
    matrix=ahp[:,None]/ahp[None,:]
    w=p.theta*ahp+(1-p.theta)*ew
    for r,ff in zip(rows,f): r['J']=float(-w@ff-r['soft_penalty'])
    return sorted(rows,key=lambda r:r['J'],reverse=True),dict(combined=w.tolist(),entropy=ew.tolist(),ahp=ahp.tolist(),ahp_matrix=matrix.tolist(),consistency_ratio=0.)

def refine(p,weights,starts):
    """32 binary combinations, explicit deficit variables for soft constraints.

    Epigraph variables remove max() kinks from the SLSQP objective. The reported
    score always recomputes the true shortfalls, not the auxiliary variables.
    """
    rows=[];logs=[];ww=np.asarray(weights)
    penalties=np.array([p.penalty_S,p.penalty_R,p.penalty_H])
    for combo in itertools.product([0,1],repeat=5):
        best=None
        for start in starts:
            start=np.asarray(start)[:4]
            def ev(z):return evaluate(np.r_[z[:4],combo],p)
            def gaps(r):return np.array([p.S_min-r['S'],(p.R_min-r['R'])/p.R0,r['C_soc']-p.H_max])
            r0,_,_=ev(start)
            initial=np.r_[start,np.maximum(gaps(r0),0)+1e-6]
            def objective(z):
                r,f,g=ev(z);return float(ww@f+penalties@z[4:])
            def constraints(z):
                r,f,g=ev(z);return np.r_[-g,z[4:]-gaps(r)]
            opt=minimize(objective,initial,method='SLSQP',bounds=[(p.tau_min,p.tau_max)]+[(0,1)]*3+[(0,None)]*3,
                constraints={'type':'ineq','fun':constraints},options={'maxiter':250,'ftol':1e-11})
            r,f,g=ev(opt.x)
            logs.append(dict(combo=''.join(map(str,combo)),success=bool(opt.success),message=str(opt.message),max_violation=r['max_violation'],epigraph_violation=max(0.,-float(np.min(constraints(opt.x))))))
            if opt.success and r['feasible']:
                r['J']=float(-ww@f-r['soft_penalty'])
                if best is None or r['J']>best['J']:best=r
        if best:rows.append(best)
    return sorted(rows,key=lambda r:r['J'],reverse=True),logs
