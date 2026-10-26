import unittest
from dataclasses import replace
import numpy as np
from model import Params,validate,evaluate,state_at,response,solve,weights_and_rank
class Round2Tests(unittest.TestCase):
    def setUp(self):self.p=Params();self.zero=np.array([.11,0,0,0,0,0,0,0,0],float)
    def test_baseline(self):
        r,_,_=evaluate(self.zero,self.p)
        for k,t in [('x',160),('R',3.75),('B',.4125),('xd',2),('S',.7),('C_soc',1),('C_env',1),('C_inf',1)]:self.assertAlmostEqual(r[k],t,places=7)
        self.assertTrue(r['feasible'])
    def test_diminishing_returns_and_bounds(self):
        for k in 'IEC':
            eta,L,K=[getattr(self.p,x+'_'+k) for x in ['eta','L','K']]
            y=response(np.linspace(0,.8,101),eta,L,K)
            self.assertEqual(y[0],0);self.assertTrue((np.diff(y)>0).all());self.assertTrue((np.diff(y,2)<0).all());self.assertTrue((y<eta*L).all())
    def test_community_direct_effect_and_no_clip(self):
        a=state_at(160,self.zero,self.p);z=self.zero.copy();z[3]=.3;b=state_at(160,z,self.p)
        self.assertLess(b['C_soc'],a['C_soc']);self.assertGreater(b['S'],a['S']);self.assertLess(b['S'],1)
        z[3]=.4;c=state_at(160,z,self.p);self.assertGreater(c['S'],b['S'])
    def test_baseline_maintenance_not_investment(self):
        a=state_at(160,self.zero,self.p);b=state_at(160,self.zero,replace(self.p,maintenance_multiplier=0))
        for k in ['S','I','C_env','C_soc','C_inf']:self.assertEqual(a[k],b[k])
        self.assertAlmostEqual(a['maintenance_charged'],.012553)
    def test_incremental_no_double_charge(self):
        r,_,_=evaluate(self.zero,replace(self.p,budget_mode='incremental'))
        self.assertAlmostEqual(r['available'],0,places=7);self.assertEqual(r['maintenance_charged'],0)
        self.assertGreater(r['maintenance_I'],0)
    def test_positive_environment_no_negative_credit(self):
        z=self.zero.copy();z[2]=1
        r=state_at(160,z,self.p);self.assertGreater(r['C_env'],0)
    def test_crowding_lowers_acceptance(self):
        a=state_at(160,self.zero,self.p);b=state_at(180,self.zero,self.p);self.assertLess(b['S'],a['S'])
    def test_overbudget_rejected(self):
        z=[.11,.3,.3,.2,1,1,1,1,1];r,_,_=evaluate(z,self.p)
        self.assertFalse(r['feasible']);self.assertLess(r['margin_budget'],0)
    def test_random_equilibrium_accounting(self):
        rng=np.random.default_rng(15)
        for i in range(80):
            v=np.r_[rng.uniform(.09,.15),rng.dirichlet([1,1,1])*.6,rng.integers(0,2,5)]
            r,_,_=evaluate(v,self.p)
            self.assertLess(abs(r['demand_residual']),1e-7)
            self.assertAlmostEqual(r['budget_used'],sum(r[k] for k in ['spend_I','spend_E','spend_C','measure_cost','maintenance_charged']))
            self.assertAlmostEqual(r['xd'],self.p.peak_factor*r['x']/self.p.season_days)
    def test_reject_invalid_response_scale(self):
        with self.assertRaises(ValueError):validate(replace(self.p,K_C=0))
        with self.assertRaises(ValueError):validate(replace(self.p,eta_C=2))
    def test_constant_entropy(self):
        r,_,_=evaluate(self.zero,self.p);_,w=weights_and_rank([r.copy(),r.copy()],self.p)
        self.assertTrue(np.isfinite(w['combined']).all());self.assertAlmostEqual(sum(w['combined']),1)
    def test_optimizer_returns_feasible(self):
        rows=solve(self.p,7,40,10);self.assertTrue(rows)
        for r in rows:self.assertTrue(r['feasible'])
if __name__=='__main__':unittest.main()
