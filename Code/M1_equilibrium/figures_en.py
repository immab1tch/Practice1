"""English paper figures (PNG + SVG, 300 dpi); every figure reads exported evidence.
Paper-ready English labels; run via redraw_en.py (reads existing CSVs, no model rerun)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

COLORS=['#286187','#BA8B33','#7A8560','#B96583','#888888']
CAT=['Infrastructure','Ecology','Community']
LABELS={'epsilon':'Tax demand elasticity','eta_I':'Infrastructure efficiency','eta_E':'Ecology efficiency','eta_C':'Community efficiency','K_I':'Infra. saturation scale','K_E':'Ecology saturation scale','K_C':'Community saturation scale','acceptance_slope':'Acceptance responsiveness','peak_factor':'Peak concentration','measure_cost_multiplier':'Measure cost','R0':'Baseline revenue','S_min':'Acceptance floor'}

def create_figures(out,root,p):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':14,'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'path'})
    figdir=out/'figures';figdir.mkdir(exist_ok=True);entries=[]
    def load(name):return pd.read_csv(out/(name+'.csv'))
    best=load('recommended').iloc[0];old=pd.read_csv(root/'baseline_round1/recommended.csv').iloc[0]
    sc=load('scenarios').set_index('case');comp=load('comparable_policies').set_index('case');front=load('pareto');sens=load('sensitivity');curves=load('response_curves')
    def finish(fig,name,title,source,note):
        fig.suptitle(title,y=.99,fontweight='bold');fig.text(.015,.012,note,fontsize=8.5,color='#555555')
        fig.tight_layout(rect=(0,.06,1,.94))
        fig.savefig(figdir/(name+'.png'),dpi=300);fig.savefig(figdir/(name+'.svg'));plt.close(fig)
        entries.append(dict(file=name,title=title,source=source,note=note))
    # 01
    fig,ax=plt.subplots(figsize=(9,5.5));x=np.arange(3)
    a=np.array([old['spend_'+k]*100 for k in 'IEC']);b=np.array([best['spend_'+k]*100 for k in 'IEC'])
    ba=ax.bar(x-.19,a,.36,label='Round 1: new investment',color='#B5C8D5');bb=ax.bar(x+.19,b,.36,label='Round 2: new investment',color=COLORS[0])
    ax.bar_label(ba,fmt='%.2f',padding=3);ax.bar_label(bb,fmt='%.2f',padding=3);ax.set_xticks(x,CAT);ax.set_ylabel('Million USD per year');ax.set_ylim(0,max(a.max(),b.max())*1.16);ax.legend()
    finish(fig,'01_new_investment_rounds','New investment allocation across rounds','baseline_round1/recommended.csv; recommended.csv','Same scoring weights; the two rounds use different response structures. Amounts only; pressure indices are not differenced across models.')
    # 02
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    keys=['x','R','C_env','C_soc','C_inf','S'];titles=['Annual visitors (10k)','Tourism revenue (100M USD)','Environmental pressure','Social pressure','Infrastructure pressure','Resident acceptance']
    for ax,k,title in zip(axes.flat,keys,titles):
        vals=[comp.loc['round1_policy_in_round2',k],best[k]];bars=ax.bar(['Old policy in new model','Round 2 solution'],vals,color=['#B5C8D5',COLORS[0]])
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=10);ax.set_title(title,fontsize=12);ax.set_ylim(0,max(vals)*1.25)
    oldfeas=bool(comp.loc['round1_policy_in_round2','feasible'])
    finish(fig,'02_same_model_comparison','Same fixed model: compare policies, not yardsticks','comparable_policies.csv',f'Old policy is feasible under the new hard constraints: {oldfeas}; an infeasible counterfactual is not an executable alternative.')
    # 03
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for ax,k,title in zip(axes,['C_env','C_soc','C_inf'],['Environmental pressure','Social pressure','Infrastructure pressure']):
        ax.scatter(front.R*100,front[k],s=12,c=COLORS[0],alpha=.35)
        ax.scatter(best.R*100,best[k],s=150,c=COLORS[3],marker='*',label='Round 2 recommended')
        ax.set(xlabel='Tourism revenue (million USD)',ylabel=title);ax.legend(fontsize=9)
    finish(fig,'03_pareto_projections','Three projections of the four-objective near-nondominated set','pareto.csv; recommended.csv','Points may overlap in a projection yet remain nondominated in four objectives; random search + local refinement does not prove global optimality.')
    # 04
    fig,axes=plt.subplots(1,3,figsize=(13,4.7))
    for ax,k,label,color in zip(axes,'IEC',CAT,COLORS):
        c=curves[curves.category==k];ax.plot(c.spend_million,c.response*100,color=color,lw=2.3)
        spend=best['spend_'+k]*100;gain=best[{'I':'capacity_gain','E':'eco_reduction','C':'community_reduction'}[k]]*100
        ax.scatter(spend,gain,color=color,s=55);ax.axvline(spend,color='#666666',ls=':',lw=1)
        ax.set(title=label,xlabel='New investment (million USD/year)',ylabel='Capacity gain (%)' if k=='I' else 'Raw-pressure reduction (%)');ax.set_ylim(0,100*max(p.eta_I*p.L_I,p.eta_E*p.L_E,p.eta_C*p.L_C)*1.1)
    finish(fig,'04_diminishing_response','Investment response curves and recommended spending','response_curves.csv; recommended.csv','Curve parameters are transparent scenario assumptions, not fitted to data; reduction ratios are not glacier retreat or satisfaction rates.')
    # 05
    ledger=[('Infrastructure maintenance',best.maintenance_I*100),('Community basic services',best.maintenance_C*100),('New infrastructure',best.spend_I*100),('New ecology',best.spend_E*100),('New community',best.spend_C*100),('Measure cost',best.measure_cost*100),('Unallocated receipts',(best.available-best.budget_used)*100)]
    pd.DataFrame(ledger,columns=['category','million_usd']).to_csv(out/'budget_ledger.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(10,5.4));bars=ax.barh([r[0] for r in ledger],[r[1] for r in ledger],color=[COLORS[4],COLORS[4],COLORS[0],COLORS[1],COLORS[2],COLORS[3],COLORS[4]])
    ax.bar_label(bars,fmt='%.3f',padding=3);ax.invert_yaxis();ax.set_xlabel('Million USD per year');ax.set_xlim(0,max(v for _,v in ledger)*1.23)
    finish(fig,'05_budget_ledger','Total-receipt scenario: maintenance, new investment, measures separately booked','budget_ledger.csv','Annualised program cost is not first-year cash need; unallocated receipts include the reserved 20% and are not all discretionary.')
    # 06
    fig,ax=plt.subplots(figsize=(9,4.9));ratios=[best.x/best.I,best.xd/(p.peak0-p.cap_reduction*best.m1),best.budget_used/best.budget_limit,best.C_env/p.E_max]
    bars=ax.barh(['Annual capacity','Daily peak ceiling','Allocatable budget','Ecology threshold'],np.array(ratios)*100,color=COLORS[0]);ax.bar_label(bars,fmt='%.1f%%',padding=3);ax.axvline(100,color='#333333',ls='--',lw=1);ax.set_xlim(0,118);ax.invert_yaxis();ax.set_xlabel('Usage relative to constraint limit (%)')
    finish(fig,'06_constraint_usage','Four resource/constraint usage ratios of the recommended solution','recommended.csv; parameters_used.json','The ecology cap is a scenario threshold, not a proven safe line; all raw constraint margins are in constraint_audit.csv.')
    # 07
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    order=list(LABELS)
    for ax,method,title in zip(axes,['fixed','reoptimized'],['Policy held fixed','Policy re-optimized']):
        sub=sens[sens.method==method]
        for j,factor in enumerate([.9,1.1]):
            vals=[];flags=[]
            for key in order:
                row=sub[(sub.parameter==key)&(sub.factor==factor)].iloc[0];vals.append(row.J-best.J);flags.append(bool(row.feasible))
            yy=np.arange(len(order))+(j-.5)*.3
            ax.barh(yy,vals,height=.27,color=[COLORS[0],COLORS[1]][j],label=['Parameter x0.9','Parameter x1.1'][j])
            for y,v,flag in zip(yy,vals,flags):
                if not flag and np.isfinite(v):ax.scatter(v,y,c='#333333',marker='x',s=35,zorder=4)
        ax.set_yticks(np.arange(len(order)),[LABELS[k] for k in order]);ax.invert_yaxis();ax.axvline(0,color='#666666',lw=.8);ax.set_title(title);ax.set_xlabel('Score change vs. default solution');ax.legend(fontsize=9)
    delta=sens.J-best.J;lo=min(float(delta.min()),0);hi=max(float(delta.max()),0);pad=max((hi-lo)*.05,.005)
    for ax in axes:ax.set_xlim(lo-pad,hi+pad)
    finish(fig,'07_sensitivity_scores','Single-parameter +/-10%: fixed policy vs. re-optimization','sensitivity.csv','Both panels share the x-axis; x marks infeasible. Scores include soft penalties; negative elasticity x1.1 means a larger absolute value.')
    # 08
    re=sens[sens.method=='reoptimized'].copy();re['label']=[LABELS[k]+(' x0.9' if f<1 else ' x1.1') for k,f in zip(re.parameter,re.factor)]
    fig,ax=plt.subplots(figsize=(11,9));left=np.zeros(len(re))
    for k,label,col in zip('IEC',CAT,COLORS):
        vals=re['spend_'+k].to_numpy()*100;ax.barh(re.label,vals,left=left,label=label,color=col);left+=vals
    ax.invert_yaxis();ax.set_xlabel('New investment (million USD/year)');ax.legend(ncol=3,loc='lower right')
    finish(fig,'08_sensitivity_allocation','New investment allocation under parameter changes','sensitivity.csv','Each row is re-optimized; total budget moves with visitor volume and tax, so bar lengths need not match.')
    # 09
    cases=['default','economic_priority','ecological_priority','community_priority','new_entropy_weights'];names=['Frozen round-1 weights','Economy priority','Ecology priority','Community priority','Recomputed AHP-entropy']
    fig,ax=plt.subplots(figsize=(10,5));left=np.zeros(len(cases))
    for k,label,col in zip('IEC',CAT,COLORS):
        vals=sc.loc[cases,'spend_'+k].to_numpy()*100;ax.barh(names,vals,left=left,color=col,label=label);left+=vals
    ax.invert_yaxis();ax.set_xlabel('New investment (million USD/year)');ax.legend(ncol=3)
    finish(fig,'09_weight_scenarios','Investment structure under alternative preference weights','scenarios.csv','J is not comparable across weights; this compares each optimized amount. Positive investment is not a forced ratio.')
    # 10
    cases=['default','no_maintenance','incremental'];names=['Total-receipt budget','Maintenance reserve removed','Incremental receipts only']
    fig,axes=plt.subplots(1,2,figsize=(12,5));xx=np.arange(3)
    for j,(k,label,col) in enumerate(zip('IEC',CAT,COLORS)):
        vals=sc.loc[cases,'spend_'+k].to_numpy()*100;bars=axes[0].bar(xx+(j-1)*.23,vals,.22,color=col,label=label);axes[0].bar_label(bars,fmt='%.1f',fontsize=9,padding=3)
    axes[0].set_xticks(xx,names);axes[0].set_ylabel('New investment (million USD/year)');axes[0].legend();axes[0].set_ylim(0,sc.loc[cases,['spend_I','spend_E','spend_C']].max().max()*1.25)
    for k,label,col in zip(['C_env','C_soc','C_inf'],['Environmental','Social','Infrastructure'],COLORS):axes[1].plot(names,sc.loc[cases,k],marker='o',label=label,color=col,lw=2)
    axes[1].set_ylabel('Pressure index');axes[1].legend();axes[1].set_ylim(bottom=0)
    finish(fig,'10_budget_maintenance','Required maintenance vs. positive investment; fiscal space and results','scenarios.csv','The incremental-only scenario leaves existing maintenance to the prior budget; maintenance is not double-paid from new receipts.')
    # 11
    stress=load('stress_samples');flags=(stress.feasible==True);fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].hist(stress.x,bins=26,color=COLORS[0],alpha=.85);axes[0].axvline(best.x,c='#333333',ls='--',label='Default solution');axes[0].set(xlabel='Annual visitors (10k)',ylabel='Number of scenarios');axes[0].legend()
    constraints=['capacity','ecology','daily_cap','budget'];fractions=[(stress['margin_'+k]<-1e-7).mean()*100 for k in constraints]
    bars=axes[1].barh(['Annual capacity','Ecology threshold','Daily peak','Budget'],fractions,color=COLORS[1]);axes[1].bar_label(bars,fmt='%.1f%%',padding=3);axes[1].set_xlim(0,105);axes[1].set_xlabel('Share of samples violating the constraint (%)');axes[1].invert_yaxis()
    finish(fig,'11_joint_stress_test','600 joint-perturbation scenarios: fixed-policy robustness','stress_samples.csv',f'Independent uniform sampling; all-hard-constraint feasible share = {flags.mean():.1%}; this is not a real-world success rate or confidence interval. Violations may overlap.')
    # 12
    landscape=load('policy_landscape');landscape['plot_J']=landscape.J.where(landscape.feasible)
    pivot=landscape.pivot(index='transfer',columns='tau',values='plot_J').sort_index()
    fig,ax=plt.subplots(figsize=(9,6));cm=plt.get_cmap('cividis').copy();cm.set_bad('#eeeeee')
    im=ax.pcolormesh(pivot.columns.to_numpy()*100,pivot.index.to_numpy()*100,pivot.to_numpy(),shading='nearest',cmap=cm)
    ax.scatter(best.tau*100,0,marker='*',s=170,c='#E8A9BD',edgecolors='#333333',label='Recommended');ax.set(xlabel='Tax lever (%)',ylabel='Share moved from ecology to infrastructure (pct points)');ax.legend();fig.colorbar(im,ax=ax,label='Fixed-weight score J')
    finish(fig,'12_policy_landscape','Local policy slice over tax and fund transfer','policy_landscape.csv','Community share and measures held at recommended; grey is infeasible/undefined. A 2-D slice, not proof of global optimality.')
    # 13
    seeds=load('seed_comparison');fig,ax=plt.subplots(figsize=(8.5,4.7));xs=np.arange(len(seeds))
    ax.scatter(xs,seeds.best_J,color=COLORS[0],s=65,label='Best candidate for that seed');ax.axhline(best.J,color=COLORS[1],ls='--',label='Recommended after 32-combo multi-start refinement')
    ax.set_xticks(xs,['seed '+str(int(v)) for v in seeds.seed]);ax.set_ylabel('Fixed-weight score J');ax.legend()
    finish(fig,'13_seed_comparison','Random-search variation and local refinement','seed_comparison.csv; recommended.csv','A randomization check, not a convergence curve; three seeds do not prove global optimality.')
    # 14
    cases=['default','no_community_effect','no_investment_effects'];names=['Full feedback','Community benefit off','All investment effects off']
    fig,axes=plt.subplots(1,2,figsize=(12,5));left=np.zeros(len(cases))
    for k,label,col in zip('IEC',CAT,COLORS):
        vals=sc.loc[cases,'spend_'+k].to_numpy()*100;axes[0].barh(names,vals,left=left,label=label,color=col);left+=vals
    axes[0].invert_yaxis();axes[0].set_xlabel('New investment (million USD/year)');axes[0].legend()
    axes[1].plot(names,sc.loc[cases,'C_soc'],marker='o',color=COLORS[2],label='Social pressure');axes[1].plot(names,sc.loc[cases,'S'],marker='s',ls='--',color=COLORS[0],label='Resident acceptance');axes[1].set_ylim(0,1.1);axes[1].legend()
    finish(fig,'14_feedback_ablation','Re-optimization after disabling feedback channels','scenarios.csv','Zero-effect channels are set to zero spend among equal scores; this is not a minimum-spend constraint.')
    # 15
    from scipy.special import expit
    cgrid=np.linspace(0,1.6,161);acceptance=[]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for gamma,col in zip([1.,2.,3.],COLORS):
        ss=expit(np.log(p.S0/(1-p.S0))-gamma*(cgrid-1))
        axes[0].plot(cgrid,ss,color=col,label=f'Responsiveness {gamma:.0f}',lw=2)
        acceptance.extend(dict(social_pressure=c,acceptance=v,slope=gamma) for c,v in zip(cgrid,ss))
    pd.DataFrame(acceptance).to_csv(out/'acceptance_curve.csv',index=False,encoding='utf-8-sig')
    axes[0].scatter([best.C_soc],[best.S],marker='*',s=120,c='#333333');axes[0].set(xlabel='Social pressure index',ylabel='Resident acceptance',ylim=(0,1));axes[0].legend()
    cc=['default','strict_acceptance'];xx=np.arange(2)
    for j,(k,label,col) in enumerate(zip('IEC',CAT,COLORS)):
        vv=sc.loc[cc,'spend_'+k].to_numpy()*100;bars=axes[1].bar(xx+(j-1)*.23,vv,.22,label=label,color=col);axes[1].bar_label(bars,fmt='%.1f',fontsize=9,padding=3)
    axes[1].set_xticks(xx,['Soft floor 0.7','Soft floor 0.9']);axes[1].set_ylabel('New investment (million USD/year)');axes[1].legend();axes[1].set_ylim(0,sc.loc[cc,['spend_I','spend_E','spend_C']].max().max()*1.25)
    finish(fig,'15_acceptance_response','Smooth resident response and the stricter acceptance target','acceptance_curve.csv; scenarios.csv','The index is not a satisfaction rate; 0.9 is still a soft target; check S and soft_penalty in scenarios.csv.')
    pd.DataFrame(entries).to_csv(out/'figure_index_en.csv',index=False,encoding='utf-8-sig')
