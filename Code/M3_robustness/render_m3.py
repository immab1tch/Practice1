"""English paper-ready figures for M3, rendered from saved CSV (no re-optimization)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';FIG=ROOT/'figures'
CASES={'default':'Recommended','no_maintenance':'Maintenance zeroed','incremental':'Incremental budget only','no_community_effect':'Community benefit off','no_investment_effects':'All investment benefits off','strict_acceptance':'Acceptance target 0.9','economic_priority':'Economy priority','ecological_priority':'Ecology priority','community_priority':'Community priority','new_entropy_weights':'Updated entropy weights'}
LABELS={'epsilon':'Tax demand elasticity','eta_I':'Infrastructure efficiency','eta_E':'Ecology efficiency','eta_C':'Community efficiency','K_I':'Infrastructure saturation scale','K_E':'Ecology saturation scale','K_C':'Community saturation scale','acceptance_slope':'Resident response strength','peak_factor':'Peak-season concentration','measure_cost_multiplier':'Measure cost','R0':'Revenue baseline (calibration)','S_min':'Acceptance soft target'}
COLORS=['#286187','#C38F32','#758D66','#BA6985','#A0A6AD','#D8DCE0','#F1F2F3']
def read(n):return pd.read_csv(OUT/(n+'.csv'))
def render():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'path'})
    entries=[]
    def finish(fig,name,title,source,note):
        fig.suptitle(title,fontsize=16,fontweight='bold',y=.98)
        fig.text(.025,.018,note,fontsize=9,color='#555555')
        fig.tight_layout(rect=(.01,.09,.99,.92))
        fig.savefig(FIG/(name+'.png'),dpi=300);fig.savefig(FIG/(name+'.svg'));plt.close(fig)
        entries.append({'file':name,'source':source,'note':note})
    rank=read('M3_sensitivity_ranking');sens=read('M3_sensitivity');order=rank.parameter.tolist()
    fig,axes=plt.subplots(1,2,figsize=(13,7.2),sharex=True)
    for ax,method,title in zip(axes,['fixed','reoptimized'],['Fixed recommended policy','Re-optimized policy (round-2 solve reused)']):
        for j,factor in enumerate([.9,1.1]):
            sub=sens[(sens.method==method)&(sens.factor==factor)].set_index('parameter').loc[order]
            y=np.arange(len(order))+(j-.5)*.32
            ax.barh(y,sub.delta_J,height=.28,color=COLORS[j],label=f'Parameter x {factor}')
            for yy,(_,r) in zip(y,sub.iterrows()):
                if not r.feasible:ax.scatter(r.delta_J,yy,marker='x',color='black',zorder=5)
        ax.set_yticks(range(len(order)),[LABELS[k] for k in order]);ax.invert_yaxis();ax.axvline(0,c='#777777',lw=.8);ax.set_xlabel('Score change dJ (relative to round-2 recommendation)');ax.set_title(title,fontsize=12);ax.legend(fontsize=9)
    finish(fig,'M3_01_Sensitivity','M3: Ranked by maximum absolute score change after re-optimization','M3_sensitivity.csv; M3_sensitivity_ranking.csv','x marks an unmet hard constraint; changing the revenue baseline shifts revenue, budget and the soft penalty. The ranking is not causal importance or a Sobol index.')
    stress=read('M3_stress_samples');viol=read('M3_constraint_violations');positive=viol[viol.violations>0]
    fig,axes=plt.subplots(1,2,figsize=(11,5.3))
    names={'budget':'Budget','daily_cap':'Daily peak cap','ship_capacity':'Ship capacity','capacity':'Annual capacity','ecology':'Ecology threshold'}
    bars=axes[0].barh([names.get(k,k) for k in positive.constraint],positive.violation_fraction*100,color=COLORS[3]);axes[0].bar_label(bars,fmt='%.1f%%',padding=4);axes[0].invert_yaxis();axes[0].set_xlim(0,100);axes[0].set_xlabel('Share of samples violating the constraint (%)');axes[0].set_title('Constraints can be violated simultaneously',fontsize=12)
    counts=[int(stress.feasible.sum()),int((~stress.feasible).sum())];bars=axes[1].bar(['All hard constraints met','At least one not met'],counts,color=[COLORS[0],COLORS[3]]);axes[1].bar_label(bars,labels=[f'{c}/600\n{c/6:.2f}%' for c in counts],padding=5);axes[1].set_ylim(0,600);axes[1].set_ylabel('Number of stress scenarios');axes[1].set_title('Fixed recommended policy, equilibria re-solved',fontsize=12)
    finish(fig,'M3_02_Stress_test','M3: Feasibility across 600 hypothetical perturbation scenarios','M3_stress_samples.csv; M3_constraint_violations.csv','Independent uniform sampling ranges are reused from round 2; the result is scenario coverage, not a real-world policy success probability or confidence interval.')
    pref=read('M3_preference_scenarios').set_index('case').loc[['default','economic_priority','ecological_priority','community_priority']]
    fig,axes=plt.subplots(1,2,figsize=(12,5.9));left=np.zeros(len(pref))
    for k,label,color in zip('IEC',['Infrastructure','Ecology','Community'],COLORS):
        vals=pref['spend_'+k].to_numpy()*100;axes[0].barh(range(len(pref)),vals,left=left,label=label,color=color);left+=vals
    axes[0].set_yticks(range(len(pref)),[CASES[k] for k in pref.index]);axes[0].invert_yaxis();axes[0].set_xlabel('New improvement spending (million USD/yr)');axes[0].legend(fontsize=9,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.18),frameon=False)
    for j,(key,label) in enumerate([('C_env','Environmental pressure'),('C_soc','Social pressure'),('C_inf','Infrastructure pressure')]):axes[1].plot(range(len(pref)),pref[key],marker=['o','s','^'][j],color=COLORS[j],label=label)
    axes[1].set_xticks(range(len(pref)),[CASES[k] for k in pref.index]);axes[1].set_ylabel('Pressure index (lower is better)');axes[1].set_ylim(0,1.05);axes[1].legend(fontsize=9)
    finish(fig,'M3_03_Preference_tradeoff','M3: Preferences change funding allocation and pressure outcomes','M3_preference_scenarios.csv','Different preferences use different scoring weights, so J levels across scenarios are not ranked against each other. Lines only help identify the same indicator.')
    seeds=read('M3_search_stage_gaps');ref=seeds.refined_reference_J.iloc[0]
    fig,axes=plt.subplots(1,2,figsize=(12,5.7));y=np.arange(3)
    axes[0].scatter(seeds.best_J,y,s=70,color=COLORS[0],label='Highest post-hoc J among NSGA-II candidates')
    axes[0].axvline(ref,color=COLORS[1],ls='--',label='Refined by enumerating 32 combos with multi-start')
    for yy,j in zip(y,seeds.best_J):axes[0].annotate(f'{j:.5f}',(j,yy),xytext=(4,10),textcoords='offset points',fontsize=10)
    axes[0].set_yticks(y,[f'seed {s}' for s in seeds.seed]);axes[0].set_ylim(-.5,2.5);axes[0].invert_yaxis();axes[0].set_xlim(min(seeds.best_J)-.005,ref+.005);axes[0].set_xlabel('Fixed-weight score J (higher is better, local scale)');axes[0].set_title(f'Two solve stages; refined reference J={ref:.5f}',fontsize=12)
    axes[0].legend(loc='upper left',bbox_to_anchor=(-.02,-.20),fontsize=9,frameon=False)
    bars=axes[1].barh(y,seeds.absolute_score_gap,color=COLORS[0]);axes[1].bar_label(bars,fmt='%.5f',padding=4);axes[1].set_yticks(y,[f'seed {s}' for s in seeds.seed]);axes[1].invert_yaxis();axes[1].set_xlim(0,seeds.absolute_score_gap.max()*1.35);axes[1].set_xlabel('Refined reference J minus random candidate J (absolute gap)');axes[1].set_title('No improvement percentage from negative scores',fontsize=12)
    finish(fig,'M3_04_Search_stage_gap','Figure 13 revision: stage gap between random candidates and added refinement','M3_search_stage_gaps.csv','NSGA-II searches four-objective Pareto candidates with J scored afterwards; the two stages differ in computation and per-seed refinement is absent. This figure cannot prove convergence.')
    floors=read('M3_income_floor_sweep')
    key='R_min' if 'R_min' in floors else 'floor'
    if key not in floors:raise ValueError(f'Unexpected floor schema: {list(floors.columns)}')
    fig,ax=plt.subplots(figsize=(9,5.2));ax.plot(floors[key],floors.x,'o-',color=COLORS[0],label='Re-optimized under each income soft target')
    ax.plot(floors[key],160*floors[key]/3.75,'--',color=COLORS[1],label='Visitor count when income target is exactly met');ax.scatter([3.45],[147.2],s=90,color=COLORS[3],zorder=5);ax.annotate('345 million USD -> 1.472 million visitors',(3.45,147.2),xytext=(12,-40),textcoords='offset points',arrowprops={'arrowstyle':'-','color':'#777777'},fontsize=10)
    ax.set_xlabel('Income soft target (100 million USD)');ax.set_ylabel('Annual visitors (10,000)');ax.legend(fontsize=9)
    finish(fig,'M3_05_Income_floor_check','M3 supplement: why 1.472 million visitors occurs','M3_income_floor_sweep.csv; inputs/income_findings.json','Reuses the completed revenue-floor audit; the curve shows piecewise response and measure switching, and 1.472 million is not an independently estimated carrying-capacity ceiling.')
    pd.DataFrame(entries).to_csv(OUT/'figure_index_en.csv',index=False,encoding='utf-8-sig')
    print('M3 figures rendered:',len(entries))

if __name__=='__main__':render()
