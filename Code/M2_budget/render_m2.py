"""English paper-ready figures for M2, rendered from saved CSV (no re-optimization)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';FIG=ROOT/'figures'
CASES={'default':'Recommended','no_maintenance':'Maintenance zeroed','incremental':'Incremental budget only','no_community_effect':'Community benefit off','no_investment_effects':'All investment benefits off','strict_acceptance':'Acceptance target 0.9'}
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
    sc=read('M2_scenarios').set_index('case');ledger=read('M2_budget_ledger')
    fig,ax=plt.subplots(figsize=(11,5.5));cases=['default','no_maintenance','incremental'];left=np.zeros(3)
    cats=['infrastructure_new','ecology_new','community_new','measures','maintenance_charged','available_unallocated','baseline_retained']
    labs=['Infrastructure (new)','Ecology (new)','Community (new)','Measures (annualized)','Maintenance charged to budget','Available funds unallocated','Baseline funds (retained)']
    for cat,label,color in zip(cats,labs,COLORS):
        vals=np.array([ledger[(ledger.case==c)&(ledger.category==cat)].million_usd.iloc[0] for c in cases])
        ax.barh(np.arange(3),vals,left=left,label=label,color=color,edgecolor='white',height=.58)
        for y,v,l in zip(range(3),vals,left):
            if v>2.5:ax.text(l+v/2,y,f'{v:.2f}',ha='center',va='center',fontsize=10,color='white' if cat in cats[:4] else '#333333')
        left+=vals
    ax.set_yticks(range(3),[CASES[c] for c in cases]);ax.invert_yaxis();ax.set_xlabel('Million USD per year (row totals equal tax revenue B)');ax.set_xlim(0,max(left)*1.03)
    ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.2),fontsize=9,frameon=False)
    finish(fig,'M2_01_Budget_breakdown','M2: Allocation of tax revenue and accounting of available funds','M2_budget_ledger.csv','Incremental-only scenario assumes baseline funds keep funding existing services and maintenance, and are not part of the new allocation. Measure costs are annualized.')
    fig,axes=plt.subplots(1,3,figsize=(12,4.8))
    sub=read('M2_feedback');sub=sub[sub.case=='default'].set_index('channel')
    for ax,key,label,color in zip(axes,['infrastructure','ecology','community'],['Infrastructure capacity (10k visitors/yr)','Environmental pressure index','Social pressure index'],COLORS):
        r=sub.loc[key];bars=ax.bar(['Before investment','After investment'],[r.before_investment,r.after_investment],color=['#D0D9DF',color]);ax.bar_label(bars,fmt='%.3f',padding=4);ax.set_ylim(0,max(r.before_investment,r.after_investment)*1.28);ax.set_title(label,fontsize=12)
        ax.text(.5,.9,('Increase ' if key=='infrastructure' else 'Reduction ')+f'{100*r.relative_effect:.2f}%',transform=ax.transAxes,ha='center')
    finish(fig,'M2_02_Feedback_decomposition','M2: Direct investment feedback at the same equilibrium','M2_feedback.csv','Decomposition fixed at the recommended equilibrium; not an equilibrium comparison with another policy, and not a measured abatement or approval rate.')
    pd.DataFrame(entries).to_csv(OUT/'figure_index_en.csv',index=False,encoding='utf-8-sig')
    print('M2 figures rendered:',len(entries))

if __name__=='__main__':render()
