"""English paper figures for the revenue-floor audit (PNG + SVG, 300 dpi).
English labels; reads existing CSVs. Run: python report_en.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE='#286187';GOLD='#BA8B33';PINK='#B96583';GRAY='#888888'

def render(root):
    out=root/'results';figdir=out/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'path'})
    floor=pd.read_csv(out/'floor_sweep.csv',dtype={'combo':str});pen=pd.read_csv(out/'penalty_sweep.csv');prof=pd.read_csv(out/'visitor_profile.csv')
    counter=pd.read_csv(out/'counterfactuals.csv',dtype={'combo':str}).set_index('case');base=counter.loc['default'];free=counter.loc['income_penalty_removed']
    cross=pd.read_csv(out/'independent_optimizer.csv');p=json.loads((root/'inputs/parameters_used.json').read_text());w=json.loads((root/'inputs/weights_used.json').read_text())['weights']
    entries=[]
    def save(fig,name,title,note,source):
        fig.suptitle(title,y=.99,fontweight='bold',fontsize=15);fig.text(.015,.014,note,color='#555555',fontsize=9)
        fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(figdir/(name+'.png'),dpi=300);fig.savefig(figdir/(name+'.svg'));plt.close(fig)
        entries.append(dict(name=name,title=title,note=note,source=source))
    fig,axes=plt.subplots(1,2,figsize=(13,5.4))
    f=floor[floor.R_min>=2.9]
    axes[0].plot(f.R_min,f.x,'o-',color=BLUE,lw=1.8,label='Re-optimized visitors')
    axes[0].plot(f.R_min,f.floor_x,'--',color=GOLD,label='Exactly meeting revenue target: Rmin/p0')
    axes[0].scatter([base.R_min],[base.x],c=PINK,marker='*',s=180,zorder=5,label='Round 2 default')
    axes[0].axhline(160,color=GRAY,ls=':',label='Annual ceiling from base daily peak')
    axes[0].set(xlabel='Soft revenue floor (100M USD/year)',ylabel='Annual visitors (10k)',ylim=(120,175));axes[0].legend(fontsize=9)
    local=f[f.R_min.between(3.39,3.6)]
    axes[1].plot(local.R_min,local.x,'o-',color=BLUE,label='Re-optimized')
    axes[1].plot(local.R_min,local.floor_x,'--',color=GOLD,label='Rmin/p0')
    axes[1].scatter([base.R_min],[base.x],c=PINK,marker='*',s=180,zorder=5)
    axes[1].set(xlabel='Soft revenue floor (100M USD/year)',ylabel='Annual visitors (10k)',title='Zoom near 1.472 million visitors');axes[1].legend()
    save(fig,'01_revenue_floor_sweep','Varying the revenue floor, other parameters and weights fixed','Each point re-selects among 32 measure combos and optimizes continuous variables; the line joins sampled points only.','floor_sweep.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5.1))
    axes[0].plot(pen.penalty_R,pen.x,'o-',color=BLUE);axes[0].axhline(base.x,c=GRAY,ls='--');axes[0].set_xscale('symlog',linthresh=.005);axes[0].set(xlabel='Revenue-gap penalty coefficient lambda_R (symlog incl. 0)',ylabel='Annual visitors (10k)')
    axes[1].plot(pen.penalty_R,pen.income_gap*100,'o-',c=GOLD);axes[1].set_xscale('symlog',linthresh=.005);axes[1].set(xlabel='Revenue-gap penalty coefficient lambda_R',ylabel='Unmet revenue target (million USD/year)');axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlim(0,12);ax.set_xticks([0,.01,.1,1,10],['0','0.01','0.1','1','10'])
    save(fig,'02_penalty_sweep','Same 345M USD target, varying the cost of violating it','A larger penalty does not make the preference more accurate; 0 removes the penalty while revenue stays in the four-objective score.','penalty_sweep.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5.3))
    for ax,d,title in [(axes[0],prof,'Wide range'),(axes[1],prof[prof.target_x.between(147.1,147.3)],'Near the revenue kink')]:
        ax.plot(d.target_x,d.J_without_income_penalty,'o-',c=BLUE,label='Without revenue-gap penalty')
        ax.plot(d.target_x,d.J,'s--',c=GOLD,label='With current penalty lambda_R=5')
        ax.axvline(base.x,c=GRAY,ls=':',lw=1);ax.set(xlabel='Fixed annual visitors (10k)',ylabel='Re-optimized composite score (higher is better)',title=title);ax.legend(fontsize=9,loc='lower right')
    save(fig,'03_fixed_visitor_profile','Fix visitors then optimize the rest: inspect the revenue kink','Each visitor level enumerates feasible measure combos; curves are a numerical upper-envelope approximation, not proof of global optimality.','visitor_profile.csv')
    components=np.array([w[0]*(base.R-free.R)/p['R0'], -w[1]*(base.C_env-free.C_env),-w[2]*(base.C_soc-free.C_soc),-w[3]*(base.C_inf-free.C_inf),
        p['penalty_R']*max(p['R_min']-free.R,0)/p['R0']-base.income_penalty,-base.other_penalty+free.other_penalty])
    names=['Economic revenue term','Environmental pressure term','Social pressure term','Infrastructure pressure term','Avoided revenue-gap penalty','Other soft penalties']
    expected=base.J-(free.J-p['penalty_R']*max(p['R_min']-free.R,0)/p['R0'])
    assert np.isclose(components.sum(),expected,atol=1e-8)
    pd.DataFrame({'component':names,'delta_J':components}).to_csv(out/'score_decomposition_en.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(10,5.3));bars=ax.barh(names,components,color=[BLUE if v>=0 else GOLD for v in components]);ax.bar_label(bars,fmt='%+.4f',padding=4);ax.invert_yaxis();ax.axvline(0,c='#555555',lw=.8);ax.set_xlim(min(components.min()*1.7,-.15),components.max()*1.17);ax.set_xlabel('Score contribution difference: default vs. no-penalty policy')
    save(fig,'04_score_decomposition','Both policies evaluated at lambda_R=5: where the difference comes from','Terms are additive; the avoided revenue-gap penalty is a preference term, not new fiscal revenue or actual economic benefit.','counterfactuals.csv; score_decomposition_en.csv')
    pd.DataFrame(entries).to_csv(out/'figure_index_en.csv',index=False,encoding='utf-8-sig')
    print('English audit figures written to results/figures')

if __name__=='__main__':
    render(Path(__file__).resolve().parent)
