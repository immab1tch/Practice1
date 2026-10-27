"""English paper figures for the preference-space and minimax-regret extension."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/preference_robustness';FIG=ROOT/'figures/preference_robustness'
COLORS=['#286187','#C38F32','#758D66','#BA6985']
LABEL_EN={'VR':'Revenue vertex','VE':'Environment vertex','VS':'Society vertex','VI':'Infrastructure vertex','ERE':'Revenue-Environment edge midpoint','ERS':'Revenue-Society edge midpoint','ERI':'Revenue-Infrastructure edge midpoint','EES':'Environment-Society edge midpoint','EEI':'Environment-Infrastructure edge midpoint','ESI':'Society-Infrastructure edge midpoint','FR':'Face excluding Revenue','FE':'Face excluding Environment','FS':'Face excluding Society','FI':'Face excluding Infrastructure','C':'Equal-weight center','MAIN':'Original recommended weight','AHP':'Original AHP weight','ENT':'Round-2 entropy weight'}

def render():
    FIG.mkdir(parents=True,exist_ok=True);s=json.loads((OUT/'summary.json').read_text())
    wp=pd.read_csv(OUT/'weights.csv');pol=pd.read_csv(OUT/'policy_menu.csv').set_index('policy_id');reg=pd.read_csv(OUT/'regret_matrix.csv').set_index('policy_id');rank=pd.read_csv(OUT/'regret_ranking.csv')
    W=wp[['w_R','w_E','w_S','w_I']].to_numpy();menu=pol.loc[['P_'+x for x in wp.weight_id]];orig=pol.loc['ORIGINAL'];rob=pol.loc['ROBUST']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'path'})
    entries=[]
    def finish(fig,name,title,source,note):
        fig.suptitle(title,fontsize=16,fontweight='bold',y=.985);fig.text(.02,.015,note,fontsize=9,color='#555555')
        fig.tight_layout(rect=(.01,.065,.99,.95));fig.savefig(FIG/(name+'.png'),dpi=300);fig.savefig(FIG/(name+'.svg'));plt.close(fig)
        entries.append(dict(file=name,title=title,source=source,note=note))
    fig,ax=plt.subplots(figsize=(10,9));im=ax.imshow(W,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(4),['Revenue wR','Environment wE','Society wS','Infrastructure wI']);ax.set_yticks(range(18),[f'{r.weight_id}  {LABEL_EN[r.weight_id]}' for r in wp.itertuples()])
    for i in range(18):
        for j in range(4):ax.text(j,i,f'{W[i,j]:.3f}',ha='center',va='center',fontsize=10,color='white' if W[i,j]>.55 else '#222222')
    for y in [3.5,9.5,13.5,14.5]:ax.axhline(y,color='white',lw=2)
    fig.colorbar(im,ax=ax,label='Weight',fraction=.035,pad=.03)
    finish(fig,'M3R_01_Weight_sampling','15 structural points + 3 named points = 18 preference sets','weights.csv','Each row is nonnegative and sums to 1; R/E/S/I are revenue, environment, society, infrastructure. Vertex scenarios keep the original soft penalties.')
    fig,axes=plt.subplots(1,3,figsize=(15,10),gridspec_kw={'width_ratios':[1.15,1.3,.9]});y=np.arange(18)
    labels=[f'{r.weight_id}  {LABEL_EN[r.weight_id]}' for r in wp.itertuples()]
    axes[0].scatter(menu.x,y,s=45,c=COLORS[0]);axes[0].axvline(147.2,ls='--',color=COLORS[3],lw=1)
    for yy,x in zip(y,menu.x):axes[0].annotate(f'{x:.2f}',(x,yy),xytext=(5,0),textcoords='offset points',va='center',fontsize=9)
    axes[0].set_yticks(y,labels);axes[0].set_xlim(145,164);axes[0].set_xlabel('Annual visitors (10,000, local scale)');axes[0].set_title('Scale plateaus, with variation',fontsize=12)
    left=np.zeros(18)
    for k,label,col in zip('IEC',['Infrastructure','Ecology','Community'],COLORS):
        values=menu['spend_'+k].to_numpy()*100;axes[1].barh(y,values,left=left,label=label,color=col,height=.65);left+=values
    axes[1].set_yticks(y,[]);axes[1].set_xlabel('New improvement spending (million USD/yr)');axes[1].set_title('Investment mix changes with preference',fontsize=12);axes[1].legend(ncol=3,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.06),frameon=False)
    measures=menu[['m1','m2','m3','m4','m5']].to_numpy();axes[2].imshow(measures,cmap='Blues',vmin=0,vmax=1,aspect='auto',extent=(-.5,4.5,17.5,-.5))
    axes[2].set_xticks(range(5),['m1','m2','m3','m4','m5']);axes[2].set_yticks(y,[]);axes[2].set_title('Measure switches (1=on)',fontsize=12)
    for i in range(18):
        for j in range(5):axes[2].text(j,i,str(int(measures[i,j])),ha='center',va='center',color='white' if measures[i,j] else '#333333',fontsize=10)
    for ax in axes[:2]:ax.set_ylim(17.5,-.5)
    finish(fig,'M3R_02_Policy_menu','Weights change scale, spending and discrete measure combinations','policy_menu.csv; weights.csv','m1 daily cap, m2 alcohol control, m3 shore power, m4 traffic management, m5 marketing. Only solved points are shown; continuous switch boundaries are not located.')
    ids=list(menu.index)+['ORIGINAL','ROBUST'];short=[x.replace('P_','') for x in ids]
    fig,ax=plt.subplots(figsize=(13,10));values=reg.loc[ids,wp.weight_id].to_numpy();im=ax.imshow(values,cmap='YlOrRd',vmin=0,vmax=max(.01,values.max()),aspect='auto')
    ax.set_xticks(range(18),wp.weight_id,rotation=45,ha='right');ax.set_yticks(range(20),short);ax.set_xlabel('Evaluating preference (columns)');ax.set_ylabel('Candidate policy (rows)')
    for i in range(20):
        for j in range(18):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if values[i,j]>values.max()*.6 else '#222222')
    ax.axhline(17.5,color='#555555',lw=1.3);fig.colorbar(im,ax=ax,label='Regret (lower is better)',fraction=.03,pad=.02)
    finish(fig,'M3R_03_Cross_weight_regret_matrix','Differences taken under the same preference, then maximum regret compared','regret_matrix.csv; reference_values.csv','Regret = the column numerically best reference score minus the policy score in that column; two decimals shown, exact values in the CSV. ORIGINAL is the original policy, ROBUST the continuously refined policy.')
    fig,ax=plt.subplots(figsize=(11,9));display=rank.copy();display['name']=[f'{r.policy_id.replace("P_","")}  {LABEL_EN.get(r.policy_id.replace("P_",""),r.policy_id)}' for r in display.itertuples()]
    colors=[COLORS[2] if k=='ROBUST' else COLORS[3] if k=='ORIGINAL' else COLORS[0] for k in display.policy_id]
    bars=ax.barh(display.name,display.max_regret,color=colors,height=.65);ax.bar_label(bars,fmt='%.5f',padding=4,fontsize=9);ax.invert_yaxis();ax.set_xlim(0,display.max_regret.max()*1.18);ax.set_xlabel('Maximum regret across 18 preferences (lower is better)')
    finish(fig,'M3R_04_Worst_regret_ranking','Minimax Regret: choose the policy with the smallest worst loss','regret_ranking.csv','P_MAIN belongs to the 18-item menu; ROBUST is separately refined in the original continuous decision space. Equal scores may correspond to the same policy, not independent votes.')
    fig,axes=plt.subplots(1,2,figsize=(12,5.7));x=np.arange(3)
    for off,r,label,color in [(-.18,orig,'Original fixed-weight recommendation',COLORS[0]),(.18,rob,'Continuously refined robust policy',COLORS[2])]:
        vals=[r['spend_'+k]*100 for k in 'IEC'];bars=axes[0].bar(x+off,vals,width=.34,color=color,label=label);axes[0].bar_label(bars,fmt='%.3f',padding=4,fontsize=9)
    axes[0].set_xticks(x,['Infrastructure','Ecology','Community']);axes[0].set_ylim(0,max(orig.spend_E,rob.spend_E)*100*1.25);axes[0].set_ylabel('New spending (million USD/yr)');axes[0].legend(fontsize=9);axes[0].set_title('Shifting a small share to infrastructure',fontsize=12)
    x=np.arange(4)
    for off,k,label,color in [(-.18,'ORIGINAL','Original fixed-weight recommendation',COLORS[0]),(.18,'ROBUST','Continuously refined robust policy',COLORS[2])]:
        vals=reg.loc[k,['VR','VE','VS','VI']].to_numpy(dtype=float);bars=axes[1].bar(x+off,vals,width=.34,color=color,label=label);axes[1].bar_label(bars,fmt='%.4f',padding=4,fontsize=9)
    axes[1].set_xticks(x,['Revenue vertex','Environment vertex','Society vertex','Infrastructure vertex']);axes[1].set_ylim(0,max(reg.loc['ORIGINAL',['VR','VE','VS','VI']])*1.3);axes[1].set_ylabel('Regret under that preference');axes[1].set_title('Worst regret balancing three pressure objectives',fontsize=12)
    finish(fig,'M3R_05_Robust_policy_comparison','Same visitor count and measure combo, funding adjusted to reduce worst regret','policy_menu.csv; regret_matrix.csv','Preference robustness is not budget feasibility under parameter perturbation: the same 600-scenario retest is reported separately.')
    pd.DataFrame(entries).to_csv(OUT/'figure_index_en.csv',index=False,encoding='utf-8-sig')
    print('Preference figures rendered:',len(entries))

if __name__=='__main__':
    render()
