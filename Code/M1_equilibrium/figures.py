"""Static research figures (PNG + SVG); every figure reads exported evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize

COLORS=['#286187','#BA8B33','#7A8560','#B96583','#888888']
CAT=['基建','生态','社区']
LABELS={'epsilon':'税费需求弹性','eta_I':'基建效率','eta_E':'生态效率','eta_C':'社区效率','K_I':'基建饱和尺度','K_E':'生态饱和尺度','K_C':'社区饱和尺度','acceptance_slope':'居民响应强度','peak_factor':'旺季集中程度','measure_cost_multiplier':'措施成本','R0':'基准经济活动','S_min':'居民指数下限'}

def create_figures(out,root,p):
    font=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':11,'axes.titlesize':14,'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'path'})
    figdir=out/'figures';figdir.mkdir(exist_ok=True);entries=[]
    def load(name):return pd.read_csv(out/(name+'.csv'))
    best=load('recommended').iloc[0];old=pd.read_csv(root/'baseline_round1/recommended.csv').iloc[0]
    sc=load('scenarios').set_index('case');comp=load('comparable_policies').set_index('case');front=load('pareto');sens=load('sensitivity');curves=load('response_curves')
    def finish(fig,name,title,source,note):
        fig.suptitle(title,y=.99,fontweight='bold');fig.text(.015,.012,note,fontsize=9,color='#555555')
        fig.tight_layout(rect=(0,.055,1,.94))
        fig.savefig(figdir/(name+'.png'),dpi=190);fig.savefig(figdir/(name+'.svg'));plt.close(fig)
        entries.append(dict(file=name,title=title,source=source,note=note))
    # 01: nominal investments in the delivered solutions, not outcome comparison.
    fig,ax=plt.subplots(figsize=(9,5.5));x=np.arange(3)
    a=np.array([old['spend_'+k]*100 for k in 'IEC']);b=np.array([best['spend_'+k]*100 for k in 'IEC'])
    ba=ax.bar(x-.19,a,.36,label='第一轮：新增投入',color='#B5C8D5');bb=ax.bar(x+.19,b,.36,label='第二轮：新增投入',color=COLORS[0])
    ax.bar_label(ba,fmt='%.2f',padding=3);ax.bar_label(bb,fmt='%.2f',padding=3);ax.set_xticks(x,CAT);ax.set_ylabel('百万美元/年');ax.set_ylim(0,max(a.max(),b.max())*1.16);ax.legend()
    finish(fig,'01_两轮新增投资','两轮新增投资分配','baseline_round1/recommended.csv; recommended.csv','同一评分权重；两轮响应结构不同。这里只比较金额，不把压力指数跨模型直接作差。')
    # 02 fair same-model comparison, mark feasibility.
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    keys=['x','R','C_env','C_soc','C_inf','S'];titles=['年游客（万人）','旅游经济活动（亿美元）','环境压力指数','社会压力指数','基建压力指数','居民接受度指数']
    for ax,k,title in zip(axes.flat,keys,titles):
        vals=[comp.loc['round1_policy_in_round2',k],best[k]];bars=ax.bar(['原政策代入新模型','第二轮方案'],vals,color=['#B5C8D5',COLORS[0]])
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=10);ax.set_title(title,fontsize=12);ax.set_ylim(0,max(vals)*1.25)
    oldfeas=bool(comp.loc['round1_policy_in_round2','feasible'])
    finish(fig,'02_同模型效果对照','固定新模型：比较政策而非量尺','comparable_policies.csv',f'原政策代入第二轮后硬约束可行：{oldfeas}；该反事实若不可行，不应作为可执行备选方案。')
    # 03 Pareto projections
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for ax,k,title in zip(axes,['C_env','C_soc','C_inf'],['环境压力','社会压力','基建压力']):
        ax.scatter(front.R*100,front[k],s=12,c=COLORS[0],alpha=.35)
        ax.scatter(best.R*100,best[k],s=150,c=COLORS[3],marker='*',label='第二轮推荐')
        ax.set(xlabel='旅游经济活动（百万美元）',ylabel=title);ax.legend(fontsize=9)
    finish(fig,'03_Pareto投影','四目标近似非支配解的三组投影','pareto.csv; recommended.csv','投影中的点可能彼此压过，但在四维目标中非支配；随机搜索与局部精修不证明全局最优。')
    # 04 investment response
    fig,axes=plt.subplots(1,3,figsize=(13,4.7))
    for ax,k,label,color in zip(axes,'IEC',CAT,COLORS):
        c=curves[curves.category==k];ax.plot(c.spend_million,c.response*100,color=color,lw=2.3)
        spend=best['spend_'+k]*100;gain=best[{'I':'capacity_gain','E':'eco_reduction','C':'community_reduction'}[k]]*100
        ax.scatter(spend,gain,color=color,s=55);ax.axvline(spend,color='#666666',ls=':',lw=1)
        ax.set(title=label,xlabel='新增投入（百万美元/年）',ylabel='容量增幅（%）' if k=='I' else '原始压力削减（%）');ax.set_ylim(0,100*max(p.eta_I*p.L_I,p.eta_E*p.L_E,p.eta_C*p.L_C)*1.1)
    finish(fig,'04_边际递减响应','投入响应曲线与推荐投入点','response_curves.csv; recommended.csv','曲线参数为透明情景假设，未拟合实测数据；削减比例不同于冰川退缩率或满意率。')
    # 05 ledger
    ledger=[('基建维护',best.maintenance_I*100),('社区基础服务',best.maintenance_C*100),('基建新增',best.spend_I*100),('生态新增',best.spend_E*100),('社区新增',best.spend_C*100),('措施等效成本',best.measure_cost*100),('未配置收入',(best.available-best.budget_used)*100)]
    pd.DataFrame(ledger,columns=['category','million_usd']).to_csv(out/'budget_ledger.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(10,5.4));bars=ax.barh([r[0] for r in ledger],[r[1] for r in ledger],color=[COLORS[4],COLORS[4],COLORS[0],COLORS[1],COLORS[2],COLORS[3],COLORS[4]])
    ax.bar_label(bars,fmt='%.3f',padding=3);ax.invert_yaxis();ax.set_xlabel('百万美元/年');ax.set_xlim(0,max(v for _,v in ledger)*1.23)
    finish(fig,'05_资金账本','总税费情景：维护、新增投资、措施分别列账','budget_ledger.csv','项目年化费用不是首年现金需求；未配置收入包含预留的20%，并非全部可自由支配。')
    # 06 resource usage
    fig,ax=plt.subplots(figsize=(9,4.9));ratios=[best.x/best.I,best.xd/(p.peak0-p.cap_reduction*best.m1),best.budget_used/best.budget_limit,best.C_env/p.E_max]
    bars=ax.barh(['年度容量','日峰值上限','可分配预算','生态压力阈值'],np.array(ratios)*100,color=COLORS[0]);ax.bar_label(bars,fmt='%.1f%%',padding=3);ax.axvline(100,color='#333333',ls='--',lw=1);ax.set_xlim(0,118);ax.invert_yaxis();ax.set_xlabel('使用量 / 约束上限（%）')
    finish(fig,'06_约束使用率','推荐方案的四项资源约束','recommended.csv; parameters_used.json','生态上限是情景阈值，不是经实测证明的安全线；所有原始约束余量另见constraint_audit.csv。')
    # 07 sensitivities
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    order=list(LABELS)
    for ax,method,title in zip(axes,['fixed','reoptimized'],['保持原政策','允许政策重新优化']):
        sub=sens[sens.method==method]
        for j,factor in enumerate([.9,1.1]):
            vals=[];flags=[]
            for key in order:
                row=sub[(sub.parameter==key)&(sub.factor==factor)].iloc[0];vals.append(row.J-best.J);flags.append(bool(row.feasible))
            yy=np.arange(len(order))+(j-.5)*.3
            ax.barh(yy,vals,height=.27,color=[COLORS[0],COLORS[1]][j],label=['参数×0.9','参数×1.1'][j])
            for y,v,flag in zip(yy,vals,flags):
                if not flag and np.isfinite(v):ax.scatter(v,y,c='#333333',marker='x',s=35,zorder=4)
        ax.set_yticks(np.arange(len(order)),[LABELS[k] for k in order]);ax.invert_yaxis();ax.axvline(0,color='#666666',lw=.8);ax.set_title(title);ax.set_xlabel('相对默认方案的评分变化');ax.legend(fontsize=9)
    delta=sens.J-best.J;lo=min(float(delta.min()),0);hi=max(float(delta.max()),0);pad=max((hi-lo)*.05,.005)
    for ax in axes:ax.set_xlim(lo-pad,hi+pad)
    finish(fig,'07_敏感性评分','单参数±10%：固定政策与重新优化','sensitivity.csv','两面板使用相同横轴；×表示不可行。评分含软罚项；负弹性×1.1表示绝对值增加。')
    # 08 allocation responses under all perturbations
    re=sens[sens.method=='reoptimized'].copy();re['label']=[LABELS[k]+(' ×0.9' if f<1 else ' ×1.1') for k,f in zip(re.parameter,re.factor)]
    fig,ax=plt.subplots(figsize=(11,9));left=np.zeros(len(re))
    for k,label,col in zip('IEC',CAT,COLORS):
        vals=re['spend_'+k].to_numpy()*100;ax.barh(re.label,vals,left=left,label=label,color=col);left+=vals
    ax.invert_yaxis();ax.set_xlabel('新增投入（百万美元/年）');ax.legend(ncol=3,loc='lower right')
    finish(fig,'08_敏感性资金分配','参数变化后的新增资金分配','sensitivity.csv','每一行均重新优化；预算总额随游客量和税费变化，条形长度不要求相同。')
    # 09 weight alternatives
    cases=['default','economic_priority','ecological_priority','community_priority','new_entropy_weights'];names=['固定第一轮权重','经济优先','生态优先','社区优先','重算AHP-熵权']
    fig,ax=plt.subplots(figsize=(10,5));left=np.zeros(len(cases))
    for k,label,col in zip('IEC',CAT,COLORS):
        vals=sc.loc[cases,'spend_'+k].to_numpy()*100;ax.barh(names,vals,left=left,color=col,label=label);left+=vals
    ax.invert_yaxis();ax.set_xlabel('新增投入（百万美元/年）');ax.legend(ncol=3)
    finish(fig,'09_权重方案','偏好权重改变时的投资结构','scenarios.csv','不同权重的J不能直接比较；此图仅比较各自优化后的金额。正投入不是程序强制比例。')
    # 10 maintenance and budget modes
    cases=['default','no_maintenance','incremental'];names=['总税费预算','取消维护预留','仅新增税费预算']
    fig,axes=plt.subplots(1,2,figsize=(12,5));xx=np.arange(3)
    for j,(k,label,col) in enumerate(zip('IEC',CAT,COLORS)):
        vals=sc.loc[cases,'spend_'+k].to_numpy()*100;bars=axes[0].bar(xx+(j-1)*.23,vals,.22,color=col,label=label);axes[0].bar_label(bars,fmt='%.1f',fontsize=9,padding=3)
    axes[0].set_xticks(xx,names);axes[0].set_ylabel('新增投入（百万美元/年）');axes[0].legend();axes[0].set_ylim(0,sc.loc[cases,['spend_I','spend_E','spend_C']].max().max()*125)
    for k,label,col in zip(['C_env','C_soc','C_inf'],['环境','社会','基建'],COLORS):axes[1].plot(names,sc.loc[cases,k],marker='o',label=label,color=col,lw=2)
    axes[1].set_ylabel('第二轮压力指数');axes[1].legend();axes[1].set_ylim(bottom=0)
    finish(fig,'10_预算与维护对照','必要维护是否造成正投入；财政空间如何改变结果','scenarios.csv','仅新增税费情景把既有维护留在原预算；没有用新增收入重复支付维护费用。')
    # 11 stress sample statistics
    stress=load('stress_samples');flags=(stress.feasible==True);fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].hist(stress.x,bins=26,color=COLORS[0],alpha=.85);axes[0].axvline(best.x,c='#333333',ls='--',label='默认解');axes[0].set(xlabel='年游客（万人）',ylabel='假设样本数');axes[0].legend()
    constraints=['capacity','ecology','daily_cap','budget'];fractions=[(stress['margin_'+k]<-1e-7).mean()*100 for k in constraints]
    bars=axes[1].barh(['年度容量','生态阈值','日峰值','预算'],fractions,color=COLORS[1]);axes[1].bar_label(bars,fmt='%.1f%%',padding=3);axes[1].set_xlim(0,105);axes[1].set_xlabel('该约束违反的样本比例（%）');axes[1].invert_yaxis()
    finish(fig,'11_联合扰动压力测试','600组假设联合扰动：固定政策承受能力','stress_samples.csv',f'独立均匀抽样，全部硬约束可行样本占{flags.mean():.1%}；这不是现实成功率或统计置信区间。各违反项可重叠。')
    # 12 landscape slice
    landscape=load('policy_landscape');landscape['plot_J']=landscape.J.where(landscape.feasible)
    pivot=landscape.pivot(index='transfer',columns='tau',values='plot_J').sort_index()
    fig,ax=plt.subplots(figsize=(9,6));cm=plt.get_cmap('cividis').copy();cm.set_bad('#eeeeee')
    # pcolormesh uses actual parameter coordinates, rather than category indexes.
    im=ax.pcolormesh(pivot.columns.to_numpy()*100,pivot.index.to_numpy()*100,pivot.to_numpy(),shading='nearest',cmap=cm)
    ax.scatter(best.tau*100,0,marker='*',s=170,c='#E8A9BD',edgecolors='#333333',label='推荐解');ax.set(xlabel='税费杠杆（%）',ylabel='从生态转给基建的份额（百分点）');ax.legend();fig.colorbar(im,ax=ax,label='固定权重评分J')
    finish(fig,'12_政策局部地形','税费与资金转移的局部政策切片','policy_landscape.csv','社区份额、措施保持推荐值；灰色为不可行或未定义区域。这里只是二维切片，不是全局最优证明。')
    # 13 seeds
    seeds=load('seed_comparison');fig,ax=plt.subplots(figsize=(8.5,4.7));xs=np.arange(len(seeds))
    ax.scatter(xs,seeds.best_J,color=COLORS[0],s=65,label='该种子的最佳候选');ax.axhline(best.J,color=COLORS[1],ls='--',label='32种组合多起点精修后推荐')
    ax.set_xticks(xs,['seed '+str(int(v)) for v in seeds.seed]);ax.set_ylabel('固定权重评分J');ax.legend()
    finish(fig,'13_随机种子对照','随机搜索差异与局部精修','seed_comparison.csv; recommended.csv','这是随机性对照，不是迭代收敛曲线；三次结果不能证明全局最优。')
    # 14 response mechanism ablation
    cases=['default','no_community_effect','no_investment_effects'];names=['完整反馈','关闭社区收益','关闭三类投资收益']
    fig,axes=plt.subplots(1,2,figsize=(12,5));left=np.zeros(len(cases))
    for k,label,col in zip('IEC',CAT,COLORS):
        vals=sc.loc[cases,'spend_'+k].to_numpy()*100;axes[0].barh(names,vals,left=left,label=label,color=col);left+=vals
    axes[0].invert_yaxis();axes[0].set_xlabel('新增投入（百万美元/年）');axes[0].legend()
    axes[1].plot(names,sc.loc[cases,'C_soc'],marker='o',color=COLORS[2],label='社会压力');axes[1].plot(names,sc.loc[cases,'S'],marker='s',ls='--',color=COLORS[0],label='居民指数');axes[1].set_ylim(0,1.1);axes[1].legend()
    finish(fig,'14_反馈通道消融','关闭投入作用后的重新优化','scenarios.csv','零收益通道按等评分少花钱原则置零；这不是最低投入约束，社区正投入来自其在模型中的实际作用。')
    # 15 smooth response and strict-acceptance scenario.
    from scipy.special import expit
    cgrid=np.linspace(0,1.6,161);acceptance=[]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for gamma,col in zip([1.,2.,3.],COLORS):
        ss=expit(np.log(p.S0/(1-p.S0))-gamma*(cgrid-1))
        axes[0].plot(cgrid,ss,color=col,label=f'响应强度 {gamma:.0f}',lw=2)
        acceptance.extend(dict(social_pressure=c,acceptance=v,slope=gamma) for c,v in zip(cgrid,ss))
    pd.DataFrame(acceptance).to_csv(out/'acceptance_curve.csv',index=False,encoding='utf-8-sig')
    axes[0].scatter([best.C_soc],[best.S],marker='*',s=120,c='#333333');axes[0].set(xlabel='社会压力指数',ylabel='居民接受度指数',ylim=(0,1));axes[0].legend()
    cc=['default','strict_acceptance'];xx=np.arange(2)
    for j,(k,label,col) in enumerate(zip('IEC',CAT,COLORS)):
        vv=sc.loc[cc,'spend_'+k].to_numpy()*100;bars=axes[1].bar(xx+(j-1)*.23,vv,.22,label=label,color=col);axes[1].bar_label(bars,fmt='%.1f',fontsize=9,padding=3)
    axes[1].set_xticks(xx,['软阈值0.7','软阈值0.9']);axes[1].set_ylabel('新增投入（百万美元/年）');axes[1].legend();axes[1].set_ylim(0,sc.loc[cc,['spend_I','spend_E','spend_C']].max().max()*125)
    finish(fig,'15_居民响应与严格目标','平滑居民响应及更严格接受度目标','acceptance_curve.csv; scenarios.csv','指数不是满意率；0.9仍是软目标，是否达到需查看scenarios.csv中的S与soft_penalty。')
    pd.DataFrame(entries).to_csv(out/'figure_index.csv',index=False,encoding='utf-8-sig')
    # Generate the Chinese interpretation with model-comparable and scoped claims.
    nm=sc.loc['no_maintenance'];inc=sc.loc['incremental']
    text=f'''# Model 1 第二轮：模型修正与结果解读

第二轮使用边际递减投资响应、社区直接减压和平滑居民接受度。主实验沿用第一轮组合权重，避免公式和偏好同时改变。未设置任何三类新增投入最低比例。

## 推荐方案（情景推演，非实测预测）

| 指标 | 第二轮 |
|---|---:|
| 年游客量 | {best.x:.3f} 万人 |
| 日峰值 | {best.xd:.4f} 万人 |
| 旅游经济活动 | {best.R:.4f} 亿美元 |
| 综合税费杠杆 | {best.tau:.3%} |
| 新增基建 / 生态 / 社区投入 | {best.spend_I*100:.3f} / {best.spend_E*100:.3f} / {best.spend_C*100:.3f} 百万美元/年 |
| 含基础维护的基建 / 社区支出 | {best.total_I*100:.3f} / {best.total_C*100:.3f} 百万美元/年 |
| 措施年度等效费用 | {best.measure_cost*100:.3f} 百万美元 |
| 居民指数（非满意率） | {best.S:.4f} |
| 环境 / 社会 / 基建压力 | {best.C_env:.4f} / {best.C_soc:.4f} / {best.C_inf:.4f} |
| 最大硬约束违反 | {best.max_violation:.3g} |
| 需求等式残差 | {best.demand_residual:.3g} 万人 |

措施m1至m5为：{[int(best['m'+str(i)]) for i in range(1,6)]}，依次为额外日限流、酒类限制、岸电、交通管理、营销。岸电费用为年化情景，不等于首年建设现金。无额外日限流并不取消基础日峰值/船数约束。

## 为什么不再全部投生态

生态投资每追加一美元的减压作用逐步减少；社区投资能直接降低社会压力，并经居民接受度影响需求；基建投资也有容量与压力收益。三类资金由这些反馈共同决定，未人为指定各占多少。居民接受度以0.7为参考的平滑函数，不会像第一轮一样被截断成1；0.7是情景校准点，不是调查估计。

取消维护预留后，新增基建/生态/社区仍为 {nm.spend_I*100:.3f}/{nm.spend_E*100:.3f}/{nm.spend_C*100:.3f} 百万美元。这个对照用于区分投资收益与基础费用，不能据此断言这些数值已经符合现实最优配置。

第一轮与第二轮采用不同压力函数和居民指数基准，所以不可直接用第一轮环境指数0.090与第二轮指数作改善率比较。图02使用同一第二轮公式重新计算第一轮政策。该反事实是否可行：{oldfeas}；不可行的旧政策只用于诊断。

## 仍需关注的边界与稳健性

- 当前游客量{best.x:.3f}万人，对应收入{best.R:.4f}亿；收入软下限为{p.R_min:.2f}亿。若两者接近，说明收入保障罚项正在塑造解，而非找到与该阈值无关的自然最佳规模。
- 可分配预算使用率{best.budget_used/best.budget_limit:.2%}。满额预算方案面对成本增加可能不可行，因此不能称为已经求得鲁棒最优政策。
- 仅新增税费模式：基建/生态/社区 {inc.spend_I*100:.3f}/{inc.spend_E*100:.3f}/{inc.spend_C*100:.3f} 百万美元。原有维护继续由既有预算承担；此模式重新优化时可能出现零投入，这是资金约束下允许的结果。
- 12项参数±10%分析分别报告固定政策与重新优化；阈值罚项、资金变化都会影响J，不等于因果重要性。权重方案另列，不直接跨权重比较J。
- 严格居民目标情景将S_min提高到0.9，重新优化后S为{sc.loc['strict_acceptance','S']:.4f}，软罚项为{sc.loc['strict_acceptance','soft_penalty']:.4f}，未达到0.9目标；这不是硬约束被忽略，也不是不可行性证明。不能把默认0.7解释为现实已获得的公众认可。
- 600个假设联合扰动样本中，固定政策满足全部硬约束的比例为{flags.mean():.1%}。该比例完全依赖人为抽样区间和独立假设，不是现实政策成功概率，也不是置信度。
- 投资响应上限、饱和尺度、居民响应强度仍是假设；维护金额只采用官方预算提案部分服务的代理，未覆盖朱诺全部维护需求。未把这些新参数包装成拟合结果。

## 图表阅读顺序

先看图01、02、04、10，理解投资变化、同口径效果、递减机制和维护对照；再看图07、08、09、11检查敏感性、权重和联合扰动。

'''
    for e in entries:
        text+=f"### {e['title']}\n\n![{e['title']}](figures/{e['file']}.png)\n\n{e['note']} 数据：`{e['source']}`。\n\n"
    (out/'第二轮结果解读.md').write_text(text)
