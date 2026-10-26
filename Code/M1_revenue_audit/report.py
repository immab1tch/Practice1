from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

BLUE='#286187';GOLD='#BA8B33';PINK='#B96583';GRAY='#888888'

def render(root):
    out=root/'results';figdir=out/'figures';figdir.mkdir(exist_ok=True)
    font=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'path'})
    floor=pd.read_csv(out/'floor_sweep.csv',dtype={'combo':str});pen=pd.read_csv(out/'penalty_sweep.csv');prof=pd.read_csv(out/'visitor_profile.csv')
    counter=pd.read_csv(out/'counterfactuals.csv',dtype={'combo':str}).set_index('case');base=counter.loc['default'];free=counter.loc['income_penalty_removed']
    cross=pd.read_csv(out/'independent_optimizer.csv');p=json.loads((root/'inputs/parameters_used.json').read_text());w=json.loads((root/'inputs/weights_used.json').read_text())['weights']
    entries=[]
    def save(fig,name,title,note,source):
        fig.suptitle(title,y=.99,fontweight='bold',fontsize=15);fig.text(.015,.014,note,color='#555555',fontsize=9)
        fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(figdir/(name+'.png'),dpi=200);fig.savefig(figdir/(name+'.svg'));plt.close(fig)
        entries.append(dict(name=name,title=title,note=note,source=source))
    fig,axes=plt.subplots(1,2,figsize=(13,5.4))
    f=floor[floor.R_min>=2.9]
    axes[0].plot(f.R_min,f.x,'o-',color=BLUE,lw=1.8,label='重新优化游客量')
    axes[0].plot(f.R_min,f.floor_x,'--',color=GOLD,label='若恰好满足收入目标：Rmin/p0')
    axes[0].scatter([base.R_min],[base.x],c=PINK,marker='*',s=180,zorder=5,label='第二轮默认')
    axes[0].axhline(160,color=GRAY,ls=':',label='基础日峰值对应的年上限')
    axes[0].set(xlabel='收入软下限（亿美元/年）',ylabel='年游客量（万人）',ylim=(120,175));axes[0].legend(fontsize=9)
    local=f[f.R_min.between(3.39,3.6)]
    axes[1].plot(local.R_min,local.x,'o-',color=BLUE,label='重新优化')
    axes[1].plot(local.R_min,local.floor_x,'--',color=GOLD,label='Rmin/p0')
    axes[1].scatter([base.R_min],[base.x],c=PINK,marker='*',s=180,zorder=5)
    axes[1].set(xlabel='收入软下限（亿美元/年）',ylabel='年游客量（万人）',title='147.2万人附近的放大图');axes[1].legend()
    save(fig,'01_收入阈值扫描','改变收入下限，其他参数与权重保持不变','每个点重选32种措施组合并优化连续变量；连线只连接已检查的采样点，不表示完整连续解。','floor_sweep.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5.1))
    axes[0].plot(pen.penalty_R,pen.x,'o-',color=BLUE);axes[0].axhline(base.x,c=GRAY,ls='--');axes[0].set_xscale('symlog',linthresh=.005);axes[0].set(xlabel='收入缺口罚系数 λR（含0的对数尺度）',ylabel='年游客量（万人）')
    axes[1].plot(pen.penalty_R,pen.income_gap*100,'o-',c=GOLD);axes[1].set_xscale('symlog',linthresh=.005);axes[1].set(xlabel='收入缺口罚系数 λR',ylabel='未满足的收入目标（百万美元/年）');axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlim(0,12);ax.set_xticks([0,.01,.1,1,10],['0','0.01','0.1','1','10'])
    save(fig,'02_罚系数扫描','相同3.45亿美元目标，改变违反目标的代价','罚系数越大不等于现实偏好越准确；0代表取消收入罚项，收入仍保留在四目标评分中。','penalty_sweep.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5.3))
    for ax,d,title in [(axes[0],prof,'较宽区间'),(axes[1],prof[prof.target_x.between(147.1,147.3)],'收入折点附近')]:
        ax.plot(d.target_x,d.J_without_income_penalty,'o-',c=BLUE,label='不计收入缺口罚项')
        ax.plot(d.target_x,d.J,'s--',c=GOLD,label='计入当前罚项 λR=5')
        ax.axvline(base.x,c=GRAY,ls=':',lw=1);ax.set(xlabel='固定年游客量（万人）',ylabel='重新优化后的综合评分（越高越好）',title=title);ax.legend(fontsize=9,loc='lower right')
    save(fig,'03_固定游客量剖面','固定游客量，再优化其余决策：检查收入折点','每个游客量枚举允许的措施组合；曲线是数值上包络近似，不是全局最优证明。','visitor_profile.csv')
    # Exact additive decomposition of J(default)-J(no-penalty policy evaluated at lambda=5).
    components=np.array([w[0]*(base.R-free.R)/p['R0'], -w[1]*(base.C_env-free.C_env),-w[2]*(base.C_soc-free.C_soc),-w[3]*(base.C_inf-free.C_inf),
        p['penalty_R']*max(p['R_min']-free.R,0)/p['R0']-base.income_penalty,-base.other_penalty+free.other_penalty])
    names=['经济收益项','环境压力项','社会压力项','基建压力项','避免收入缺口罚分','其他软罚项']
    expected=base.J-(free.J-p['penalty_R']*max(p['R_min']-free.R,0)/p['R0'])
    assert np.isclose(components.sum(),expected,atol=1e-8)
    pd.DataFrame({'component':names,'delta_J':components}).to_csv(out/'score_decomposition.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(10,5.3));bars=ax.barh(names,components,color=[BLUE if v>=0 else GOLD for v in components]);ax.bar_label(bars,fmt='%+.4f',padding=4);ax.invert_yaxis();ax.axvline(0,c='#555555',lw=.8);ax.set_xlim(min(components.min()*1.7,-.15),components.max()*1.17);ax.set_xlabel('默认政策相对取消罚项政策的评分贡献差')
    save(fig,'04_评分来源','把两个政策都放回当前λR=5的目标：差异来自哪里','各项可加总；“避免收入缺口罚分”是人为偏好项，不是新增财政收入或实际经济收益。','counterfactuals.csv; score_decomposition.csv')
    center=prof.loc[np.isclose(prof.target_x,base.x)].iloc[0]
    left=prof.loc[np.isclose(prof.target_x,base.x-.01)].iloc[0];right=prof.loc[np.isclose(prof.target_x,base.x+.01)].iloc[0]
    dl=(center.J_without_income_penalty-left.J_without_income_penalty)/.01
    dr=(right.J_without_income_penalty-center.J_without_income_penalty)/.01
    local_critical=max(0.,-dl*p['x0'])
    region=floor[floor.at_income_kink]
    nearby=floor[floor.R_min.between(3.4,3.75)]
    first_sticky=pen[pen.at_income_kink].penalty_R.min()
    delta_people=base.x-free.x
    # Auditable computed facts for prose and independent verification.
    facts=dict(default_x=float(base.x),default_R=float(base.R),default_floor=float(base.R_min),no_penalty_x=float(free.x),no_penalty_R=float(free.R),
        delta_x_10k=float(delta_people),profile_left_slope=float(dl),profile_right_slope=float(dr),penalty_marginal_per_10k=p['penalty_R']/p['x0'],
        local_penalty_balance_estimate=float(local_critical),first_checked_penalty_at_kink=float(first_sticky),
        expected_dx_per_100m=p['x0']/p['R0'],score_decomposition_total=float(components.sum()),independent_max_score_gap=float(cross.score_gap.abs().max()))
    (out/'findings.json').write_text(json.dumps(facts,indent=2))
    selected=floor[np.isclose(floor.R_min,3.3)|np.isclose(floor.R_min,3.4)|np.isclose(floor.R_min,3.45)|np.isclose(floor.R_min,3.5)|np.isclose(floor.R_min,3.6)|np.isclose(floor.R_min,3.8)]
    table='\n'.join(f'| {r.R_min:.3f} | {r.x:.4f} | {r.R:.4f} | {"是" if r.at_income_kink else "否"} | {r.combo} |' for _,r in selected.iterrows())
    text=f'''# 147.2万人是否被收入下限固定：专项核查

## 结论

**在第二轮当前模型、固定权重与已检查的情景下，147.2万人确实是收入软下限作用下的折点解，不能称为不依赖政策偏好的“客观最佳承载量”。** 收入下限没有作为硬等式写进游客量；但较强的收入缺口罚项使最优候选停在恰好满足目标的位置。以下结论有数值搜索支持，不是全局最优证明。

## 保持不变的条件

冻结第二轮公式、其他参数和目标权重；只改变R_min或λR。收入下限与罚系数扫描每点枚举全部32种措施组合，连续部分使用至少4个起点（相邻方案作额外起点）；固定游客量剖面每点枚举允许的措施组合并使用2个起点。没有重新计算熵权，没有改变投资效率、税费上下界或需求模型。完整模型快照与哈希保存在本目录。

## 证据一：等式吻合，而且周边重新优化也跟随阈值

经济活动代理p0=3.75/160=0.0234375亿美元/万人。若恰好满足3.45亿美元目标，则：

\\[
x_{{floor}}=R_{{min}}/p_0=3.45/(3.75/160)=147.2\\text{{ 万人}}.
\\]

默认重新求解为 x={base.x:.9f}万人、R={base.R:.9f}亿美元。以下各点均重新优化全部政策变量，而不是把公式值当作优化结果：

| 收入软下限（亿美元） | 优化游客量（万人） | 优化收入（亿美元） | 是否贴住收入目标 | 措施m1…m5 |
|---:|---:|---:|:---:|:---:|
{table}

贴住目标时，收入下限每提高0.01亿美元（100万美元），游客量对应增加约0.42667万人，即4267人。这个比例来自固定人均经济活动假设，不是新增的实证发现。

并非所有区间都跟随直线。更低收入目标下存在措施切换和平台；过高目标还受日峰值限制。由xd=2.675x/214且xd≤2，得到x≤160万人、R≤3.75亿美元。因此3.8或4亿美元目标在当前参数下不能完全满足，**软约束可以产生缺口**，不能称为硬保底。

## 证据二：取消罚项后，不再选择147.2万人

只把λR从5改为0，仍保留收入在四目标中的正向贡献，得到约 **{free.x:.4f}万人、{free.R:.4f}亿美元**，措施组合{free.combo}。与默认相比少约{delta_people:.4f}万人。另一个独立对照把R_min改为0、保留λR=5，得到同样的候选，说明变化来自收入缺口罚项被移除，而不是把经济收益目标一起删除。

在取消收入罚项的同一量尺下，默认方案评分为{base.J_without_income_penalty:.8f}，无收入罚项候选为{free.J_without_income_penalty:.8f}，后者更高。但它不满足原3.45亿美元保障目标，不能直接称为现实中更优。

## 证据三：罚系数和固定游客量剖面说明了折点机制

27个收入下限、16个罚系数和14个固定游客量剖面点均已检查。罚系数扫描中，首个贴住收入目标的已检查λR为{first_sticky:g}；这是离散扫描结果，不是精确临界值。

当游客量低于147.2万人时，收入罚项为

\\[
P_R=\\lambda_R\\max(0,R_{{min}}/R_0-x/x_0).
\\]

因此在阈值以下，每少1万人，单是收入罚项就增加5/160=0.03125评分单位。固定游客量、重优化其他决策后的无收入罚项评分，在147.2万人附近的左/右数值斜率分别为{dl:.7f}/{dr:.7f}（每万人）。图03显示收入罚项如何改变折点左侧的取舍。局部左侧平衡对应λR约{local_critical:.5f}，只是该分支附近的有限差分估计；其他措施组合可能在更低罚系数时成为更好方案，不能据此宣称全局临界值。

图04将两个政策都放回当前λR=5的目标，评分差分解为真实目标项与人为罚项。四个目标本身与收入罚项的贡献不应混为经济收益。

## 求解和数值核验

- 每个输出政策均独立回代需求等式、预算与硬约束；日志包含失败的尝试，不把“求解器成功”单独作为验收。
- 对默认和取消收入罚项的优胜措施分支，另外用差分进化法搜索连续变量；它与枚举加多起点局部求解的最大评分差为{cross.score_gap.abs().max():.3g}。这只交叉核验所选分支，不证明其他非凸分支的全局最优。
- m1…m5依次为额外日限流、酒类限制、岸电、交通管理、营销。组合切换可能引起游客量跳跃。

## 对论文结论的影响

建议表述为：“在年旅游经济活动不低于3.45亿美元的偏好设定及当前参数下，模型给出约147.2万人的折中方案。”不宜表述为：“朱诺的最佳游客承载量为147.2万人。”

下一步需要为3.45亿美元收入目标提供就业、生计或基准损失容忍度依据，并将该目标作为政策选择展示。当前检查不支持为了获得更漂亮的游客数而任意改阈值，也不意味着应直接采用取消保障后约127.69万人的方案。

## 图表与复现

运行 `audit.py` 可复现扫描、固定游客量剖面、交叉优化器检查和图表；`verify.py` 独立回代输出。当前电脑使用第一轮的Python环境即可。所有图的CSV在同目录，SVG与PNG在figures子目录。

'''
    for e in entries:text+=f"### {e['title']}\n\n![{e['title']}](figures/{e['name']}.png)\n\n{e['note']} 数据：`{e['source']}`。\n\n"
    (out/'收入下限专项核查报告.md').write_text(text)
    pd.DataFrame(entries).to_csv(out/'figure_index.csv',index=False,encoding='utf-8-sig')
