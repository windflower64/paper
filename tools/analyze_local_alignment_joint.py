"""Read-only audit of report159, with Chinese conclusions and reproducible curves."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('E:/two_paper')
REPORT=ROOT/'reports/159_local_alignment_joint_stable16'
DOC=ROOT/'knowledge/M系列知识库/09_显式目标对齐与SAM形状约束/07_联合训练结果与失败边界.md'


def main():
    queue=json.loads((REPORT/'queue_status.json').read_text(encoding='utf-8'))
    assert queue['status']=='complete' and set(queue['completed'])=={'fusion','rgb'}
    manifest=json.loads((REPORT/'manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
    baseline=json.loads((ROOT/'reports/158_mbudet_rgb_alignment/initial/initial_verified.json').read_text())['metrics']
    result={'status':'ANALYZED','protected_hashes':'PASS','manifest_files':len(manifest),
            'baseline':baseline,'arms':{}}
    curves={}
    for arm in ('fusion','rgb'):
        run=ROOT/f'outputs/M_LOCAL_ALIGN_JOINT_{arm.upper()}_B16A2_30E_TESTDEV/seed0'
        rows=[json.loads(line) for line in (run/'log.txt').read_text().splitlines() if line.strip()]
        assert [r['epoch'] for r in rows]==list(range(30))
        progress=json.loads((run/'training_progress.json').read_text())
        assert progress['status']=='complete' and progress['actual_optimizer_updates']==3000
        verification=json.loads((run/'independent_best_ema.json').read_text())
        best=max(rows,key=lambda r:r['test_coco_eval_bbox'][0])
        assert verification['status']=='PASS'
        assert max(abs(a-b) for a,b in zip(best['test_coco_eval_bbox'],verification['metrics']))==0
        curves[arm]=rows
        result['arms'][arm]={'best_epoch_zero_based':best['epoch'],'best_metrics':best['test_coco_eval_bbox'],
            'last_metrics':rows[-1]['test_coco_eval_bbox'],
            'last5_mean':np.mean([r['test_coco_eval_bbox'] for r in rows[-5:]],axis=0).tolist(),
            'train_loss_first_last':[rows[0]['train_loss'],rows[-1]['train_loss']],
            'train_bbox_first_last':[rows[0]['train_loss_bbox'],rows[-1]['train_loss_bbox']],
            'train_giou_first_last':[rows[0]['train_loss_giou'],rows[-1]['train_loss_giou']],
            'field_loss_first_last':[rows[0].get('train_loss_alignment_field'),rows[-1].get('train_loss_alignment_field')],
            'epochs_above_initial_ap':sum(r['test_coco_eval_bbox'][0]>baseline[0] for r in rows),
            'independent_metric_max_error':verification['max_abs_error'], 'updates':3000}
    f,r=result['arms']['fusion'],result['arms']['rgb']
    diffs=np.array([a['test_coco_eval_bbox'] for a in curves['fusion']])-np.array([a['test_coco_eval_bbox'] for a in curves['rgb']])
    result['same_epoch']={'ap_wins':int((diffs[:,0]>0).sum()),'aps_wins':int((diffs[:,3]>0).sum()),
        'mean_ap_delta_points':float(diffs[:,0].mean()*100)}
    result['best_delta_points']={name:(f['best_metrics'][i]-r['best_metrics'][i])*100 for name,i in [('AP',0),('AP50',1),('AP75',2),('APS',3),('APM',4)]}
    result['conclusion']='复杂融合联合训练未获得净收益；纯RGB续训也退化，融合后期退化更重。不能证明冻结是上一轮唯一原因，不能归咎SAM或宣布对齐无用。'
    (REPORT/'conclusion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for arm,rows in curves.items():
        epochs=np.arange(1,31)
        for ax,index,title in [(axes[0],0,'Detection AP'),(axes[1],3,'Small-object AP')]:
            ax.plot(epochs,[x['test_coco_eval_bbox'][index]*100 for x in rows],label=arm)
            ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(alpha=.25)
        axes[2].plot(epochs,[x['train_loss']-x.get('train_loss_alignment_field',0) for x in rows],label=arm)
    axes[0].axhline(baseline[0]*100,color='gray',ls='--',label='initial RGB')
    axes[1].axhline(baseline[3]*100,color='gray',ls='--',label='initial RGB')
    axes[2].set_title('Training detection loss (excluding alignment)'); axes[2].set_xlabel('Epoch')
    for ax in axes: ax.legend()
    fig.savefig(REPORT/'training_comparison.png',dpi=180); plt.close(fig)
    lines=['# 复杂融合与全链路联合训练：结果及失败边界','',
        '日期：2026-09-19。report159的batch16稳定队列已于00:12完整结束。两组各30轮、3000次真实优化器更新；最佳EMA独立复评12指标精确一致；源码、配置、源权重和标注SHA256复核通过。batch32此前的访问异常不属于本次结果。','',
        '## 一、主要指标','',
        '| 方案 | 最佳轮次（从1计） | AP | AP50 | AP75 | APS | APM |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label,epoch,metrics in [('原C+既有SAM RGB起点','—',baseline),('同预算RGB联合续训',r['best_epoch_zero_based']+1,r['best_metrics']),('增强局部对齐融合',f['best_epoch_zero_based']+1,f['best_metrics'])]:
        lines.append(f'| {label} | {epoch} | '+' | '.join(f'{metrics[i]*100:.4f}' for i in (0,1,2,3,4))+' |')
    lines += ['',f"增强融合相对同预算RGB最佳AP仅+{result['best_delta_points']['AP']:.4f}个百分点，APS则{result['best_delta_points']['APS']:+.4f}个百分点。两组最佳AP都低于起点；不能宣布成功，亦未超过历史完整模型53.6717。APS为各自最佳总体AP权重的APS，不单独挑小目标最优权重。",
        '', '## 二、后期变化比最佳单点更值得关注','',
        f"融合末轮AP {f['last_metrics'][0]*100:.4f}、APS {f['last_metrics'][3]*100:.4f}；RGB末轮AP {r['last_metrics'][0]*100:.4f}、APS {r['last_metrics'][3]*100:.4f}。两组30轮中超过原起点AP的轮数分别为{f['epochs_above_initial_ap']}、{r['epochs_above_initial_ap']}。",
        f"同轮比较融合AP胜出{result['same_epoch']['ap_wins']}/30轮，APS胜出{result['same_epoch']['aps_wins']}/30轮；同轮平均AP差{result['same_epoch']['mean_ap_delta_points']:+.4f}个百分点。",
        f"融合训练总损失{f['train_loss_first_last'][0]:.4f}→{f['train_loss_first_last'][1]:.4f}，训练偏移加权损失{f['field_loss_first_last'][0]:.4f}→{f['field_loss_first_last'][1]:.4f}；RGB训练总损失{r['train_loss_first_last'][0]:.4f}→{r['train_loss_first_last'][1]:.4f}。训练改善而开发集下降，符合过拟合或原有表征被破坏的表现，但尚未以梯度冲突/参数漂移等诊断区分根因。偏移训练损失降低不等于测试对齐精度改善，本轮未重做几何测试。",
        '', '## 三、能确定与不能确定的结论','',
        '1. 开放权重并增加180万融合参数未解决当前问题；前次关于冻结的解释只是待检验假设，现在不能再把冻结当唯一原因。',
        '2. RGB对照自身退化，说明本轮共同续训设置也需要检视：成熟C+SAM起点在没有新增SAM监督、全部可学习主干开放后继续训练，不保证原有泛化能力被保留。尚不能认定一定是SAM遗忘或某个学习率单项造成。',
        '3. 融合末期比同预算RGB退化更多，说明本次融合与训练组合存在额外问题，不能仅用共同续训退化解释全部损失。',
        '4. 本轮同时调整融合位置、结构与开放训练，尚未分离每个因素；无多种子，不把最佳+0.0735百分点当稳定有效证据。',
        '5. 本轮没有新SAM形状监督，不能据此宣布SAM无效；也不能声称已复现或否定MBUDet原论文。1820张原test在项目中被用作开发验证集，非未触碰最终测试。',
        '', '## 四、下一步建议（尚未启动新训练）','',
        '不建议续训本轮或立即再扩大模块。优先保留成熟RGB空间表征，继续允许编码器/解码器适配；通过分阶段开放或RGB输出保持约束，先建立不会自行退化的RGB训练对照。若需要定位本轮额外退化，先对最佳和末轮做同权重补充分支关闭诊断，明确结果只是机制诊断，不能作为部署方案或补偿支路的独立因果证明。之后再选择局部融合、偏移监督或SAM形状选择的单项修改。',
        '本轮到结果分析为止，没有擅自启动新一轮训练。',
        '', '## 五、文件','',
        '- 权威机器结果：reports/159_local_alignment_joint_stable16/conclusion.json',
        '- 曲线：reports/159_local_alignment_joint_stable16/training_comparison.png',
        '- 两组输出：outputs/M_LOCAL_ALIGN_JOINT_FUSION_B16A2_30E_TESTDEV/seed0 与 M_LOCAL_ALIGN_JOINT_RGB_B16A2_30E_TESTDEV/seed0。']
    DOC.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
