"""Summarize completed report158, cached metrics and read-only geometry audits."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('E:/two_paper'); REPORT=ROOT/'reports/158_mbudet_rgb_alignment'
DOC=ROOT/'knowledge/M系列知识库/09_显式目标对齐与SAM形状约束/04_首轮结果分析与下一步裁决.md'
NAMES=['AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL']


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    assert not (REPORT/'conclusion.json').exists()
    state=read(REPORT/'queue_status.json'); assert state['status']=='complete'
    manifest=read(REPORT/'manifest.json')
    for path,expected in manifest.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    compare=read(REPORT/'comparison.json'); initial=read(REPORT/'initial/initial_verified.json')
    assert initial['status']=='PASS'; reference=np.array(initial['metrics'])
    best_geo=read(REPORT/'learned_geometry_audit.json'); last_geo=read(REPORT/'learned_geometry_last.json')
    assert best_geo['status']==last_geo['status']=='PASS'
    assert best_geo['epoch']==3 and last_geo['epoch']==19
    curves={}; results={}
    for arm,value in compare.items():
        folder=Path(value['output_dir'])
        rows=[json.loads(s) for s in (folder/'log.txt').read_text().splitlines() if s.strip()]
        assert [r['epoch'] for r in rows]==list(range(20))
        verification=read(folder/'eval_verified.json'); fixed=read(folder/'fixed_state_verified.json')
        off=read(folder/'adapter_bypass_verified.json')
        assert verification['status']==fixed['status']==off['status']=='PASS'
        assert fixed['actual_updates']==2000
        best=max(rows,key=lambda r:r['test_coco_eval_bbox'][0])
        assert verification['metrics']==best['test_coco_eval_bbox']==value['best_metrics']
        assert off['metrics']==initial['metrics']
        curves[arm]=np.array([r['test_coco_eval_bbox'] for r in rows])
        results[arm]=dict(best_epoch=best['epoch'],best_metrics=dict(zip(NAMES,value['best_metrics'])),
            best_minus_rgb_percentage_points=dict(zip(NAMES,((np.array(value['best_metrics'])-reference)*100).tolist())),
            last3_minus_rgb_percentage_points=dict(zip(NAMES,((curves[arm][-3:].mean(0)-reference)*100).tolist())),
            rounds_above_rgb_AP=int((curves[arm][:,0]>reference[0]).sum()),
            rounds_above_rgb_APS=int((curves[arm][:,3]>reference[3]).sum()),
            end_telemetry=read(folder/'training_progress.json'), independent_eval_exact=True,fixed_streams_exact=True)
    diff=(curves['aligned']-curves['unaligned'])*100
    paired=dict(mean_percentage_point_difference=dict(zip(NAMES,diff.mean(0).tolist())),
        last3_percentage_point_difference=dict(zip(NAMES,diff[-3:].mean(0).tolist())),
        aligned_wins_AP=int((diff[:,0]>0).sum()),aligned_wins_APS=int((diff[:,3]>0).sum()))
    ir_folder=ROOT/'outputs/M_IR_RAW_VERIFIED_GQ1_30E_TESTDEV/seed0'
    ir_rows=[json.loads(s) for s in (ir_folder/'log.txt').read_text().splitlines() if s.strip()]
    assert [r['epoch'] for r in ir_rows]==list(range(30))
    assert read(ir_folder/'training_complete.json')['actual_updates']==3000
    ir=read(ir_folder/'independent_best_ema.json'); assert ir['status']=='PASS'
    # Size mismatch is a secondary limitation, computed ONLY from train annotations.
    boxes={}
    for name,path in [('rgb',ROOT/'data/antiuav6k_common/annotations/instances_visible_common_train.json'),
                      ('ir',ROOT/'data/antiuav6k_ir_raw_verified/annotations/instances_infrared_train.json')]:
        data=read(path); images={v['id']:v['file_name'] for v in data['images']}
        boxes[name]={images[a['image_id']]:np.array(a['bbox'][2:]) for a in data['annotations']}
    ratios=np.stack([boxes['ir'][n]/boxes['rgb'][n] for n in boxes['rgb'] if n in boxes['ir']])
    conclusion=dict(status='ANALYZED',training_complete=True,manifest_rechecked=True,
        initial_rgb_metrics=dict(zip(NAMES,reference.tolist())),arms=results,paired_rounds=paired,
        correct_ir_source=ir,geometry_best=best_geo['summary'],geometry_last=last_geo['summary'],
        train_IR_over_RGB_size_ratio=dict(median_wh=np.median(ratios,axis=0).tolist(),
            p90_wh=np.quantile(ratios,.9,axis=0).tolist(),within_20_percent_both=float(((ratios>.8)&(ratios<1.2)).all(1).mean())),
        decision_zh='本次固定成熟表示的显式对齐适配未建立AP净增益，不续训或原样扫参；粗对齐有几何改善，但精度与目标区域使用仍不足。',
        sam_claim_zh='无新增SAM形状监督，不证明SAM新机制成功或失败。',
        hypotheses_zh=['残余位置误差可能造成目标与邻近背景混读。',
            '全画布残差融合、非目标处无位置监督可能扩大背景扰动。',
            '与原版MBUDet相比，全固定encoder/decoder、encoder后融合可能限制检测适配。',
            '平移未处理尺度差异；目前只有统计支持，不能断言是主要根因。'],
        next_zh='先用相同权重的有限诊断区分位置精度与融合接口，再设计局部内容选择及SAM前景/边界约束；不凭当前微小APS改善自动启动SAM训练。')
    (REPORT/'conclusion.json').write_text(json.dumps(conclusion,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for index,title in [(0,'AP'),(3,'Small-object AP')]:
        ax=axes[0 if index==0 else 1]
        for arm,v in curves.items(): ax.plot(range(20),v[:,index]*100,marker='.',label=arm)
        ax.axhline(reference[index]*100,color='gray',linestyle='--',label='fixed RGB')
        ax.set(xlabel='Epoch (zero-based)',ylabel=title,title=title); ax.legend(fontsize=8); ax.grid(alpha=.2)
    for label,geo in [('No alignment',None),('Best AP epoch 3',best_geo),('Last epoch 19',last_geo)]:
        values=[best_geo['summary']['test']['raw_center_error_pixels']['p50']]*2 if geo is None else [v['learned_center_error_pixels']['p50'] for v in geo['summary']['test']['levels']]
        axes[2].plot([16,32],values,marker='o',label=label)
    axes[2].set(xlabel='Stride',ylabel='Median center error (pixels)',title='GT-center geometry diagnosis',xticks=[16,32])
    axes[2].legend(fontsize=8); axes[2].grid(alpha=.2)
    fig.tight_layout();fig.savefig(REPORT/'metric_and_geometry_curves.png',dpi=180);plt.close(fig)
    lines=['# 显式目标对齐首轮结果分析与下一步裁决','',
        '日期：2026-09-18。原生Windows hello_word；正确框IR预训练30轮，两组融合各20轮，物理/有效batch32。',
        '三阶段18:35:16完整结束，没有中断；实际更新3000/2000/2000次。最佳EMA独立复评12项误差0，冻结流与旁路检查通过，保护清单SHA重新核验通过。','',
        '## 1. 主结论','',
        '本轮固定成熟表示、encoder后残差融合的MBUDet机制适配，没有建立检测AP净收益。显式对齐确实改善了几何读取位置，但尚未转化为比RGB底座更好的整体检测；不是原版论文复现失败，也不是SAM形状机制失败。','',
        '| 设置 | 最佳轮 | AP | AP50 | AP75 | APS | APM |',
        '|---|---:|---:|---:|---:|---:|---:|',
        '| 固定C＋已有SAM RGB底座 | 来源epoch17 | '+' | '.join(f'{reference[i]*100:.4f}' for i in range(5))+' |']
    for arm,v in compare.items(): lines.append(f"| {arm} | {v['best_epoch']} | "+' | '.join(f'{v["best_metrics"][i]*100:.4f}' for i in range(5))+' |')
    lines+=['','直接融合相对RGB AP仅+0.0502个百分点；对齐相对RGB AP−0.0157、APS+0.1601。对齐最佳AP低于直接融合0.0659个百分点，均未超过历史完整RGB-T候选53.6717。微小最佳点差异不能支持可靠性能提升，尤其未开展多种子。','',
        f"同轮比较：对齐AP高于直接融合{paired['aligned_wins_AP']}/20轮，APS高于{paired['aligned_wins_APS']}/20轮；平均AP差+{diff[:,0].mean():.4f}、APS差{diff[:,3].mean():+.4f}个百分点。末3轮对齐AP比直接融合+{diff[-3:,0].mean():.4f}，但相对固定RGB仍−{(reference[0]-curves['aligned'][-3:,0].mean())*100:.4f}个百分点。对齐组20轮AP没有一轮超过RGB底座。不能只报某一指标的正差。",'',
        '## 2. 几何是否真的学到了','',
        '只读诊断使用800个随机训练帧和全部1820个开发test；test1681帧双侧有框。模型读取两模态特征预测位移，GT仅选择评分位置和计算误差，没有用于生成推理位移。诊断在真实RGB目标中心评分，是比未知预测位置更有利的条件，不能当部署结果。','',
        '| test双侧正样本 | S16中位误差px | S32中位误差px | S16读取中心在IR框内 | S32读取中心在IR框内 |',
        '|---|---:|---:|---:|---:|',
        '| 不对齐 | 44.5410 | 44.5410 | 7.38% | 7.38% |']
    for label,geo in [('最佳AP轮epoch3',best_geo),('最后轮epoch19',last_geo)]:
        levels=geo['summary']['test']['levels']
        lines.append(f"| {label} | {levels[0]['learned_center_error_pixels']['p50']:.4f} | {levels[1]['learned_center_error_pixels']['p50']:.4f} | {levels[0]['read_center_inside_ir_gt_fraction']*100:.2f}% | {levels[1]['read_center_inside_ir_gt_fraction']*100:.2f}% |")
    lines+=['','最佳AP轮S16约91.14%的样本位置误差比不对齐小，最后轮约95.00%。说明新监督与采样方向有实质作用。可是最后轮S16仍约40.27%的读取中心不在IR框内，S32约34.56%；S16误差90分位69.67像素，尾部误差仍很大。框内仅是宽松必要诊断，位于框内也不保证读取真实机体边缘。','',
        '训练与test有精度差：最后轮训练S16/S32中位误差12.30/9.45像素，test15.46/11.40像素；训练框内比例70.69%/78.91%，test59.73%/65.44%。不是完全没学会，也不是已完成可靠精细对齐。','',
        '## 3. 为什么不建议原样续训','',
        '位置MSE加权项从epoch0的0.33385降至epoch19的0.13598，几何位置也继续改善；但AP最佳在epoch3，最后轮AP53.2823，低于底座。位置精度与检测指标脱节已经出现，不能期待只加epoch自然解决。',
        '最终训练batch残差RMS相对RGB约S16=14.45%、S32=21.80%；它不是完全没参与，也不能简单归因于3%上限过保守（本轮无该上限）。这些数是最后训练batch的遥测，不是全test统计。','',
        '## 4. 哪些是证据，哪些仍是解释假设','',
        '已确认：粗对齐改善、剩余偏差明显、检测AP收益未建立、固定表示无状态漂移。几何仍差可以导致背景混读，但尚未直接证明每一个AP损失由此造成。',
        '接口只对目标邻域监督位置，却在整幅特征上施加IR残差；非目标处的位移没有约束，可能扰动背景。冻结decoder与encoder有利于保留原RGB能力，却也限制对新的跨模态分布进行适配。原版MBUDet还训练参考支路/neck/head，本轮位置亦在encoder后，因此不能说原版方向被否定。',
        '两模态目标尺寸也不完全相同：训练IR/RGB宽高比中位1.1804/0.9992，90分位2.1503/1.5820，仅32.92%双侧宽高都在±20%范围。平移不处理尺寸变化，但这只是次要机制线索，不足以指定为根因。','',
        '## 5. 正确IR源的结果与限制','',
        f"正确IR框单模态最佳epoch{ir['best_epoch']}：AP{ir['metrics'][0]*100:.4f}、AP50{ir['metrics'][1]*100:.4f}、AP75{ir['metrics'][2]*100:.4f}、APS{ir['metrics'][3]*100:.4f}。后期AP下降而训练损失继续下降，不能说源提取器尚未收敛就原样延长。它能检测目标，但高IoU定位仍有限。",
        '这是IR原始正确框协议，不是最终RGB-T AP，也不能与旧偏移IR AP直接比较。新旧IR源、预算、数据增强和融合接口不同，本轮没有隔离标签修复的单独收益，不能把所有变化归因于标签修复。','',
        '## 6. 下一步','',
        '不续两组，不扫位置损失权重，不宣称SAM失效。本轮没有新增SAM形状监督，已有SAM只存在于固定RGB底座。',
        '先做有限、同权重诊断，判断更准确的位置读取是否还有检测潜力，以及全画布融合是否带来背景代价。若定位改善能帮助检测，下一方案应围绕“中心粗定位＋局部内容选择＋SAM目标/边界与邻近背景约束”建立，而不是继续直接整图平移后相加。',
        'SAM的明确任务是帮助区分正确目标内容与邻近背景，单側RGB轮廓不能直接监督原始未对齐IR，跨模态外形不强制逐像素一致。后续同结构框/高斯区域与真实SAM比较，才回答形状独特性。若更准确读取仍没有收益，则先调整检测融合接口，避免用SAM辅助损失掩盖接口问题。',
        '上述为下一阶段建议，本轮分析没有启动新训练。','',
        '## 7. 可读档产物','',
        '- 权威结论：reports/158_mbudet_rgb_alignment/conclusion.json。',
        '- 几何最佳/最后权重：learned_geometry_audit.json / learned_geometry_last.json。',
        '- 曲线：metric_and_geometry_curves.png。',
        '- 只读诊断工具：D-FINE/tools/audit_mbudet_learned_geometry.py；汇总工具：analyze_mbudet_alignment_results.py。',
        '- 训练快照与旧输出不改，新的分析工具位于当前tools，不写入冻结训练快照。']
    DOC.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in conclusion.items() if k in ('status','decision_zh','paired_rounds')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
