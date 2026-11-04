#!/usr/bin/env python3
"""Export clip indices and a human-readable report after successful verification."""
import csv
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
import clip_dataset as c

out=c.BASE/'step02_clipped_v1'
verification=c.jread(out/'verification.json')
if not verification['passed']:
    raise RuntimeError('Cannot publish an unverified result')
rows=list(c.jsonlines(out/'clip_manifest.jsonl'))
with (out/'curated_logs.jsonl').open('w') as f:
    for row in rows:
        final=dict(row)
        final['planning_status']=row['status']
        final['status']='protected_review' if row['status']=='protected_review' else 'cropped_verified'
        final['temporal_extent']='full_log_protected' if row['status']=='protected_review' else 'selected_intervals_ns'
        final['layout_check']='post_crop_inventory_and_sizes_verified'
        final['intervals_ns']=[[str(a),str(b)] for a,b in row['intervals_ns']]
        final['excluded_reference_lidar_timestamps_ns']=[str(t) for t in row.get('excluded_reference_lidar_timestamps_ns',[])]
        f.write(json.dumps(final,ensure_ascii=False)+'\n')
db=sqlite3.connect(out/'file_ledger.sqlite')
db.execute('PRAGMA cache_size=-65536')
clips=[]
for row in rows:
    if row['status']!='verified_plan':continue
    pool='real_test' if row['bucket']=='real_change' else 'constructed_'+row['argotweak_split']
    source_ts=db.execute('SELECT timestamp_ns,keep FROM files WHERE log_id=? AND stream="lidar" ORDER BY timestamp_ns',(row['log_id'],)).fetchall()
    ts=[t for t,keep in source_ts if keep]
    excluded=set(row.get('excluded_reference_lidar_timestamps_ns',[]))
    origin=source_ts[0][0]
    for i,(a,b) in enumerate(row['intervals_ns']):
        clips.append({'clip_id':f'{row["log_id"]}__clip{i:03d}', 'log_id':row['log_id'],
            'pool':pool,'tbv_split':row['tbv_split'],'argotweak_split':row['argotweak_split'],
            'start_ns':str(a),'end_ns':str(b),'start_relative_s':(a-origin)/c.NS,'end_relative_s':(b-origin)/c.NS,
            'duration_s':(b-a)/c.NS,
            'usable_lidar_reference_count':sum(a<=t<=b and t not in excluded for t in ts),
            'excluded_reference_lidar_timestamps_ns':[str(t) for t in sorted(excluded) if a<=t<=b],
            'log_dir':row['log_dir'],'dataset_log_dir':str(out/'dataset'/pool/row['log_id']),
            'argotweak_annotation_path':row['argotweak_path'],
            'selection':'distance_to_union_of_change_geometries_le_40m_plus_5s_each_side',
            'frame_label':'spatial_candidates_and_context_not_all_frames_positive'})
db.close()
for pool in ['real_test','constructed_train','constructed_val']:
    with (out/f'{pool}_clips.jsonl').open('w') as f:
        for clip in clips:
            if clip['pool']==pool:f.write(json.dumps(clip,ensure_ascii=False)+'\n')
with (out/'log_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=['log_id','pool','status','original_seconds','retained_seconds','deleted_files','deleted_bytes','review_reasons'])
    writer.writeheader()
    for row in rows:
        writer.writerow({'log_id':row['log_id'],'pool':row['bucket'],'status':'protected_review' if row['status']=='protected_review' else 'cropped_verified',
            'original_seconds':row['original_lidar_duration_s'],'retained_seconds':row['retained_window_duration_s'],
            'deleted_files':sum(x['delete'] for x in row['streams'].values()),
            'deleted_bytes':sum(x['delete_bytes'] for x in row['streams'].values()),
            'review_reasons':'; '.join(row['review_reasons'])})
for source,pool in [('real_no_change_controls.jsonl','control_no_change'),('unlabeled_logs.jsonl','unlabeled')]:
    for row in c.jsonlines(c.SOURCE/source):
        dest=out/'preserved'/pool/row['log_id'];dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():dest.symlink_to(os.path.relpath(row['log_dir'],dest.parent),target_is_directory=True)
summary=c.jread(out/'deletion_summary.json')
summary.update({'annotated_logs':len(rows),'usable_logs':sum(r['status']=='verified_plan' for r in rows),
    'protected_review_logs':sum(r['status']=='protected_review' for r in rows),
    'clip_counts':dict(Counter(x['pool'] for x in clips)),
    'shortened_lidar_logs':sum(r['retained_window_duration_s']<r['original_lidar_duration_s']-0.01 for r in rows),
    'retained_lidar_hours':sum(r['retained_window_duration_s'] for r in rows)/3600,
    'original_lidar_hours':sum(r['original_lidar_duration_s'] for r in rows)/3600,
    'source_boundary_reference_exclusions':sum(len(r.get('excluded_reference_lidar_timestamps_ns',[])) for r in rows),
    'preserved_controls':6,'preserved_unknown_logs':133,'verification_passed':True})
c.jwrite(out/'delivery_summary.json',summary)
text=f'''# 道路变化数据筛选与裁剪结果

本轮共处理 {len(rows)} 个有地图变化标注的场景：105 个真实变化场景、799 个人工构造变化场景。
已删除原始目录中的范围外传感器文件 {summary['deleted_files']:,} 个，逻辑文件大小合计 {summary['deleted_bytes']/1e9:.2f} GB（{summary['deleted_bytes']/1024**3:.2f} GiB）。这是实际删除，不是隐藏或移动到废纸篓。
共有 {summary['usable_logs']} 个场景通过本轮裁剪检查，{summary['protected_review_logs']} 个场景因传感器帧间断或时间匹配问题完整保留供复查。

| 分组 | 可用场景 | 连续片段 |
| --- | ---: | ---: |
| 真实变化测试集 | {verification['real_test_logs']} | {summary['clip_counts']['real_test']} |
| 人工构造训练集 | {verification['constructed_train_logs']} | {summary['clip_counts']['constructed_train']} |
| 人工构造验证集 | {verification['constructed_val_logs']} | {summary['clip_counts']['constructed_val']} |

## 保留规则

- 在每个 log 自己的 city 坐标系中，将 TbV 变化事件与 ArgoTweak 非零 changes 元素的 old/new 几何取并集。
- 保留自车距该区域不超过 40 米的连续时间段，再向前、向后各扩展 5 秒；原始录制不足时在序列边界截断。
- 扩展后间隔不超过 2 秒的片段合并。七路相机在边界额外保留 100 毫秒，支持最近帧匹配。
- 所有保留区间内的原始采样率不变：不抽帧，不重编码，不降采样。
- 某些场景的大部分行程都在变化区域附近，因此保留整段是筛选结果，并非漏处理。

## 直接交给队友的入口

`dataset/real_test/`：通过检查的真实变化数据。

`dataset/constructed_train/`、`dataset/constructed_val/`：人工构造的训练、验证数据，保持官方划分。

`real_test_clips.jsonl`、`constructed_train_clips.jsonl`、`constructed_val_clips.jsonl`：连续片段清单，每行给出起止时间、相对秒数、路径和可用 LiDAR 参考帧数。

`curated_logs.jsonl`：核验完成后的场景总清单，状态为 `cropped_verified` 或 `protected_review`。含所有保留区间及来源信息，纳秒时间戳使用字符串。

这些目录使用本磁盘内的相对软链接，实际数据仍在 `/Volumes/初棠/Argoverse 2/`，且已物理裁剪。不要只复制软链接目录给队友；分享时必须按清单复制实际数据并同时带上对应 ArgoTweak 标注。

ArgoTweak 人工构造样本的冲突依赖于它提供的先验地图。不能仅加载 TbV 原始 map 就期望看到同样冲突。`argotweak_annotation_path` 保留了正确的标注入口；old/new 的语义组合仍应遵循官方解析逻辑。

## 完整保留的配套数据

- 原始地图、地面高度栅格、完整自车位姿、相机内外参以及全部 ArgoTweak JSON 标注。
- 6 个无变化对照场景：`preserved/control_no_change/`，用于置信度与误报检查。
- 133 个缺少公开变化标签的场景：`preserved/unlabeled/`，未把未知当成正常行驶后删除。
- 需要复查的场景：`dataset/protected_review/` 和 `protected_review.jsonl`。
- `review_diagnostics.json` 记录这 4 个场景原始传感器序列的具体间断时间。
- nuPlan、nuScenes 和本磁盘其他目录未纳入本轮裁剪。

## 验证与使用边界

规划时逐项记录传感文件名、大小、修改时间和文件标识；删除前比对所有保留文件，并再次核验待删除文件。删除后重新枚举全部处理场景，核对保留清单、大小和删除状态，结果见 `verification.json`。
对各片段首、中、尾的七路 JPEG 检查格式头，对 LiDAR 样本读取 xyz 列；并检查时间戳对应的位姿以及相机最近帧时间差。这不等于逐张图像完整解码、逐帧点云内容复核或人工确认变化可见性。

少量原始序列首尾的 LiDAR 比相机早/晚不到 150 毫秒：保留这些数据作上下文，但将不满足 50 毫秒匹配阈值的参考帧列入 `excluded_reference_lidar_timestamps_ns`。读取片段时跳过它们作为同步参考帧。

保留片段包含变化候选和前后上下文，不能把所有保留帧都标记为变化阳性。下一步仍需做精细时空对齐、可见性判断、语义标注和动态目标追踪。
时间戳是纳秒级，按 int64 或字符串读取，不要经过 float/Excel 浮点数转换。JSONL 片段索引中起止纳秒已保存为字符串。

## 审计文件

- `delivery_summary.json`：本轮结果汇总。
- `delivery_audit.json`：最终目录链接、片段索引、分组数量与时间戳格式的检查结果。
- `log_summary.csv`：按场景比较原始/保留时长与删除量。
- `clip_manifest.jsonl`：删除前的场景级规划记录，保留原始规划状态；实际交付请用 `curated_logs.jsonl` 与三个片段索引。
- `file_ledger.sqlite`：逐文件保留/删除记录；files 表中 keep=0、deleted=1 为已删文件。
- `config.json`、`plan_summary.json`、`deletion_summary.json`、`verification.json`：参数及各阶段结果。

清单可以追溯删除了哪些文件，本身不包含已删数据的副本。
'''
(out/'README_筛选结果.md').write_text(text)
print(json.dumps(summary,ensure_ascii=False,indent=2))
