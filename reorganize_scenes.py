#!/usr/bin/env python3
"""Flatten TbV shard directories by atomic same-volume log moves.

No sensor file is copied, decoded, resampled, or deleted. Original path-bearing
manifests and SQLite log metadata are archived before refreshing their paths.
"""
import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import stat
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAW = Path('/Volumes/初棠/Argoverse 2')
STEP1 = BASE/'step01/output_real_v1'
STEP2 = BASE/'step02_clipped_v1'
OUT = BASE/'step03_scene_layout_v1'
CAMERAS = ('ring_front_center','ring_front_left','ring_front_right',
           'ring_rear_left','ring_rear_right','ring_side_left','ring_side_right')

def readj(p): return json.loads(Path(p).read_text())
def lines(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()

def atomic_text(p, text):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.migration-tmp')
    with tmp.open('w') as f:
        f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def writej(p, value): atomic_text(p,json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def fingerprint(p):
    s=p.lstat()
    if not (stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode)):
        raise ValueError(f'Unexpected non-regular entry: {p}')
    result={'device':s.st_dev,'inode':s.st_ino,'is_directory':stat.S_ISDIR(s.st_mode)}
    if not result['is_directory']:result.update(size=s.st_size,mtime_ns=s.st_mtime_ns)
    return result

def snapshot(log):
    paths=[log,log/'sensors',log/'sensors/cameras',log/'sensors/lidar',
           *[log/'sensors/cameras'/c for c in CAMERAS],log/'map',log/'calibration']
    paths.extend(x for x in log.iterdir() if not x.is_dir())
    for folder in ['map','calibration']:
        paths.extend((log/folder).rglob('*'))
    return {str(p.relative_to(log)):fingerprint(p) for p in paths}

def assert_original_entries(log, expected):
    for rel, record in expected.items():
        if fingerprint(log/rel)!=record:raise ValueError(f'Original entry changed: {log/rel}')

def move_log(source, destination, expected):
    source=Path(source);destination=Path(destination)
    if destination.exists() or destination.is_symlink():
        if source.exists():raise ValueError(f'Destination collision: {destination}')
        assert_original_entries(destination,expected)
        return 'already_moved'
    assert_original_entries(source,expected)
    if source.stat().st_dev!=destination.parent.stat().st_dev:
        raise ValueError('Cross-device move is prohibited')
    os.rename(source,destination)
    assert_original_entries(destination,expected)
    return 'moved'

def group_for(row, selected):
    if selected and selected['status']=='protected_review':return 'protected_review'
    if row['bucket']=='real_change':return 'real_test'
    if row['bucket']=='constructed_pool':return 'constructed_'+row['argotweak_split']
    if row['bucket']=='real_no_change_control':return 'control_no_change'
    if row['bucket']=='unlabeled':return 'unlabeled'
    raise ValueError('Unexpected selection bucket')

def rewrite(value, moves, annotations):
    if isinstance(value,dict):return {k:rewrite(v,moves,annotations) for k,v in value.items()}
    if isinstance(value,list):return [rewrite(v,moves,annotations) for v in value]
    if isinstance(value,str):
        if value in annotations:return annotations[value]
        # All source log directories have the same depth below the dataset root.
        if value.startswith(str(RAW)+'/TbV_v1.0_shard'):
            tail=value[len(str(RAW))+1:].split('/',2)
            if len(tail)>=2:
                old=str(RAW)+'/'+tail[0]+'/'+tail[1]
                if old in moves:return moves[old]+('/'+tail[2] if len(tail)==3 else '')
    return value

def replace_link(link, expected_source, destination):
    if not link.is_symlink():raise ValueError(f'Expected an existing grouping symlink: {link}')
    target=Path(os.path.abspath(link.parent/os.readlink(link)))
    if target not in (Path(expected_source),Path(destination)):
        raise ValueError(f'Unexpected symlink target: {link}: {target}')
    tmp=link.with_name(link.name+'.migration-link')
    if tmp.is_symlink():tmp.unlink()
    tmp.symlink_to(os.path.relpath(destination,link.parent),target_is_directory=True)
    os.replace(tmp,link)

def prepare():
    if (OUT/'plan.json').exists():return readj(OUT/'plan.json')
    selected={r['log_id']:r for r in lines(STEP2/'curated_logs.jsonl')}
    clips=defaultdict(list)
    for pool in ['real_test','constructed_train','constructed_val']:
        for clip in lines(STEP2/f'{pool}_clips.jsonl'):clips[clip['log_id']].append(clip)
    rows=lines(STEP1/'all_logs.jsonl')
    assert len(rows)==len({r['log_id'] for r in rows})==1043
    known={r['log_id'] for r in rows}
    shards=sorted(RAW.glob('TbV_v1.0_shard*'))
    assert len(shards)==21
    expected_sources={Path(r['log_dir']) for r in rows}
    actual_sources=set()
    for shard in shards:
        for child in shard.iterdir():
            if child.name not in known and child.name!='.DS_Store':
                raise ValueError(f'Unrecognized shard entry: {child}')
            if child.name in known:actual_sources.add(child)
    if actual_sources!=expected_sources:raise ValueError('Shard contents do not exactly match the 1043 source logs')
    def inspect(row):
        source=Path(row['log_dir']);dest=RAW/row['log_id'];sel=selected.get(row['log_id'])
        if source.resolve()!=source or source.parent not in shards or source.name!=row['log_id']:
            raise ValueError(f'Invalid source: {source}')
        if dest.exists() or dest.is_symlink():raise ValueError(f'Destination exists: {dest}')
        for name in ['clips.json','scene_info.json','argotweak_annotation.json']:
            if (source/name).exists():raise ValueError(f'New metadata name already exists: {source/name}')
        group=group_for(row,sel)
        if group in ['control_no_change','unlabeled']:link=STEP2/'preserved'/group/row['log_id']
        elif group=='protected_review':link=STEP2/'dataset/protected_review'/('constructed_'+row['argotweak_split'])/row['log_id']
        else:link=STEP2/'dataset'/group/row['log_id']
        if not link.is_symlink() or link.resolve()!=source:raise ValueError(f'Bad existing link: {link}')
        annotation=row.get('argotweak_path')
        return {'log_id':row['log_id'],'source':str(source),'destination':str(dest),
                'group':group,'row':row,'selected':sel,'clips':clips[row['log_id']],
                'original_entries':snapshot(source),'annotation_sha256':sha(annotation) if annotation else None,
                'group_link':str(link)}
    entries=[]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for entry in executor.map(inspect,rows):
            entries.append(entry)
            if len(entries)%100==0:print(json.dumps({'preflight_logs':len(entries),'total':len(rows)}),flush=True)
    plan={'created_utc':now(),'raw_root':str(RAW),'entries':entries,'shards':[str(s) for s in shards],
          'groups':dict(Counter(e['group'] for e in entries)),'sensor_operation':'same_volume_directory_rename_only'}
    OUT.mkdir(parents=True,exist_ok=True)
    # Archive only path-bearing small manifests and the SQLite rows being edited.
    for folder in [STEP1,STEP2]:
        for file in folder.glob('*.jsonl'):
            dest=OUT/'before'/file.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
    with sqlite3.connect(STEP2/'file_ledger.sqlite') as db:
        writej(OUT/'before/sqlite_logs_and_meta.json',{'logs':db.execute('SELECT log_id,data FROM logs').fetchall(),
              'meta':db.execute('SELECT key,value FROM meta').fetchall()})
    shutil.copy2(STEP2/'README_筛选结果.md',OUT/'before/README_筛选结果.md')
    writej(OUT/'plan.json',plan)
    print(json.dumps({'preflight_complete':True,'groups':plan['groups']}),flush=True)
    return plan

def metadata_for(entry):
    row=entry['row'];selected=entry['selected'];available=bool(row.get('argotweak_path'))
    clips=[]
    for c in entry['clips']:
        item=dict(c);item['log_dir']='.';item.pop('dataset_log_dir',None)
        item['argotweak_annotation_path']='argotweak_annotation.json';clips.append(item)
    status=selected['status'] if selected else ('retained_control' if entry['group']=='control_no_change' else 'unlabeled_unfiltered')
    info={'layout_version':1,'scene_id':entry['log_id'],'group':entry['group'],'status':status,
          'tbv_split':row['tbv_split'],'argotweak_split':row.get('argotweak_split'),
          'change_origin':'constructed' if row['bucket']=='constructed_pool' else ('real' if row['bucket']=='real_change' else 'not_asserted'),
          'source_shard':row['shard'],'original_log_path':entry['source'],
          'files':{'sensors':'sensors/','map':'map/','calibration':'calibration/',
                   'ego_poses':'city_SE3_egovehicle.feather','clips':'clips.json',
                   'argotweak_annotation':'argotweak_annotation.json' if available else None},
          'argotweak_annotation_sha256':entry['annotation_sha256'],
          'original_tbv_map_role':'current_base_map' if row['tbv_split']=='train' else 'prior_map',
          'separate_prior_and_current_gt_exports':False,
          'map_note':'map/ 保留原始 TbV 地图及高度栅格；ArgoTweak 保留原始编辑标注，不能直接按 old/new 字段名称拆成新旧地图。',
          'tracking_clip_count':len(clips),'review_reasons':selected.get('review_reasons',[]) if selected else [],
          'annotation_status':'available' if available else 'no_public_argotweak_annotation'}
    spans=selected.get('intervals_ns',[]) if selected and not clips else []
    clip_doc={'scene_id':entry['log_id'],'status':status,'clips':clips,
              'tracking_rule':'每个 clip 独立处理，不跨两个 clip 的时间空档延续目标轨迹；保留帧不等于全部为变化阳性。',
              'unvalidated_spans_ns':spans,
              'note':'有可用连续片段索引。' if clips else '整段保留，尚无通过连续性检查的片段索引；不得将整段默认视为无缺帧序列。'}
    return info,clip_doc

def apply(plan):
    moves={e['source']:e['destination'] for e in plan['entries']}
    annotations={e['row']['argotweak_path']:str(Path(e['destination'])/'argotweak_annotation.json')
                 for e in plan['entries'] if e['row'].get('argotweak_path')}
    with (OUT/'move_journal.jsonl').open('a') as journal:
        for i,e in enumerate(plan['entries'],1):
            result=move_log(e['source'],e['destination'],e['original_entries']);dest=Path(e['destination'])
            source_annotation=e['row'].get('argotweak_path')
            if source_annotation:
                if sha(source_annotation)!=e['annotation_sha256']:raise ValueError('Source annotation changed')
                target=dest/'argotweak_annotation.json'
                if target.exists():
                    if sha(target)!=e['annotation_sha256']:raise ValueError('Scene annotation collision')
                else:
                    tmp=dest/'argotweak_annotation.json.migration-tmp';shutil.copy2(source_annotation,tmp)
                    if sha(tmp)!=e['annotation_sha256']:raise ValueError('Annotation copy mismatch')
                    os.replace(tmp,target)
            info,clips=metadata_for(e);writej(dest/'scene_info.json',info);writej(dest/'clips.json',clips)
            replace_link(Path(e['group_link']),e['source'],e['destination'])
            journal.write(json.dumps({'log_id':e['log_id'],'result':result,'timestamp':now()})+'\n');journal.flush()
            if i%50==0:os.fsync(journal.fileno());print(json.dumps({'organized_logs':i,'total':len(plan['entries'])}),flush=True)
    # Rebuild working manifests from the archived originals; reruns are idempotent.
    for archive in (OUT/'before').rglob('*.jsonl'):
        relative=archive.relative_to(OUT/'before')
        destination=BASE/relative
        data=[rewrite(r,moves,annotations) for r in lines(archive)]
        atomic_text(destination,''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    original=readj(OUT/'before/sqlite_logs_and_meta.json')
    with sqlite3.connect(STEP2/'file_ledger.sqlite',timeout=60) as db:
        for log_id,data in original['logs']:
            db.execute('UPDATE logs SET data=? WHERE log_id=?',(json.dumps(rewrite(json.loads(data),moves,annotations)),log_id))
        db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',('scene_layout_migration',json.dumps({'root':str(RAW),'plan':str(OUT/'plan.json'),'updated_utc':now()})))
    for shard_path in plan['shards']:
        shard=Path(shard_path)
        if not shard.exists():continue
        for child in shard.iterdir():
            if child.name!='.DS_Store':raise ValueError(f'Shard is not empty after moves: {child}')
            dest=OUT/'shard_finder_metadata'/shard.name/child.name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():raise ValueError('Finder metadata archive collision')
            os.rename(child,dest)
        shard.rmdir()
    index=[{'scene_id':e['log_id'],'scene_dir':e['destination'],'relative_scene_dir':e['log_id'],
            'group':e['group'],'tbv_split':e['row']['tbv_split'],'argotweak_split':e['row'].get('argotweak_split'),
            'tracking_clip_count':len(e['clips']),'argotweak_available':bool(e['row'].get('argotweak_path'))} for e in plan['entries']]
    atomic_text(RAW/'scene_index.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in index))
    readme='''# 按原始场景整理的 TbV + ArgoTweak 数据

每个场景 ID 目录都是实际数据文件夹，不再按 21 个下载分包组织。
共 1,043 个原始场景：真实变化 105、人工构造训练 693、人工构造验证 102、待复核 4、无变化对照 6、标签未知 133。
其中 900 个可用场景包含 930 个连续片段；一个场景可能有多个片段，追踪不能跨片段空档。

每个场景包含原有 sensors/、map/、calibration/、city_SE3_egovehicle.feather，以及新增 scene_info.json、clips.json。
有 ArgoTweak 对应标注的 910 个场景还包含 argotweak_annotation.json，复制内容经过 SHA-256 比对。
133 个未知场景没有公开对应标注，文件缺省并在 scene_info.json 明确标记。

scene_info.json 的 group 标明分组；clips.json 使用纳秒字符串列出可用连续片段及应排除的同步参考帧。
待复核、对照和未知场景完整保留，clips 为空表示未提供通过本轮连续性检查的片段，不能理解成没有传感器数据。
map/ 是原始 TbV 地图和地面高度栅格；本次未另外生成 prior_map/current_gt_map，不能直接把 ArgoTweak 的 old/new 字段当作历史时间方向。
真值标注用于训练监督和评价，不能作为在线检测器的答案输入。

scene_index.jsonl 是全场景目录索引。已有分组入口仍位于：
/Volumes/初棠/HDMap_Change_Prep/step02_clipped_v1/dataset/
/Volumes/初棠/HDMap_Change_Prep/step02_clipped_v1/preserved/
分组入口是软链接；本目录中的场景文件夹是实际数据，分享时可复制整个场景文件夹。

本次仅移动完整场景目录、集中标注并更新索引，没有进一步裁剪、删除或改写传感器文件。
迁移方案、原路径映射、原索引备份和核验结果：
/Volumes/初棠/HDMap_Change_Prep/step03_scene_layout_v1/
'''
    atomic_text(RAW/'README_按场景组织.md',readme)
    old=(OUT/'before/README_筛选结果.md').read_text()
    old=old.replace('实际数据仍在 `/Volumes/初棠/Argoverse 2/`，且已物理裁剪。',
                    '实际数据现位于 `/Volumes/初棠/Argoverse 2/场景ID/`，且已完成前次物理裁剪和本次目录迁移。')
    old=old.replace('并同时带上对应 ArgoTweak 标注。','；对应 ArgoTweak 标注现在已放入场景文件夹。')
    old='> 目录已更新：21 个 shard 层级已移除。每个原始场景现在位于 `/Volumes/初棠/Argoverse 2/场景ID/`。分组链接和 JSONL 路径已更新。\n\n'+old
    atomic_text(STEP2/'README_筛选结果.md',old)
    print('MIGRATION_APPLIED',flush=True)

def verify(plan):
    errors=[];annotation_count=0;clip_count=0
    for i,e in enumerate(plan['entries'],1):
        dest=Path(e['destination'])
        try:
            assert_original_entries(dest,e['original_entries'])
            assert not dest.is_symlink() and not Path(e['source']).exists()
            assert Path(e['group_link']).is_symlink() and Path(e['group_link']).resolve()==dest
            info,clips=metadata_for(e)
            assert readj(dest/'scene_info.json')==info and readj(dest/'clips.json')==clips
            if e['annotation_sha256']:
                assert sha(dest/'argotweak_annotation.json')==e['annotation_sha256'];annotation_count+=1
            clip_count+=len(clips['clips'])
        except Exception as exc:errors.append({'log_id':e['log_id'],'error':repr(exc)})
        if i%100==0:print(json.dumps({'verified_scene_layouts':i,'errors':len(errors)}),flush=True)
    source_rows=lines(STEP1/'all_logs.jsonl')
    assert len(source_rows)==1043
    for row in source_rows:
        assert Path(row['log_dir'])==RAW/row['log_id'] and Path(row['log_dir']).is_dir()
        if row.get('argotweak_path'):assert Path(row['argotweak_path']).is_file()
        for mp in row.get('vector_map_paths',[]):assert Path(mp).is_file()
    for pool in ['real_test','constructed_train','constructed_val']:
        for c in lines(STEP2/f'{pool}_clips.jsonl'):
            assert Path(c['log_dir'])==RAW/c['log_id'] and Path(c['dataset_log_dir']).resolve()==Path(c['log_dir'])
            assert Path(c['argotweak_annotation_path'])==Path(c['log_dir'])/'argotweak_annotation.json'
    with sqlite3.connect(STEP2/'file_ledger.sqlite') as db:
        db_rows=db.execute('SELECT log_id,data FROM logs').fetchall();assert len(db_rows)==904
        for log_id,data in db_rows:
            row=json.loads(data);assert row['log_dir']==str(RAW/log_id)
            assert row['argotweak_path']==str(RAW/log_id/'argotweak_annotation.json')
    assert not list(RAW.glob('TbV_v1.0_shard*'))
    actual={x.name for x in RAW.iterdir() if x.is_dir() and '__' in x.name}
    assert actual=={e['log_id'] for e in plan['entries']}
    assert annotation_count==910 and clip_count==930
    result={'passed':not errors,'verified_utc':now(),'scene_root':str(RAW),'physical_scene_directories':len(actual),
            'groups':plan['groups'],'annotation_copies_sha256_verified':annotation_count,'usable_clips':clip_count,
            'old_shard_directories_remaining':0,'sqlite_log_paths_updated':len(db_rows),'errors':errors,
            'verification_method':'Original log, sensor-stream, map and calibration directory inodes preserved; original metadata file inode/size/mtime preserved; annotation hashes and all grouping links checked. Same-volume directory rename, no sensor file operations.',
            'original_entries_checked':sum(len(e['original_entries']) for e in plan['entries'])}
    writej(OUT/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if errors:raise ValueError('Migration verification failed')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['plan','apply','verify'])
    args=parser.parse_args()
    plan=prepare() if args.mode=='plan' else readj(OUT/'plan.json')
    if args.mode=='apply':apply(plan)
    if args.mode=='verify':verify(plan)
