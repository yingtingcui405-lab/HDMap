#!/usr/bin/env python3
"""Conservative spatial/temporal curation with an explicit per-file deletion ledger.

Plan and verify retained files before apply. Only timestamped sensor files outside
retained intervals in the 904 annotated change logs may be unlinked. Maps, poses,
calibration, annotations, control logs, and unlabeled logs are never removed.
"""
import argparse
import contextlib
import stat
import bisect
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import shapely
from shapely.geometry import LineString, Polygon, MultiPoint
from PIL import Image

BASE = Path(__file__).resolve().parent
RAW = Path('/Volumes/初棠/Argoverse 2').resolve()
SOURCE = BASE / 'step01/output_real_v1'
CAMERAS = ('ring_front_center', 'ring_front_left', 'ring_front_right',
           'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right')
STREAMS = {'lidar': ('sensors/lidar', '.feather'),
           **{c: (f'sensors/cameras/{c}', '.jpg') for c in CAMERAS}}
NS = 1_000_000_000


def jread(p):
    return json.loads(Path(p).read_text())


def jwrite(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def jsonlines(p):
    with Path(p).open() as f:
        for line in f:
            yield json.loads(line)


def coords(points):
    a = np.array([[p['x'], p['y']] for p in points], dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise ValueError('Invalid geometry coordinates')
    return a


def element_geometry(layer, obj):
    if obj is None:
        return None
    if layer == 'laneSegments':
        left, right = coords(obj['left_lane_boundary']), coords(obj['right_lane_boundary'])
        p = Polygon(np.concatenate([left, right[::-1]]))
    elif layer == 'pedCrossings':
        p = MultiPoint(coords(obj['edges'])).convex_hull
    else:
        p = Polygon(coords(obj['area_boundary']))
    if p.is_empty:
        raise ValueError('Empty changed-element geometry')
    return shapely.make_valid(p)


def change_geometry(annotation, events):
    geometries, count = [], 0
    for layer in ('laneSegments', 'pedCrossings', 'drivableAreas'):
        for obj in annotation[layer].values():
            codes = obj['changes']
            if not isinstance(codes, list) or not codes or any(type(c) is not int for c in codes):
                raise ValueError('Invalid change code list')
            if codes == [0]:
                continue
            count += 1
            for side in ('old', 'new'):
                geom = element_geometry(layer, obj[side])
                if geom is not None:
                    geometries.append(geom)
    for event in events:
        if event['supercategory'] == 'crosswalk_change':
            geometries.append(shapely.make_valid(Polygon(event['polygon'])))
        elif event['supercategory'] == 'lane_geometry_change':
            points = [p['City Coords'] for p in event['change_endpoints']]
            geometries.append(LineString(points))
        else:
            raise ValueError('Unexpected TbV event category')
    if not geometries:
        raise ValueError('No changed geometry; preserve full log for review')
    return shapely.union_all(geometries), count


def merge_intervals(intervals, gap=0):
    merged = []
    for start, end in sorted(intervals):
        if start > end:
            raise ValueError('Inverted interval')
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([int(start), int(end)])
    return merged


def select_intervals(ts, distances, radius, padding, merge_gap):
    hit = ts[distances <= radius]
    if not len(hit):
        return []
    # Separate genuine temporal gaps before padding and merging the windows.
    chunks = np.split(hit, np.flatnonzero(np.diff(hit) > 500_000_000) + 1)
    return merge_intervals([[max(int(ts[0]), int(c[0]) - padding),
                             min(int(ts[-1]), int(c[-1]) + padding)] for c in chunks], merge_gap)


def in_intervals(ts, intervals):
    result = np.zeros(len(ts), dtype=bool)
    for start, end in intervals:
        result |= (ts >= start) & (ts <= end)
    return result


def directory_metadata(directory):
    helper=BASE/'bulk_inventory'
    if helper.is_file():
        output=subprocess.check_output([str(helper),str(directory)],text=True)
        result={}
        for line in output.splitlines():
            name,size,mtime,inode=line.split('\t')
            if name in result:raise ValueError('Duplicate directory entry')
            result[name]=(int(size),int(mtime),int(inode))
        return result
    result={}
    with os.scandir(directory) as entries:
        for e in entries:
            name=Path(e.name)
            if name.suffix not in ('.jpg','.feather') or not name.stem.isdigit():continue
            if not e.is_file(follow_symlinks=False):raise ValueError('Non-regular sensor file')
            st=e.stat(follow_symlinks=False)
            result[e.name]=(st.st_size,st.st_mtime_ns,st.st_ino)
    return result


def inventory(log):
    result={}
    for stream,(relative,suffix) in STREAMS.items():
        metadata=directory_metadata(log/relative)
        files=sorted((int(Path(name).stem),f'{relative}/{name}',*values)
                     for name,values in metadata.items() if Path(name).suffix==suffix)
        if not files or any(b[0]<=a[0] for a,b in zip(files,files[1:])):
            raise ValueError(f'Empty or duplicate-timestamp stream: {stream}')
        result[stream]=files
    return result


def boundary_checks(log, streams, masks, intervals, pose_ts):
    errors, samples = [], []
    for stream, files in streams.items():
        selected = [f for f, keep in zip(files, masks[stream]) if keep]
        if not selected or any(f[2] == 0 for f in selected):
            errors.append(f'{stream}: empty selected stream or zero-byte retained file')
            continue
        ts = np.array([f[0] for f in selected], dtype=np.int64)
        if not np.isin(ts, pose_ts).all():
            errors.append(f'{stream}: some retained timestamps have no exact pose')
        for start, end in intervals:
            sub = [f for f in selected if start <= f[0] <= end]
            if not sub:
                errors.append(f'{stream}: no data in a retained interval')
                continue
            gaps = np.diff([f[0] for f in sub])
            limit = 350_000_000 if stream == 'lidar' else 150_000_000
            if len(gaps) and np.max(gaps) > limit:
                errors.append(f'{stream}: retained interval has a long sensor gap')
            for f in {sub[0], sub[len(sub)//2], sub[-1]}:
                p = log / f[1]
                if stream == 'lidar':
                    table = feather.read_table(p, columns=['x', 'y', 'z'])
                    if not table.num_rows:
                        raise ValueError(f'Empty sampled point cloud: {p}')
                else:
                    with Image.open(p) as im:
                        im.verify()
                samples.append(f[1])
    # Check each retained lidar frame has a nearby retained frame in all cameras.
    lidar_ts = np.array([f[0] for f, k in zip(streams['lidar'], masks['lidar']) if k], dtype=np.int64)
    worst = {}
    excluded_reference_ns = set()
    for camera in CAMERAS:
        camera_ts = np.array([f[0] for f, k in zip(streams[camera], masks[camera]) if k], dtype=np.int64)
        if not len(camera_ts):
            continue
        idx = np.searchsorted(camera_ts, lidar_ts)
        delta = np.minimum(abs(camera_ts[np.clip(idx, 0, len(camera_ts)-1)] - lidar_ts),
                           abs(camera_ts[np.clip(idx-1, 0, len(camera_ts)-1)] - lidar_ts))
        worst[camera] = int(delta.max()) if len(delta) else None
        if len(delta) and delta.max() > 50_000_000:
            bad = delta > 50_000_000
            interior = (lidar_ts >= camera_ts[0]) & (lidar_ts <= camera_ts[-1])
            if np.any(bad & interior) or np.sum(bad) > 2 or delta.max() > 150_000_000:
                errors.append(f'{camera}: nearest-image delta exceeds 50 ms away from a minor source boundary')
            else:
                excluded_reference_ns.update(int(t) for t in lidar_ts[bad])
    for start, end in intervals:
        valid = [int(t) for t in lidar_ts if start <= t <= end and int(t) not in excluded_reference_ns]
        if len(valid) < 10:
            errors.append('Fewer than 10 synchronized lidar reference frames in an interval')
    return sorted(set(errors)), len(set(samples)), worst, sorted(excluded_reference_ns)


def process_log(row, events, args):
    log = Path(row['log_dir'])
    if log.resolve() != log or RAW not in log.parents or log.name != row['log_id']:
        raise ValueError(f'Unsafe log path: {log}')
    inv = inventory(log)
    pose_path = log / 'city_SE3_egovehicle.feather'
    poses = feather.read_table(pose_path, columns=['timestamp_ns', 'tx_m', 'ty_m']).to_pandas()
    pt = poses.timestamp_ns.to_numpy(dtype=np.int64)
    if len(pt) < 2 or np.any(np.diff(pt) <= 0):
        raise ValueError(f'Invalid pose timestamps: {log}')
    ts = np.array([f[0] for f in inv['lidar']], dtype=np.int64)
    intervals, errors, dmin, changed, samples, worst, excluded_ns = [], [], None, None, 0, {}, []
    try:
        if not np.isin(ts, pt).all():
            raise ValueError('Lidar timestamp missing exact pose')
        annotation = jread(row['argotweak_path'])
        if annotation['filename'] != log.name:
            raise ValueError('Annotation/log ID mismatch')
        geom, changed = change_geometry(annotation, events.get(log.name, []))
        xy = poses[['tx_m', 'ty_m']].to_numpy()[np.searchsorted(pt, ts)]
        distances = shapely.distance(shapely.points(xy), geom)
        if not np.isfinite(distances).all():
            raise ValueError('Non-finite spatial distances')
        dmin = float(np.min(distances))
        intervals = select_intervals(ts, distances, args.radius, int(args.padding*NS), 2*NS)
        if not intervals:
            raise ValueError('No lidar pose close enough to annotated change; preserve for review')
        masks = {}
        for stream, files in inv.items():
            guard = 100_000_000 if stream != 'lidar' else 0
            window = [[a-guard, b+guard] for a,b in intervals]
            masks[stream] = in_intervals(np.array([f[0] for f in files], dtype=np.int64), window)
        errors, samples, worst, excluded_ns = boundary_checks(log, inv, masks, intervals, pt)
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    if errors:
        intervals = [[int(ts[0]), int(ts[-1])]]
        masks = {s: np.ones(len(fs), dtype=bool) for s, fs in inv.items()}
    info = {**row, 'intervals_ns': intervals,
            'status': 'protected_review' if errors else 'verified_plan',
            'review_reasons': errors, 'min_distance_m': dmin,
            'changed_element_count': changed, 'sampled_files_checked': samples,
            'nearest_camera_delta_max_ns': worst, 'excluded_reference_lidar_timestamps_ns': excluded_ns, 'qa_version': 2,
            'original_lidar_duration_s': float((ts[-1]-ts[0])/NS),
            'retained_window_duration_s': sum((b-a)/NS for a,b in intervals),
            'streams': {}, 'metadata_sha256': {
                'poses': hashlib.sha256(pose_path.read_bytes()).hexdigest(),
                'argotweak': hashlib.sha256(Path(row['argotweak_path']).read_bytes()).hexdigest()}}
    entries = []
    for stream, files in inv.items():
        keep = masks[stream]
        info['streams'][stream] = {'before': len(files), 'keep': int(keep.sum()),
                                  'delete': int((~keep).sum()),
                                  'keep_bytes': sum(f[2] for f,k in zip(files,keep) if k),
                                  'delete_bytes': sum(f[2] for f,k in zip(files,keep) if not k)}
        entries.extend((log.name, f[1], stream, f[0], int(k), f[2], f[3], f[4]) for f,k in zip(files,keep))
    return info, entries


def make_plan(args):
    out = args.out.resolve()
    exists = out.exists()
    if exists and (out/'plan_summary.json').exists():
        raise ValueError('Completed plan already exists')
    out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(out / 'file_ledger.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS files (log_id TEXT, relative_path TEXT, stream TEXT, timestamp_ns INTEGER, keep INTEGER, size INTEGER, mtime_ns INTEGER, inode INTEGER, deleted INTEGER DEFAULT 0, PRIMARY KEY(log_id, relative_path))')
    db.execute('CREATE TABLE IF NOT EXISTS logs (log_id TEXT PRIMARY KEY, data TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
    config = {'raw_root': str(RAW), 'radius_m': args.radius, 'padding_s': args.padding,
              'merge_gap_s': 2, 'camera_boundary_guard_ms': 100, 'source': str(SOURCE),
              'scope': ['real_change', 'constructed_pool'], 'complete': False, 'qa_version': 2,
              'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if exists:
        old=jread(out/'config.json')
        for key in ['raw_root','radius_m','padding_s','source','scope']:
            if old[key]!=config[key]: raise ValueError('Resume configuration mismatch')
    jwrite(out / 'config.json', config)
    rows = [r for r in jsonlines(SOURCE / 'all_logs.jsonl') if r['bucket'] in config['scope']]
    events = {r['log_id']: r['events'] for r in jread(BASE / 'step01/metadata/tbv_val_split_annotations.json')}
    existing={log_id:json.loads(data) for log_id,data in db.execute('SELECT log_id,data FROM logs')}
    todo=[r for r in rows if r['log_id'] not in existing or existing[r['log_id']]['status']=='protected_review']
    done=len(rows)-len(todo)
    print(json.dumps({'resuming_verified_logs':done,'remaining':len(todo),'read_workers':4}),flush=True)
    # Independent read-only log planning; all ledger writes stay on this thread.
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending={}
        iterator=iter(todo)
        def submit_next():
            r=next(iterator,None)
            if r is not None: pending[r['log_id']]=executor.submit(process_log,r,events,args)
        for _ in range(min(4,len(todo))): submit_next()
        while pending:
            log_id=next(iter(pending))
            info,entries=pending.pop(log_id).result()
            db.execute('DELETE FROM files WHERE log_id=?',(log_id,))
            db.executemany('INSERT INTO files(log_id,relative_path,stream,timestamp_ns,keep,size,mtime_ns,inode) VALUES(?,?,?,?,?,?,?,?)',entries)
            db.execute('INSERT OR REPLACE INTO logs VALUES(?,?)',(log_id,json.dumps(info)))
            db.commit();existing[log_id]=info;done+=1
            submit_next()
            if done % 10 == 0 or done==len(rows):
                delete_bytes=sum(v['delete_bytes'] for r in existing.values() for v in r['streams'].values())
                print(json.dumps({'planned':done,'total':len(rows),'delete_GB':round(delete_bytes/1e9,2),
                                  'protected':sum(r['status']=='protected_review' for r in existing.values())}),flush=True)
    output_rows=[existing[r['log_id']] for r in rows]
    stats=Counter()
    for r in output_rows:
        stats[r['status']]+=1
        for v in r['streams'].values():
            stats['keep_files']+=v['keep'];stats['delete_files']+=v['delete']
            stats['keep_bytes']+=v['keep_bytes'];stats['delete_bytes']+=v['delete_bytes']
    config['complete']=True
    db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',('config',json.dumps(config)))
    db.commit();db.close();jwrite(out/'config.json',config)
    jwrite(out/'plan_summary.json',{'logs':len(rows),**stats,'by_pool':dict(Counter(r['bucket'] for r in output_rows)),
        'notes':'All selected files stat-checked; sample JPEG headers and point clouds checked. A few source-boundary lidar frames are retained as context but excluded as synchronized references.'})
    for name, rows2 in [('clip_manifest.jsonl', output_rows),
                        ('protected_review.jsonl', [r for r in output_rows if r['status']=='protected_review'])]:
        with (out / name).open('w') as f:
            for r in rows2: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print('PLAN_COMPLETE',out,flush=True)


def safe_sensor_path(log, rel, stream):
    prefix, suffix = STREAMS[stream]
    rp = Path(rel)
    if str(rp.parent) != prefix or rp.suffix != suffix or not rp.stem.isdigit():
        raise ValueError(f'Not an authorized sensor file: {rel}')
    p = log / rp
    if p.parent.resolve() != p.parent or RAW not in p.parents:
        raise ValueError(f'Path traverses a symlink or leaves dataset: {p}')
    return p


def sensor_basename(relative, stream):
    prefix,suffix=STREAMS[stream]
    rp=Path(relative)
    if str(rp.parent)!=prefix or rp.suffix!=suffix or not rp.stem.isdigit():
        raise ValueError(f'Invalid sensor path in ledger: {relative}')
    return rp.name


def apply_plan(args):
    out=args.out.resolve()
    with contextlib.closing(sqlite3.connect(out/'file_ledger.sqlite')) as db, ThreadPoolExecutor(max_workers=4) as reader:
        db.execute('PRAGMA cache_size=-65536')
        stored=db.execute('SELECT value FROM meta WHERE key="config"').fetchone()
        if not stored or not json.loads(stored[0])['complete']:
            raise ValueError('Incomplete plan; deletion prohibited')
        all_logs=db.execute('SELECT log_id,data FROM logs ORDER BY log_id').fetchall()
        done,deleted,bytes_freed=0,0,0
        for log_id,data in all_logs:
            info=json.loads(data);log=Path(info['log_dir'])
            if info['status']!='verified_plan':continue
            if log.resolve()!=log or RAW not in log.parents or log.name!=log_id:
                raise ValueError(f'Unsafe log path: {log}')
            for key,p in [('poses',log/'city_SE3_egovehicle.feather'),('argotweak',Path(info['argotweak_path']))]:
                if hashlib.sha256(p.read_bytes()).hexdigest()!=info['metadata_sha256'][key]:
                    raise ValueError(f'Decision metadata changed: {p}')
            candidates=db.execute('SELECT relative_path,stream,size,mtime_ns,inode FROM files WHERE log_id=? AND keep=0 AND deleted=0',(log_id,)).fetchall()
            # Open each verified sensor directory once, then stat/unlink by dir_fd.
            # This avoids millions of repeated parent traversals and prevents a
            # subsequent directory symlink swap from redirecting deletion.
            with contextlib.ExitStack() as stack:
                fds={}
                current_metadata={}
                streams={r[0] for r in db.execute('SELECT DISTINCT stream FROM files WHERE log_id=?',(log_id,))} if candidates else set()
                pending_metadata={}
                for stream in streams:
                    relative,suffix=STREAMS[stream]
                    parent=safe_sensor_path(log,f'{relative}/0{suffix}',stream).parent
                    fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
                    stack.callback(os.close,fd);fds[stream]=fd
                    pending_metadata[stream]=reader.submit(directory_metadata,parent)
                for stream,future in pending_metadata.items():
                    current_metadata[stream]=future.result()
                retained=db.execute('SELECT relative_path,stream,size,mtime_ns,inode FROM files WHERE log_id=? AND keep=1',(log_id,)) if candidates else []
                for rel,stream,size,mtime,ino in retained:
                    name=sensor_basename(rel,stream)
                    if current_metadata[stream].get(name)!=(size,mtime,ino):
                        raise ValueError(f'Retained file changed: {log/rel}')
                already_started=bool(info.get('apply_started'))
                info['apply_started']=True
                db.execute('UPDATE logs SET data=? WHERE log_id=?',(json.dumps(info),log_id));db.commit()
                for rel,stream,size,mtime,ino in candidates:
                    name=sensor_basename(rel,stream)
                    try:st=os.stat(name,dir_fd=fds[stream],follow_symlinks=False)
                    except FileNotFoundError:
                        if not already_started:raise
                        # Resume an interrupted, explicitly recorded application.
                        db.execute('UPDATE files SET deleted=1 WHERE log_id=? AND relative_path=?',(log_id,rel))
                        continue
                    if (st.st_size,st.st_mtime_ns,st.st_ino)!=(size,mtime,ino) or not stat.S_ISREG(st.st_mode):
                        raise ValueError(f'Deletion candidate changed: {log/rel}')
                    os.unlink(name,dir_fd=fds[stream])
                    db.execute('UPDATE files SET deleted=1 WHERE log_id=? AND relative_path=?',(log_id,rel))
                    deleted+=1;bytes_freed+=size
                    if deleted % 1000==0:db.commit()
            db.commit();done+=1
            if done % 10==0:
                print(json.dumps({'applied_logs':done,'deleted_files_this_run':deleted,'deleted_GB_this_run':round(bytes_freed/1e9,2)}),flush=True)
        summary=dict(zip(['deleted_files','deleted_bytes'],db.execute('SELECT COUNT(*),COALESCE(SUM(size),0) FROM files WHERE deleted=1').fetchone()))
        summary['pending_deletions']=db.execute('SELECT COUNT(*) FROM files WHERE keep=0 AND deleted=0').fetchone()[0]
        jwrite(out/'deletion_summary.json',summary)
        print('APPLY_COMPLETE',json.dumps(summary),flush=True)


def verify_plan(args):
    out=args.out.resolve();db=sqlite3.connect(out/'file_ledger.sqlite')
    db.execute('PRAGMA cache_size=-65536')
    errors=[];counts=Counter()
    dataset=out/'dataset'
    follow_log=getattr(args,'follow_apply_log',None)
    applied_sequence=0
    all_logs=db.execute('SELECT log_id,data FROM logs ORDER BY log_id').fetchall()
    for log_number,(log_id,data) in enumerate(all_logs,1):
        info=json.loads(data);log=Path(info['log_dir'])
        if follow_log and info['status']=='verified_plan':
            applied_sequence+=1
            while True:
                progress=Path(follow_log).read_text()
                completed=0
                for line in progress.splitlines():
                    try: completed=max(completed,json.loads(line).get('applied_logs',0))
                    except (ValueError,AttributeError):pass
                if completed>=applied_sequence:break
                if 'Traceback (most recent call last)' in progress:
                    raise RuntimeError('Application stopped; resume deletion before verification')
                time.sleep(5)
        expected_by_stream={stream:{} for stream in STREAMS}
        for relative_path,stream,size in db.execute('SELECT relative_path,stream,size FROM files WHERE log_id=? AND keep=1',(log_id,)):
            expected_by_stream[stream][relative_path]=size
        with ThreadPoolExecutor(max_workers=4) as reader:
            inventories=dict(zip(STREAMS,reader.map(directory_metadata,[log/relative for relative,suffix in STREAMS.values()])))
        for stream,(relative,suffix) in STREAMS.items():
            expected=expected_by_stream[stream]
            actual={f'{relative}/{name}':values[0] for name,values in inventories[stream].items()
                    if Path(name).suffix==suffix}
            if actual!=expected:
                errors.append({'log_id':log_id,'stream':stream,'missing':len(set(expected)-set(actual)),
                               'unexpected':len(set(actual)-set(expected)),'size_mismatch':sum(actual.get(k)!=v for k,v in expected.items())})
            counts['retained_files']+=len(actual);counts['retained_bytes']+=sum(actual.values())
        for key,p in [('poses',log/'city_SE3_egovehicle.feather'),('argotweak',Path(info['argotweak_path']))]:
            if hashlib.sha256(p.read_bytes()).hexdigest()!=info['metadata_sha256'][key]:
                errors.append({'log_id':log_id,'metadata_changed':key})
        pool='real_test' if info['bucket']=='real_change' else 'constructed_'+info['argotweak_split']
        if info['status']=='protected_review':
            pool='protected_review/'+pool
        dest=dataset/pool/log_id;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            dest.symlink_to(os.path.relpath(log,dest.parent),target_is_directory=True)
        counts[pool+'_logs']+=1
        if log_number % 20==0 or log_number==len(all_logs):
            print(json.dumps({'verified_logs':log_number,'total':len(all_logs),
                              'errors':len(errors),'retained_files':counts['retained_files']}),flush=True)
    if follow_log:
        while not (out/'deletion_summary.json').exists():
            if 'Traceback (most recent call last)' in Path(follow_log).read_text():
                raise RuntimeError('Application stopped before final deletion summary')
            time.sleep(5)
    pending=db.execute('SELECT COUNT(*) FROM files WHERE keep=0 AND deleted=0').fetchone()[0]
    result={'passed':not errors and pending==0,'pending_deletions':pending,'errors':errors,**counts}
    jwrite(out/'verification.json',result);db.close()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if not result['passed']:
        raise ValueError('Post-deletion verification failed; inspect verification.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['plan','apply','verify'])
    p.add_argument('--out',type=Path,default=BASE/'step02_clipped_v1')
    p.add_argument('--radius',type=float,default=40)
    p.add_argument('--padding',type=float,default=5)
    p.add_argument('--follow-apply-log',type=Path,help='Verify each log only after its committed apply progress is published')
    args=p.parse_args()
    if args.radius<=0 or args.padding<0:
        p.error('radius must be positive; padding must be nonnegative')
    {'plan':make_plan,'apply':apply_plan,'verify':verify_plan}[args.mode](args)
