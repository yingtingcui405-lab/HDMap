import unittest
import argparse
import contextlib
import hashlib
import io
import json
import sqlite3
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
import shapely
import clip_dataset as c


class ClipTests(unittest.TestCase):
    def apply_fixture(self, root, mutate_kept=False, interrupt_after_unlink=False):
        raw=root/'raw';log=raw/'TbV_v1.0_shard0'/'example__Summer_2020'
        sensors=log/'sensors/lidar';sensors.mkdir(parents=True)
        kept=sensors/'100.feather';removed=sensors/'200.feather'
        kept.write_bytes(b'keep');removed.write_bytes(b'delete')
        pose=log/'city_SE3_egovehicle.feather';pose.write_bytes(b'pose')
        annotation=root/'annotation.json';annotation.write_text('{}')
        untouched=sensors/'notes.txt';untouched.write_text('preserve')
        out=root/'out';out.mkdir();db=sqlite3.connect(out/'file_ledger.sqlite')
        db.execute('CREATE TABLE files(log_id TEXT,relative_path TEXT,stream TEXT,keep INTEGER,size INTEGER,mtime_ns INTEGER,inode INTEGER,deleted INTEGER DEFAULT 0)')
        db.execute('CREATE TABLE logs(log_id TEXT,data TEXT)')
        db.execute('CREATE TABLE meta(key TEXT,value TEXT)')
        db.execute('INSERT INTO meta VALUES(?,?)',('config',json.dumps({'complete':True})))
        info={'log_dir':str(log),'status':'verified_plan','argotweak_path':str(annotation),
              'metadata_sha256':{'poses':hashlib.sha256(pose.read_bytes()).hexdigest(),
                                 'argotweak':hashlib.sha256(annotation.read_bytes()).hexdigest()}}
        db.execute('INSERT INTO logs VALUES(?,?)',(log.name,json.dumps(info)))
        for file,k in [(kept,1),(removed,0)]:
            st=file.stat();db.execute('INSERT INTO files(log_id,relative_path,stream,keep,size,mtime_ns,inode) VALUES(?,?,?,?,?,?,?)',
                (log.name,str(file.relative_to(log)),'lidar',k,st.st_size,st.st_mtime_ns,st.st_ino))
        db.commit();db.close()
        if mutate_kept:
            kept.write_bytes(b'changed')
        with patch.object(c,'RAW',raw),contextlib.redirect_stdout(io.StringIO()):
            if mutate_kept:
                with self.assertRaises(ValueError):c.apply_plan(argparse.Namespace(out=out))
            else:
                if interrupt_after_unlink:
                    original_unlink=c.os.unlink
                    def interrupting_unlink(*args,**kwargs):
                        original_unlink(*args,**kwargs)
                        raise KeyboardInterrupt('simulate interruption before ledger commit')
                    with patch.object(c.os,'unlink',side_effect=interrupting_unlink):
                        with self.assertRaises(KeyboardInterrupt):c.apply_plan(argparse.Namespace(out=out))
                c.apply_plan(argparse.Namespace(out=out))
        self.assertTrue(kept.exists());self.assertTrue(pose.exists());self.assertTrue(annotation.exists());self.assertTrue(untouched.exists())
        self.assertEqual(removed.exists(),mutate_kept)
        if not mutate_kept:
            summary=c.jread(out/'deletion_summary.json')
            self.assertEqual(summary,{'deleted_files':1,'deleted_bytes':6,'pending_deletions':0})

    def test_apply_deletes_only_manifest_sensor_files(self):
        with tempfile.TemporaryDirectory() as tmp:self.apply_fixture(Path(tmp).resolve())

    def test_changed_retained_file_blocks_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:self.apply_fixture(Path(tmp).resolve(),mutate_kept=True)

    def test_resume_after_unlink_before_ledger_commit(self):
        with tempfile.TemporaryDirectory() as tmp:self.apply_fixture(Path(tmp).resolve(),interrupt_after_unlink=True)

    def test_verification_waits_for_commit_and_detects_unexpected_sensor_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();log=root/'raw'/'example';out=root/'out';out.mkdir()
            db=sqlite3.connect(out/'file_ledger.sqlite')
            db.execute('CREATE TABLE files(log_id TEXT,relative_path TEXT,stream TEXT,keep INTEGER,size INTEGER,deleted INTEGER)')
            db.execute('CREATE TABLE logs(log_id TEXT,data TEXT)')
            for stream,(relative,suffix) in c.STREAMS.items():
                p=log/relative/f'100{suffix}';p.parent.mkdir(parents=True);p.write_bytes(b'keep')
                db.execute('INSERT INTO files VALUES(?,?,?,?,?,?)',(log.name,str(p.relative_to(log)),stream,1,4,0))
            pose=log/'city_SE3_egovehicle.feather';pose.write_bytes(b'pose')
            annotation=root/'annotation.json';annotation.write_text('{}')
            info={'log_dir':str(log),'status':'verified_plan','argotweak_path':str(annotation),
                  'bucket':'constructed_pool','argotweak_split':'train',
                  'metadata_sha256':{'poses':hashlib.sha256(pose.read_bytes()).hexdigest(),
                                     'argotweak':hashlib.sha256(annotation.read_bytes()).hexdigest()}}
            db.execute('INSERT INTO logs VALUES(?,?)',(log.name,json.dumps(info)));db.commit();db.close()
            progress=out/'apply.log';progress.write_text('{"applied_logs":0}\n')
            def publish_committed_progress(_):
                progress.write_text('{"applied_logs":1}\n')
                c.jwrite(out/'deletion_summary.json',{'pending_deletions':0})
            with contextlib.redirect_stdout(io.StringIO()):
                with patch.object(c.time,'sleep',side_effect=publish_committed_progress) as wait:
                    c.verify_plan(argparse.Namespace(out=out,follow_apply_log=progress))
                    wait.assert_called_once()
                self.assertTrue(c.jread(out/'verification.json')['passed'])
                self.assertTrue((out/'dataset/constructed_train/example').is_symlink())
                (log/'sensors/lidar/200.feather').write_bytes(b'unexpected')
                with self.assertRaises(ValueError):c.verify_plan(argparse.Namespace(out=out))
            result=c.jread(out/'verification.json')
            self.assertFalse(result['passed'])
            self.assertEqual(result['errors'][0]['unexpected'],1)

    def test_padding_merge_and_nanosecond_precision(self):
        base=315968867659956000
        ts=base+np.arange(101,dtype=np.int64)*100_000_000
        distances=np.ones(len(ts))*100
        distances[30:40]=0
        distances[65:70]=0
        windows=c.select_intervals(ts,distances,40,1_000_000_000,2_000_000_000)
        self.assertEqual(windows,[[int(ts[20]),int(ts[79])]])
        self.assertEqual(int(c.in_intervals(ts,windows).sum()),60)

    def test_out_of_range_is_not_treated_as_normal(self):
        self.assertEqual(c.select_intervals(np.array([0,100]),np.array([60.,70.]),40,0,0),[])

    def test_lane_polygon_includes_inside_and_both_map_versions(self):
        def lane(x):
            return {'left_lane_boundary':[{'x':x,'y':0},{'x':x+10,'y':0}],
                    'right_lane_boundary':[{'x':x,'y':4},{'x':x+10,'y':4}]}
        annotation={'laneSegments':{'a':{'changes':[3], 'old':lane(0),'new':lane(100)}},
                    'pedCrossings':{},'drivableAreas':{}}
        g,n=c.change_geometry(annotation,[])
        self.assertEqual(n,1)
        np.testing.assert_allclose(shapely.distance(shapely.points([[5,2],[105,2]]),g),[0,0])

    def test_unchanged_null_new_does_not_create_change_region(self):
        with self.assertRaises(ValueError):
            c.change_geometry({'laneSegments':{'a':{'changes':[0],'old':{},'new':None}},
                               'pedCrossings':{},'drivableAreas':{}},[])

    def test_only_timestamped_sensor_paths_can_be_deleted(self):
        log=c.RAW/'TbV_v1.0_shard0'/'example__Summer_2020'
        for rel in ['city_SE3_egovehicle.feather','sensors/lidar/../../map/1.feather',
                    'sensors/lidar/anything.feather','sensors/cameras/ring_front_center/1.jpg']:
            with self.assertRaises(ValueError):
                c.safe_sensor_path(log,rel,'lidar')


if __name__=='__main__':
    unittest.main()
