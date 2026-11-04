import tempfile
import unittest
from pathlib import Path
import reorganize_scenes as m

class SceneLayoutTests(unittest.TestCase):
    def test_atomic_move_preserves_data_and_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();source=root/'shard'/'scene';source.mkdir(parents=True)
            (source/'sensors').mkdir();(source/'sensors/frame').write_bytes(b'original sensor bytes')
            expected={str(p.relative_to(source)):m.fingerprint(p) for p in [source,source/'sensors',source/'sensors/frame']}
            dest=root/'scene'
            self.assertEqual(m.move_log(source,dest,expected),'moved')
            self.assertEqual((dest/'sensors/frame').read_bytes(),b'original sensor bytes')
            (dest/'scene_info.json').write_text('{}')
            self.assertEqual(m.move_log(source,dest,expected),'already_moved')
            self.assertFalse(source.exists())

    def test_existing_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';dest=root/'dest';source.mkdir();dest.mkdir()
            (dest/'valuable').write_text('keep')
            with self.assertRaises(ValueError):m.move_log(source,dest,{'.':m.fingerprint(source)})
            self.assertTrue(source.is_dir());self.assertEqual((dest/'valuable').read_text(),'keep')

    def test_path_refresh_handles_nested_maps_and_preserves_change_fields(self):
        old=str(m.RAW/'TbV_v1.0_shard0'/'id');new=str(m.RAW/'id')
        obj={'log_dir':old,'map':[old+'/map/a.json'],'argotweak_path':'/ann/id.json',
             'old':{'x':1},'new':None,'other':old+'-different'}
        result=m.rewrite(obj,{old:new},{'/ann/id.json':new+'/argotweak_annotation.json'})
        self.assertEqual(result['map'],[new+'/map/a.json'])
        self.assertEqual(result['argotweak_path'],new+'/argotweak_annotation.json')
        self.assertEqual(result['old'],{'x':1});self.assertIsNone(result['new'])
        self.assertEqual(result['other'],old+'-different')

    def test_foreign_symlink_is_not_repointed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();link=root/'link';link.symlink_to(root/'foreign',target_is_directory=True)
            with self.assertRaises(ValueError):m.replace_link(link,root/'old',root/'new')
            self.assertEqual(link.readlink(),root/'foreign')

if __name__=='__main__':unittest.main()
