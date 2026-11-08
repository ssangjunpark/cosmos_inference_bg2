"""CPU tests for offline WAM contracts, conditioning, rendering and isolation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from wam_checkpoint import FRAMEWORK_COMMIT,validate_checkpoint,validate_experiment
from bg2_wam import prepare_tensors,decode_actions,decoded_rgb,save_rollout,exported_formatter,restore_wam_config
from bg2_inference import JOINT_NAMES


def observation():
    return dict(head=np.full((384,512,3),20,np.uint8),wrist_left=np.full((256,256,3),60,np.uint8),
                wrist=np.full((256,256,3),100,np.uint8),joint_position=np.arange(8,dtype=np.float32)/10,
                instruction='Pick up the police car.',source_time_ns=np.asarray(123),joint_names=np.asarray(JOINT_NAMES))


def contract():
    return {'framework_commit':FRAMEWORK_COMMIT,'data_manifest_sha256':'dataset',
            'vision':{'frame_interval_ns':50_000_000,'first_target_offset_frames':1},
            'normalization':{'joint_names':JOINT_NAMES,'joint_position':{'q01':[-1.]*8,'q99':[1.]*8}}}


def experiment():
    return {'training_mode':'wam','framework_commit':FRAMEWORK_COMMIT,'data':{'sha256':{'manifest':'dataset'}},
            'action':{'horizon':16,'fps':20,'dim':8,'domain_id':33,'num_embodiment_domains':34,'names':JOINT_NAMES,'first_offset':1},
            'clip_audit':{'frames':17,'future_actions':16,'passed':True,'frame_interval_ns':50_000_000}}


class WAMCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.p=Path(self.temp.name)
        (self.p/'model').mkdir()
        for name,value in [('.metadata',b'metadata'),('__0_0.distcp',b'weights'),('__1_0.distcp',b'')]:
            (self.p/'model'/name).write_bytes(value)
        self.write('READY.json',{'step':20000,'model_tier':'nano','variant':'nano_wam','code_variant':'cosmos_wam','format':'pytorch_distributed_checkpoint'})
        self.write('TRANSFER_VERIFIED.json',{'state':'complete','verified':True,'sha256_verified':True,'tier':'nano','step':20000})
        self.write('input_contract.json',contract())
        self.write('VERIFIED.json',{'step':20000,'format':'pytorch_distributed_checkpoint','files':[
            {'path':'model/'+p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (self.p/'model').iterdir()]})
        self.write('config.yaml',{'model':{'_target_':'bg2_cosmos.model.BG2PolicyModel','config':{
            'tokenizer':{'encode_exact_durations':[17]},'num_embodiment_domains':34,'max_action_dim':64,'vision_gen':True,'action_gen':True,
            'rectified_flow_training_config':{'image_loss_scale':.1}}}})
        (self.p/'vision_contract.py').write_text('VALUE=17')

    def write(self,name,value): (self.p/name).write_text(json.dumps(value))

    def test_accepts_delivered_layout_including_empty_distributed_shards_without_writes(self):
        before={str(p.relative_to(self.p)):(p.stat().st_mtime_ns,p.read_bytes()) for p in self.p.rglob('*') if p.is_file()}
        self.assertEqual(validate_checkpoint(self.p,hashes=True)['tier'],'nano')
        after={str(p.relative_to(self.p)):(p.stat().st_mtime_ns,p.read_bytes()) for p in self.p.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_rejects_incomplete_transfer(self):
        self.write('TRANSFER_VERIFIED.json',{'state':'copying'})
        with self.assertRaisesRegex(ValueError,'transfer'):validate_checkpoint(self.p)

    def test_rejects_changed_shard_even_same_size_when_hash_requested(self):
        (self.p/'model/__0_0.distcp').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):validate_checkpoint(self.p,hashes=True)

    def test_rejects_policy_only_vae_duration(self):
        d=json.loads((self.p/'config.yaml').read_text());d['model']['config']['tokenizer']['encode_exact_durations']=[1]
        self.write('config.yaml',d)
        with self.assertRaisesRegex(ValueError,'configuration'):validate_checkpoint(self.p)

    def test_rejects_path_escape_and_missing_shard(self):
        d=json.loads((self.p/'VERIFIED.json').read_text());d['files'][0]['path']='../outside'
        self.write('VERIFIED.json',d)
        with self.assertRaisesRegex(ValueError,'Unsafe'):validate_checkpoint(self.p)

    def test_exported_formatter_does_not_create_bytecode_beside_checkpoint(self):
        old=sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode=False
            self.assertEqual(exported_formatter(self.p).VALUE,17)
        finally:sys.dont_write_bytecode=old
        self.assertFalse((self.p/'__pycache__').exists())

    def test_experiment_must_match_domain_mode_and_dataset(self):
        self.write('experiment.json',experiment());validate_experiment(self.p/'experiment.json',contract())
        for key,value in [('domain_id',32),('fps',15),('horizon',32)]:
            d=experiment();d['action'][key]=value;self.write('experiment.json',d)
            with self.assertRaises(ValueError):validate_experiment(self.p/'experiment.json',contract())

    def test_experiment_requires_passed_clip_audit_and_exact_future_offsets(self):
        for field,value in [('passed',False),('frame_interval_ns',100_000_000)]:
            d=experiment();d['clip_audit'][field]=value;self.write('experiment.json',d)
            with self.assertRaisesRegex(ValueError,'audit'):validate_experiment(self.p/'experiment.json',contract())
        d=experiment();d['action']['first_offset']=0;self.write('experiment.json',d)
        with self.assertRaisesRegex(ValueError,'timing'):validate_experiment(self.p/'experiment.json',contract())


class WAMConfigTests(unittest.TestCase):
    def test_delivered_configs_keep_every_value_except_exact_nano_factory(self):
        import yaml
        root=Path('/mnt/sata1/RobotEscapeCkpt/cosmos3_wam_room20k_20261105')
        if not (root/'edge/020000/config.yaml').exists():self.skipTest('Local WAM checkpoint unavailable')
        for tier in ('edge','nano'):
            model=yaml.safe_load((root/tier/'020000/config.yaml').read_text())['model']
            overrides=[];runtime=restore_wam_config(model,overrides);expected=copy.deepcopy(model)
            if tier=='nano':
                base=expected['config']['vlm_config']['model_instance']['config']['base_config']
                base['_target_']='cosmos_framework.model.generator.mot.unified_mot.Qwen3VLMoTConfig.from_json_file'
            self.assertEqual(runtime,expected);self.assertEqual(len(overrides),int(tier=='nano'))

    def test_resolves_exported_abstract_nano_factory_without_mutating_saved_config(self):
        prefix='cosmos_framework.model.generator.mot.unified_mot.'
        model={'config':{'vlm_config':{'model_instance':{'_target_':prefix+'Qwen3VLTextForCausalLM','config':{
            'base_config':{'_target_':prefix+'_MoTConfigBase.from_json_file',
                'json_file':'cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json'},
            'tie_word_embeddings':True}}}}}
        before=copy.deepcopy(model);overrides=[];runtime=restore_wam_config(model,overrides)
        self.assertEqual(model,before);self.assertEqual(len(overrides),1)
        base=runtime['config']['vlm_config']['model_instance']['config']['base_config']
        self.assertEqual(base['_target_'],prefix+'Qwen3VLMoTConfig.from_json_file')
        unchanged=[];self.assertEqual(restore_wam_config(runtime,unchanged),runtime);self.assertEqual(unchanged,[])
        model['config']['vlm_config']['model_instance']['_target_']=prefix+'UnknownModel'
        with self.assertRaisesRegex(ValueError,'unknown'):restore_wam_config(model,[])


@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch required')
class WAMInputTests(unittest.TestCase):
    def test_only_present_images_and_current_state_fill_the_input(self):
        import torch
        class Formatter:
            def compose_views(self,head,left,right):
                return torch.cat([head,torch.cat([left,right],dim=-1)],dim=-2)
        batch=prepare_tensors(observation(),contract(),experiment(),Formatter())
        self.assertEqual(tuple(batch['video'].shape),(3,17,640,512))
        self.assertEqual(int(torch.count_nonzero(batch['video'][:,1:])),0)
        self.assertTrue(torch.all(batch['video'][:,0,:384]==20))
        self.assertTrue(torch.all(batch['video'][:,0,384:,:256]==60))
        self.assertTrue(torch.all(batch['video'][:,0,384:,256:]==100))
        self.assertEqual(tuple(batch['action'].shape),(17,64))
        np.testing.assert_array_equal(batch['action'][0,:8].numpy(),observation()['joint_position'])
        self.assertEqual(int(torch.count_nonzero(batch['action'][1:])),0)
        self.assertEqual(int(torch.count_nonzero(batch['action'][:,8:])),0)
        self.assertEqual(int(batch['domain_id']),33)

    def test_decode_retains_raw_negative_gripper_and_absolute_arm_values(self):
        normalized=np.zeros((17,8),dtype=np.float32);normalized[0]=observation()['joint_position']
        normalized[1:,0]=2.;normalized[1:,7]=-.1
        actions=decode_actions(normalized,observation(),contract())
        self.assertEqual(actions.shape,(16,8));self.assertTrue(np.all(actions[:,0]==2.))
        self.assertTrue(np.all(actions[:,7]==np.float32(-.1)))

    def test_decode_rejects_conditioning_change_or_wrong_horizon(self):
        for value in [np.zeros((16,8)),np.zeros((17,8)),np.full((17,8),np.nan)]:
            with self.assertRaises(ValueError):decode_actions(value,observation(),contract())

    def test_decoded_video_shape_color_range_and_order(self):
        import torch
        decoded=torch.zeros((1,3,17,640,512));decoded[:,0]=1;decoded[:,2]=-1
        rgb=decoded_rgb(decoded)
        self.assertEqual(rgb.shape,(17,640,512,3));self.assertEqual(rgb.dtype,np.uint8)
        np.testing.assert_array_equal(rgb[0,0,0],[255,127,0])
        with self.assertRaises(ValueError):decoded_rgb(decoded[:,:,:1])


@unittest.skipUnless(importlib.util.find_spec('av'),'PyAV required')
class WAMRenderTests(unittest.TestCase):
    def test_mp4_roundtrip_has_real_future_frames_and_correct_views(self):
        import av
        frames=np.zeros((17,640,512,3),np.uint8)
        frames[:, :384, :, 0]=255;frames[:,384:,:256,1]=255;frames[:,384:,256:,2]=255
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);metadata=save_rollout(path,frames)
            for name,count,wh,channel in [('rollout_full',17,(512,640),0),('imagined_future',16,(512,640),0),
                    ('imagined_head',16,(512,384),0),('imagined_wrist_left',16,(256,256),1),('imagined_wrist_right',16,(256,256),2)]:
                with av.open(str(path/(name+'.mp4'))) as container:
                    stream=container.streams.video[0];decoded=list(container.decode(stream))
                    self.assertEqual(len(decoded),count);self.assertEqual((stream.width,stream.height),wh)
                    self.assertAlmostEqual(float(stream.average_rate),20.)
                    pixel=decoded[0].to_ndarray(format='rgb24')[20,20]
                    self.assertGreater(int(pixel[channel]),240)
            self.assertEqual(metadata['frame_offsets_ns'][-1],800_000_000)
            self.assertEqual(metadata['future_only_seconds'],.8)
            with self.assertRaises(FileExistsError):save_rollout(path,frames)


class WAMLauncherTests(unittest.TestCase):
    def test_gpu_preflight_refuses_other_workload(self):
        import cosmos3
        with mock.patch.object(cosmos3.subprocess,'check_output',side_effect=['GPU-test, 100, 0','GPU-test\n']):
            with self.assertRaisesRegex(ValueError,'occupied'):cosmos3.idle_gpu('2')
        with mock.patch.object(cosmos3.subprocess,'check_output',side_effect=['GPU-test, 0, 0','']):
            self.assertEqual(cosmos3.idle_gpu('2')['uuid'],'GPU-test')

    def test_actual_dry_run_never_starts_gpu_or_opens_network(self):
        root=Path('/mnt/sata1/RobotEscapeCkpt/cosmos3_wam_room20k_20261105')
        if not (root/'edge/020000/READY.json').exists():self.skipTest('Local WAM checkpoint unavailable')
        import cosmos3
        args=['python3','scripts/cosmos3.py','wam','--variant','edge','--gpu','2','--observation','inputs/recorded_observation.npz',
              '--tokenizer','assets/nvidia--Cosmos3-Edge-344d602b128d1bbdacb43b08d0a3626f46343e29',
              '--vae','assets/Wan-AI--Wan2.2-TI2V-5B-921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth','--dry-run']
        r=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,check=True)
        self.assertIn('--network=none',r.stdout);self.assertIn('--read-only',r.stdout)
        self.assertIn('target=/checkpoint,readonly',r.stdout);self.assertIn('/app/wam_infer.py',r.stdout)
        self.assertIn('--guidance-interval 960.0 1001.0',r.stdout)
        self.assertNotIn('18008',r.stdout);self.assertNotIn('--privileged',r.stdout)


if __name__=='__main__':unittest.main()
