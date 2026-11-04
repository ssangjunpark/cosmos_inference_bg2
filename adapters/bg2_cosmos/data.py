"""BG2 room samples with exact source pixels and measured future joint labels.

The lightweight core has no Cosmos import so the data contract is CPU-testable.
Cosmos-specific tokenization and batching are imported only in the loader.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .instruction_data import InstructionDataset
from .vision_contract import compose_views,build_policy_prompt,HEIGHT,WIDTH


def quantile_affine(value, stats):
    lo=np.asarray(stats['q01'],dtype=np.float64)
    hi=np.asarray(stats['q99'],dtype=np.float64)
    scale=np.maximum((hi-lo)/2,1e-6)
    return (np.asarray(value,dtype=np.float64)-(hi+lo)/2)/scale


class RoomSamples:
    def __init__(self, root, instructions, split='train', mode='train'):
        self.root=Path(root)
        self.source=InstructionDataset(self.root,split,Path(instructions),mode)
        normalizer=json.loads((self.root/'joint_train_stats.json').read_text())
        digest=hashlib.sha256((self.root/'manifest.json').read_bytes()).hexdigest()
        if normalizer['manifest_sha256'] != digest or normalizer['statistics_split']!='train only':
            raise ValueError('Joint normalization must belong to this training split')
        self.stats=normalizer['joint_position']

    def __len__(self):
        return len(self.source)

    def __getitem__(self,item):
        src=self.source[item]
        state=src['observation.state'].numpy()
        absolute=src['action'].numpy()
        valid=int((~src['action_is_pad']).sum())
        if not 1 <= valid <= 16:
            raise ValueError('Empty or invalid action horizon')
        if state.shape!=(8,) or absolute.shape!=(16,8):
            raise ValueError('Expected eight actual measured joints')
        if not np.all(src['target_time_ns'][:valid]>src['observation_time_ns']):
            raise ValueError('Targets must be strictly later than the observation')
        normalized=quantile_affine(absolute,self.stats)[:valid]
        observed=quantile_affine(state,self.stats)
        video=compose_views(src['observation.images.head'],src['observation.images.wrist_left'],src['observation.images.wrist'])
        clean=torch.from_numpy(observed.copy()).float().unsqueeze(0)
        target=torch.from_numpy(normalized.copy()).float()
        action=torch.cat([clean,target],dim=0)
        if not torch.isfinite(action).all():
            raise ValueError('Non-finite model-space action')
        return {'video':video,'action':action,'action_raw':torch.from_numpy(np.concatenate([state[None],absolute[:valid]],axis=0)).float(),
                'instruction':src['task'],'valid_future_steps':valid,'current_state':torch.from_numpy(state.copy()),'absolute_targets':torch.from_numpy(absolute[:valid].copy())}


class GlobalSampler(Sampler):
    """Exact global sample order at any world size; resume by consumed microbatches."""
    def __init__(self,n,batch,rank,world,start_micro=0,stop_micro=1,seed=20261030):
        self.n=n;self.batch=batch;self.rank=rank;self.world=world
        self.start=start_micro;self.stop=stop_micro;self.seed=seed
    def __len__(self):return self.stop-self.start
    def __iter__(self):
        cached_epoch=None;perm=None
        for micro in range(self.start,self.stop):
            first=(micro*self.world+self.rank)*self.batch;items=[]
            for offset in range(first,first+self.batch):
                epoch,pos=divmod(offset,self.n)
                if cached_epoch!=epoch:
                    perm=np.random.default_rng(self.seed+epoch).permutation(self.n);cached_epoch=epoch
                items.append((int(perm[pos]),epoch))
            yield items


class CosmosRoomDataset(RoomSamples):
    def __init__(self,*args,tokenizer_config=None,**kwargs):
        super().__init__(*args,**kwargs)
        from cosmos_framework.utils.lazy_config import instantiate
        self.tokenizer=instantiate(tokenizer_config)
        self.token_cache={}
    def __getitem__(self,item):
        from cosmos_framework.data.generator.sequence_packing import SequencePlan
        raw=super().__getitem__(item)
        instruction=raw['instruction']
        if instruction not in self.token_cache:
            prompt=build_policy_prompt(instruction)
            self.token_cache[instruction]=(prompt,self.tokenizer.tokenize_text(prompt,is_video=False,use_system_prompt=False))
        prompt,tokens=self.token_cache[instruction]
        if not tokens:raise ValueError('Empty instruction tokens')
        # Native Cosmos action tensors are padded to 64 channels. Only eight measured joints are valid.
        action=torch.nn.functional.pad(raw['action'],(0,56))
        return {'__key__':str(item),'mode':'policy','video':raw['video'],'action':action,
                'action_raw':raw['action_raw'],'raw_action_dim':torch.tensor(8),
                'domain_id':torch.tensor(33),'ai_caption':prompt,
                'text_token_ids':torch.tensor(tokens,dtype=torch.long),
                'conditioning_fps':torch.tensor(20,dtype=torch.long),'image_size':torch.tensor([float(HEIGHT),float(WIDTH),float(HEIGHT),float(WIDTH)]),
                'sequence_plan':SequencePlan(has_text=True,has_vision=True,has_action=True,
                    condition_frame_indexes_vision=[0],condition_frame_indexes_action=[0],action_start_frame_offset=0)}


class RoomLoader:
    """Rank-local deterministic batches, checkpoint resumes without prefetch drift."""
    def __init__(self,root,instructions,tokenizer_config,split='train',mode='train',microbatch=2,global_batch=128,steps=20000,workers=2,max_samples=None):
        import torch.distributed as dist
        self.rank=dist.get_rank() if dist.is_initialized() else 0
        self.world=dist.get_world_size() if dist.is_initialized() else 1
        self.batch=microbatch;self.global_batch=global_batch;self.steps=steps;self.workers=workers
        if global_batch%(self.world*microbatch):raise ValueError('Global batch must divide across ranks/accumulation')
        self.accum=global_batch//(self.world*microbatch);self.start=0
        self.dataset=CosmosRoomDataset(root,instructions,split,mode,tokenizer_config=tokenizer_config)
        self.split=split
        self.max_samples=max_samples
    def set_start_iteration(self,raw_batch_position):self.start=int(raw_batch_position)
    def __iter__(self):
        from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
        if self.split=='train':
            sampler=GlobalSampler(len(self.dataset),self.batch,self.rank,self.world,self.start,self.steps*self.accum)
        else:
            # Equal counts per rank to avoid collective hangs. All configured evaluation budgets truncate before a repeated tail.
            count=(len(self.dataset)+self.batch*self.world-1)//(self.batch*self.world)
            if self.max_samples is not None:
                if self.max_samples%(self.batch*self.world):raise ValueError("Evaluation subset must divide across ranks")
                count=min(count,self.max_samples//(self.batch*self.world))
            sampler=GlobalSampler(len(self.dataset),self.batch,self.rank,self.world,0,count)
        dl=DataLoader(self.dataset,batch_sampler=sampler,num_workers=self.workers,pin_memory=True,collate_fn=custom_collate_fn,
                      **({'prefetch_factor':2,'persistent_workers':True} if self.workers else {}))
        for batch in dl:
            # PackingDataLoader emits the same nested one-item lists for per-sample media metadata.
            batch['dataset_name']=['bg2_room']*len(batch['video'])
            for key in ['text_token_ids','video','action','action_raw']:
                batch[key]=[[v] for v in batch[key]]
            batch['image_size']=[v.unsqueeze(0) for v in batch['image_size']]
            batch['conditioning_fps']=[v.unsqueeze(0) for v in batch['conditioning_fps']]
            yield batch
