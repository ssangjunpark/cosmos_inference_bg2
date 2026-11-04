"""Measured present/future right joints; exact reviewed images at present time."""
from pathlib import Path
from collections import OrderedDict
import json
import numpy as np
import torch
from torch.utils.data import Dataset,Sampler

class BG2Dataset(Dataset):
 def __init__(self,root,split,horizon=16):
  self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
  assert self.manifest['ready'] and self.manifest['schema']==3 and self.manifest['action_dim']==8
  assert self.manifest['contract']['first_target_offset_frames']==1
  self.eps=[e for e in self.manifest['episodes'] if e['split']==split]
  self.ends=np.cumsum([e['frames'] for e in self.eps]);self.horizon=horizon;self.cache=OrderedDict()
 def __len__(self):return int(self.ends[-1])
 def __getitem__(self,index):
  ep=int(np.searchsorted(self.ends,index,side='right'));i=int(index-(0 if ep==0 else self.ends[ep-1]));e=self.eps[ep]
  if ep not in self.cache:
   with np.load(self.root/e['samples']) as z:s={k:z[k] for k in ['joint_positions','source_time_ns']}
   assert len(s['joint_positions'])==e['frames']+1
   self.cache[ep]=(s,{k:np.load(self.root/v,mmap_mode='r') for k,v in e['videos'].items()})
   if len(self.cache)>40:self.cache.popitem(last=False)
  self.cache.move_to_end(ep);s,v=self.cache[ep]
  idx=np.arange(i+1,i+1+self.horizon);valid=idx<e['recorded_frames'];idx=idx.clip(max=e['recorded_frames']-1)
  return {'observation.state':torch.from_numpy(s['joint_positions'][i].copy()).float(),
    'action':torch.from_numpy(s['joint_positions'][idx].copy()).float(),'action_is_pad':torch.from_numpy(~valid),
    'observation_time_ns':int(s['source_time_ns'][i]),'target_time_ns':s['source_time_ns'][idx].copy(),
    'task':e['instruction_en'],**{f'observation.images.{k}':torch.from_numpy(np.array(a[i],copy=True)).permute(2,0,1) for k,a in v.items()}}

class FixedStepSampler(Sampler):
 def __init__(self,n,microbatch,rank,world,start,stop,accum,seed):self.n=n;self.b=microbatch;self.r=rank;self.w=world;self.start=start;self.stop=stop;self.accum=accum;self.seed=seed;self.perms={}
 def __len__(self):return (self.stop-self.start)*self.accum
 def __iter__(self):
  for micro in range(self.start*self.accum,self.stop*self.accum):
   begin=(micro*self.w+self.r)*self.b;out=[]
   for offset in range(begin,begin+self.b):
    epoch,pos=divmod(offset,self.n)
    if epoch not in self.perms:self.perms={epoch:np.random.default_rng(self.seed+epoch).permutation(self.n)}
    out.append(int(self.perms[epoch][pos]))
   yield out
