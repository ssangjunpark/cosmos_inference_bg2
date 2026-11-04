"""Deterministic language augmentation independent of worker count/prefetch."""
import hashlib,json
import numpy as np
from .bg2_data import BG2Dataset,FixedStepSampler
class InstructionDataset(BG2Dataset):
    def __init__(self, root, split, instruction_config, mode='train'):
        super().__init__(root,split)
        self.spec=json.loads(instruction_config.read_text());self.mode=mode
    def __getitem__(self, item):
        index,epoch=item if isinstance(item,tuple) else (item,0)
        result=super().__getitem__(index)
        ep=self.eps[int(np.searchsorted(self.ends,index,side='right'))]
        spec=self.spec;target=spec['targets'][str(ep['target'])]
        if self.mode=='canonical': return result
        if self.mode in ('heldout_validation','heldout_test','semantic'):
            car=spec['cars'][ep['car']][0];name=target['names'][0]
            template='Pick up the {car} and place it on the {target}.' if self.mode=='semantic' else spec[self.mode]
        else:
            raw=f"{spec['seed']}:{epoch}:{ep['id']}:{index}".encode()
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(raw).digest()[:8],'little'))
            car=str(rng.choice(spec['cars'][ep['car']]));name=str(rng.choice(target['names']))
            u=rng.random()
            if u < spec['mixture']['vlm_style']:
                template='Pick up the {car} and place it on the {target}.'
            elif u < spec['mixture']['vlm_style']+spec['mixture']['paraphrase']:
                template=str(rng.choice(spec['paraphrases']))
            else:
                template='Pick up the {car} and place it on the {target}.'
                name=f"{target['position']} {target['picture']} picture"
        result['task']=template.format(car=car,target=name)
        return result
class EpochSampler(FixedStepSampler):
    def __iter__(self):
        for micro in range(self.start*self.accum,self.stop*self.accum):
            begin=(micro*self.w+self.r)*self.b;out=[]
            for offset in range(begin,begin+self.b):
                epoch,pos=divmod(offset,self.n)
                if epoch not in self.perms:self.perms={epoch:np.random.default_rng(self.seed+epoch).permutation(self.n)}
                out.append((int(self.perms[epoch][pos]),epoch))
            yield out
