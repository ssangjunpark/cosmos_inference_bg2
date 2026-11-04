"""Identical train/eval objective, with explicit native batch-coverage checks."""
import torch
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

class BG2PolicyModel(OmniMoTModel):
    def training_step(self, data_batch, iteration):
        expected_samples=len(data_batch['video'])
        expected_action_tokens=sum(v[0].shape[0] for v in data_batch['action'])
        output,loss=super().training_step(data_batch,iteration)
        if output['batch_size']!=expected_samples:
            raise RuntimeError(f"Native sample coverage mismatch: {output['batch_size']} != {expected_samples}")
        if output['action_token_length']!=expected_action_tokens:
            raise RuntimeError(f"Native action coverage mismatch: {output['action_token_length']} != {expected_action_tokens}")
        return output,loss

    @torch.no_grad()
    def validation_step(self, data_batch, iteration):
        return self.training_step(data_batch, iteration)
