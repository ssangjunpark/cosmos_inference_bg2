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
        if output['is_image_batch']:
            raise RuntimeError('WAM was incorrectly encoded as an image batch')
        for key in ['flow_matching_loss_vision','flow_matching_loss_action']:
            if not torch.isfinite(output[key]).all() or output[key].detach().item()<=0:
                raise RuntimeError('Missing or invalid WAM objective: '+key)
        for key in ['condition_mask_vision','condition_mask_action']:
            masks=output[key]
            if masks is None or any(not bool(mask.any()) or bool(mask.all()) for mask in masks):
                raise RuntimeError('Invalid WAM conditioning/target mask: '+key)
        return output,loss

    @torch.no_grad()
    def validation_step(self, data_batch, iteration):
        return self.training_step(data_batch, iteration)
