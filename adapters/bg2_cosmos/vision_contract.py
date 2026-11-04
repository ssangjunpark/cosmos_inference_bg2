"""Shared train/eval/deployment compositor and native Cosmos prompt formatter."""
import json
import torch
HEIGHT,WIDTH=640,512
LAYOUT='The top row shows the head-mounted camera. The bottom row shows the left wrist camera on the left and the right wrist camera on the right.'
def compose_views(head,left,right):
    if head.shape!=(3,384,512) or left.shape!=(3,256,256) or right.shape!=(3,256,256):
        raise ValueError('Expected head CHW=3x384x512 and wrists=3x256x256')
    if any(v.dtype!=torch.uint8 for v in (head,left,right)):raise TypeError('Require RGB uint8 cropped views')
    return torch.cat([head,torch.cat([left,right],dim=-1)],dim=-2).unsqueeze(1).contiguous()
def build_policy_prompt(instruction):
    from cosmos_framework.data.generator.action.utils.json_formatter import ActionPromptJsonFormatter
    # Known fixed 0.8s prediction horizon only. No future state, idle count,
    # subtask-end time, success label, or held-out annotation enters the prompt.
    payload={'mode':'policy','ai_caption':instruction,'viewpoint':'concat_view','additional_view_description':LAYOUT,
       'conditioning_fps':20,'image_size':[HEIGHT,WIDTH,HEIGHT,WIDTH],
       'video':torch.empty(3,1,1,1,dtype=torch.uint8),'video_num_frames':16}
    structured=ActionPromptJsonFormatter(float_seconds=True)(payload)['ai_caption']
    return json.dumps(structured)
