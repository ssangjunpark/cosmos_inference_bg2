"""Recording-only paths shared by the model service and gateway."""
import json
from pathlib import Path
import re

ARTIFACT_API = 'cosmos_run_recordings_v1'
VIDEOS = ('imagined_future.mp4', 'imagined_head.mp4', 'imagined_wrist_left.mp4',
          'imagined_wrist_right.mp4', 'rollout_full.mp4')
PUBLIC_FILES = (*VIDEOS, 'decoded_rollout.npy', 'rollout.json', 'model_output.json', 'imagined_last_frame.png',
                'context_reconstruction.png')


def run_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise ValueError('Run name must use letters, numbers, underscore, dot or hyphen (up to 128 characters)')
    return value


def artifact_path(run_id, epoch, request_id):
    from .rpc import identifier
    return '/artifacts/' + '/'.join((run_name(run_id), identifier(epoch), identifier(request_id)))


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)
