#!/usr/bin/env python3
"""Prepare WAM metadata and a gateway profile from a verified sidecar handoff.

Copies small config/source files only, into this checkout. Model shards remain in
their original directory and are selected with cosmos3.py's --model-dir option.
No Docker, model, network connection, or robot process is started.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from bg2_bridge.core import validate_profile
from bg2_inference import JOINT_NAMES, canonical_hash, verify_adapter
from wam_checkpoint import FRAMEWORK_COMMIT, validate_checkpoint, validate_experiment

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def workspace_path(value):
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def prepare(config_path):
    config = read_json(config_path)
    if config['schema'] != 1 or config['model_tier'] not in ('edge', 'nano'):
        raise ValueError('Unsupported deployment config')
    version, step, tier = config['variant'], config['step'], config['model_tier']
    source_root = Path(config['source_root']).resolve(strict=True)
    source = (source_root / config['source_checkpoint']).resolve(strict=True)
    source.relative_to(source_root)
    destination = workspace_path(config['prepared_root'])
    profile_path = workspace_path(config['gateway_profile'])
    if any(path == source or source in path.parents or path in source.parents
           for path in (destination, profile_path)):
        raise ValueError('Generated files must be separate from the source checkpoint')

    sidecars = {r['path']: r for r in read_json(source_root / 'SIDECAR_MANIFEST.json')['files']}

    def supplied(name):
        path = source / name
        relative = path.relative_to(source_root).as_posix()
        record = sidecars[relative]
        data = path.read_bytes()
        if len(data) != record['bytes'] or digest(data) != record['sha256']:
            raise ValueError(f'Unverified sidecar: {relative}')
        return data

    contract = json.loads(supplied('input_contract.json'))
    inference = json.loads(supplied('inference_contract.json'))
    stats = json.loads(supplied('joint_train_stats.json'))
    camera = json.loads(supplied('camera_profile.json'))
    cpu = json.loads(supplied('CPU_CONTRACT_VERIFIED.json'))
    training_bytes = supplied('config.yaml')
    training = yaml.safe_load(training_bytes)
    cfg = training['model']['config']
    expected = (version, step, FRAMEWORK_COMMIT, 16, 20, 8, 64, 33)
    actual = tuple(inference[k] for k in ('variant', 'step', 'framework_commit', 'horizon', 'fps',
                                         'raw_action_dim', 'native_action_dim', 'domain_id'))
    audit = cpu['variants'][version]
    if (actual != expected or stats != contract['normalization'] or stats['joint_names'] != JOINT_NAMES or
            inference['base_model'] != 'nvidia/Cosmos3-' + tier.capitalize() or
            cfg['num_embodiment_domains'] != 34 or cfg['tokenizer']['encode_exact_durations'] != [17] or
            not cpu['passed'] or not audit['native_prompt_exact_match'] or
            not audit['raw_camera_compositor_matches_training_cache'] or
            audit['action_padding_shape'] != [17, 64] or audit['frame_shape'] != [640, 512, 3]):
        raise ValueError('Incompatible WAM sidecar contract')

    transfer = read_json(source_root / 'TRANSFER_VERIFIED.json')
    transfer_manifest = read_json(source_root / 'TRANSFER_MANIFEST.json')
    if transfer['state'] != 'verified' or transfer['files'] != transfer_manifest['files']:
        raise ValueError('Transfer receipt differs from its original manifest')
    source_prefix = source.relative_to(source_root).as_posix() + '/'
    files = []
    for record in transfer['files']:
        if not record['path'].startswith(source_prefix + 'model/'):
            continue
        relative = Path(record['path'][len(source_prefix):])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe model manifest path')
        path = (source / relative).resolve(strict=True)
        path.relative_to(source / 'model')
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError(f'Incomplete model file: {relative}')
        files.append({**record, 'path': relative.as_posix()})
    if not files:
        raise ValueError('No verified model files for this checkpoint')

    profile = copy.deepcopy(read_json(workspace_path(config['base_profile'])))
    for key, label in (('head', 'head'), ('wrist_left', 'left_wrist'), ('wrist', 'right_wrist')):
        spec, trained = profile['cameras'][key], camera[label]
        wh = spec['raw_wh']
        oriented_wh = wh[::-1] if spec['live_to_training_rotation_deg'] in (90, 270) else wh
        if oriented_wh != trained['source_wh']:
            raise ValueError(f'{key}: live orientation/dimensions differ from supplied camera contract')
        spec['crop_xyxy'] = trained['crop_xyxy'] or [0, 0, *trained['source_wh']]
        spec['output_wh'] = trained['output_wh']
    profile['inference_checkpoint'] = {'variant': version, 'step': step,
        'source': str(source), 'camera_profile_sha256': digest(supplied('camera_profile.json'))}
    validate_profile(profile, live=True, execute=profile.get('execution_enabled') is True)

    experiment = {'schema': 1, 'training_mode': 'wam', 'framework_commit': FRAMEWORK_COMMIT,
        'data': {'name': 'yellowstick_' + version,
                 'sha256': {'manifest': contract['data_manifest_sha256']}},
        'action': {'horizon': 16, 'fps': 20, 'dim': 8, 'domain_id': 33,
                   'num_embodiment_domains': 34, 'names': JOINT_NAMES, 'first_offset': 1},
        'clip_audit': {'frames': 17, 'future_actions': 16, 'frame_interval_ns': 50_000_000,
                      'passed': True, 'scope': 'Supplied CPU contract checks and saved model/input contract',
                      'source': str(source / 'CPU_CONTRACT_VERIFIED.json')},
        'instruction': inference['instruction'], 'variant': version, 'step': step,
        'source_checkpoint': str(source), 'gpu_inference_tested': False}

    adapter_files = {p.relative_to(source).as_posix(): supplied(p.relative_to(source).as_posix())
                     for p in sorted((source / 'bg2_cosmos').glob('*.py'))}
    adapter_provenance = {'source_bundle': str(source), 'source_case': version,
        'framework_commit': FRAMEWORK_COMMIT, 'data_manifest_sha256': contract['data_manifest_sha256'],
        'normalization_sha256': canonical_hash(contract['normalization']),
        'model_target': training['model']['_target_'], 'domain_id': 33, 'action_dim': 8,
        'max_action_dim': 64, 'future_steps': 16, 'conditioning_fps': 20,
        'files': {name: {'source': str(source / name), 'sha256': digest(data), 'bytes': len(data)}
                  for name, data in adapter_files.items()}}
    model = copy.deepcopy(training['model'])
    model['config']['vlm_config']['tokenizer']['tokenizer_type'] = '/assets/tokenizer'
    model['config']['tokenizer']['vae_path'] = '/assets/Wan2.2_VAE.pth'
    # Native inference uses only this model subtree: no training warm-start,
    # skipped action heads, optimizer, scheduler or trainer resume configuration.
    checkpoint_relative = Path(tier) / f'{step:06d}'
    generated = {checkpoint_relative / 'config.yaml': yaml.safe_dump({'model': model}, sort_keys=False).encode(),
        checkpoint_relative / 'training_config.yaml': training_bytes,
        checkpoint_relative / 'input_contract.json': supplied('input_contract.json'),
        checkpoint_relative / 'vision_contract.py': supplied('vision_contract.py')}

    def add_json(name, value):
        generated[Path(name)] = (json.dumps(value, indent=2) + '\n').encode()

    add_json(checkpoint_relative / 'READY.json', {'step': step, 'variant': tier + '_wam',
        'model_tier': tier, 'code_variant': 'cosmos_wam', 'format': 'pytorch_distributed_checkpoint',
        'source_checkpoint': str(source), 'dataset_variant': version})
    add_json(checkpoint_relative / 'VERIFIED.json', {'step': step,
        'format': 'pytorch_distributed_checkpoint', 'files': files,
        'source_receipt': str(source_root / 'TRANSFER_VERIFIED.json')})
    add_json(checkpoint_relative / 'TRANSFER_VERIFIED.json', {'state': 'complete', 'verified': True,
        'sha256_verified': True, 'tier': tier, 'step': step, 'source': str(source),
        'source_receipt': str(source_root / 'TRANSFER_VERIFIED.json'),
        'verification': 'Reuses supplied SHA256 transfer receipt; local shard sizes checked, not rehashed'})
    add_json('experiment_contract.json', experiment)
    add_json('adapter/SOURCE.json', adapter_provenance)
    generated.update({Path('adapter') / name: data for name, data in adapter_files.items()})
    profile_data = (json.dumps(profile, indent=2) + '\n').encode()
    # Refuse to overwrite changed generated files or an operator-edited profile.
    for path, data in [(destination / name, data) for name, data in generated.items()] + [(profile_path, profile_data)]:
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f'Existing file differs; review before replacing: {path}')
    for name, data in generated.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (destination / checkpoint_relative / 'model').mkdir(exist_ok=True)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_bytes(profile_data)
    publication = validate_checkpoint(destination / checkpoint_relative, tier, step, model_dir=source / 'model')
    validate_experiment(destination / 'experiment_contract.json', publication['contract'])
    verify_adapter(destination / 'adapter', publication['contract'])
    return {'variant': version, 'step': step, 'prepared_root': str(destination),
            'model_dir': str(source / 'model'), 'gateway_profile': str(profile_path),
            'metadata_valid': True, 'model_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config), indent=2))


if __name__ == '__main__':
    main()
