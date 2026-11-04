#!/usr/bin/env python3
"""Create an explicitly operator-enabled profile; never connects to the robot."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from bg2_bridge.core import require, validate_profile


def prepare(source, *, snupi_head_override=False):
    validate_profile(source, live=True)
    result = copy.deepcopy(source)
    result['status'] = 'operator_enabled_for_execution'
    result['execution_enabled'] = True
    evidence = result.setdefault('evidence', {})
    evidence['execution_authorization'] = {
        'basis': 'operator requested normal robot execution',
        'physical_tracking_tested': False,
    }
    if snupi_head_override:
        limit = result['reference_limits']['head_joint1']
        require(limit['upper'] == .6951, 'SNUPI override expects the original 0.6951-radian head limit')
        limit['upper'] = .78
        evidence['execution_authorization']['head_joint1_upper_override'] = {
            'original': .6951, 'value': .78,
            'reason': 'operator requested the existing SNUPI initialization pose',
            'scope': 'companion validation and captured head hold; ROBOTIS model is unchanged',
        }
    validate_profile(result, live=True, execute=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snupi-head-override', action='store_true')
    args = parser.parse_args()
    payload = args.source.read_bytes()
    result = prepare(json.loads(payload), snupi_head_override=args.snupi_head_override)
    result['evidence']['execution_authorization']['source_profile_sha256'] = hashlib.sha256(payload).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'status': result['status'],
                      'head_joint1_upper': result['reference_limits']['head_joint1']['upper'],
                      'robot_connection_opened': False}))


if __name__ == '__main__':
    main()
