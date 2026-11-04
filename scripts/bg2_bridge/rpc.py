"""Small HTTP protocol, intended for localhost over an SSH tunnel."""
from io import BytesIO
import http.client
import json
import re
from urllib.parse import urlsplit
import zipfile

import numpy as np

from bg2_inference import JOINT_NAMES, load_observation
from .core import BridgeError, require
from .chunks import SCHEMA

MAX_OBSERVATION = 8 * 1024 * 1024
MAX_REPLY = 128 * 1024
FIELDS = {'head', 'wrist', 'wrist_left', 'joint_position', 'joint_names', 'source_time_ns', 'instruction', 'format'}


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{32}', value), 'Expected a 32-character hexadecimal identifier')
    return value


def encode_observation(observation):
    buffer = BytesIO()
    # Cameras are small after preprocessing. Avoid an expensive compression step.
    np.savez(buffer, **observation)
    payload = buffer.getvalue()
    require(len(payload) <= MAX_OBSERVATION, 'Observation is too large')
    return payload


def decode_observation(payload):
    require(0 < len(payload) <= MAX_OBSERVATION, 'Invalid observation size')
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        entries = archive.infolist()
        require(len(entries) == len(FIELDS) and {e.filename for e in entries} == {k + '.npy' for k in FIELDS},
                'Unexpected observation archive entries')
        require(sum(e.file_size for e in entries) <= MAX_OBSERVATION and all(e.file_size >= 0 for e in entries),
                'Expanded observation exceeds size limit')
    observation = load_observation(BytesIO(payload))
    require(len(observation['instruction']) <= 2048, 'Instruction is too long')
    return observation


class InferenceClient:
    def __init__(self, url, timeout=10.):
        parsed = urlsplit(url)
        require(parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost', '::1'} and
                parsed.username is None and parsed.password is None and not parsed.query and not parsed.fragment and
                parsed.path in ('', '/'), 'Use a localhost HTTP endpoint through an SSH tunnel')
        require(0 < timeout <= 120, 'Invalid inference timeout')
        self.host, self.port, self.timeout = parsed.hostname, parsed.port or 8008, timeout

    def _call(self, method, path, body=None, extra=None):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            headers = dict(extra or {})
            if body is not None:
                headers['Content-Type'] = 'application/x-npz'
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read(MAX_REPLY + 1)
            require(len(payload) <= MAX_REPLY, 'Oversized inference response')
            data = json.loads(payload)
            require(response.status == 200 and isinstance(data, dict) and data.get('ok') is True,
                    data.get('error', f'Inference HTTP {response.status}') if isinstance(data, dict) else 'Invalid inference reply')
            return data
        finally:
            connection.close()

    def health(self):
        result = self._call('GET', '/health')
        require(result.get('joint_names') == JOINT_NAMES and result.get('future_steps') == 16 and
                result.get('frame_interval_ns') == 50_000_000, 'Server action contract mismatch')
        require(result.get('warmed') is True, 'Server must be warmed before the gateway starts')
        return result

    def predict(self, observation, epoch, request_id):
        identifier(epoch); identifier(request_id)
        result = self._call('POST', '/predict', encode_observation(observation),
                            {'X-Session-Epoch': epoch, 'X-Request-ID': request_id})
        require(result.get('schema') == SCHEMA and result.get('epoch') == epoch and result.get('request_id') == request_id,
                'Inference reply identity mismatch')
        require(result.get('source_time_ns') == int(observation['source_time_ns']), 'Inference changed the observation timestamp')
        require(result.get('joint_names') == JOINT_NAMES and result.get('units') == ['rad'] * 8, 'Inference joint coordinates mismatch')
        positions = np.asarray(result.get('positions'), dtype=float)
        require(positions.shape == (16, 8) and np.isfinite(positions).all(), 'Invalid predicted positions')
        return result
