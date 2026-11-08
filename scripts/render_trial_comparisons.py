#!/usr/bin/env python3
"""Offline camera/forecast comparisons with real-time playback and Planning overlays.

Reads saved recordings only. No ROS, policy requests, CUDA, or live robot connections.
"""
import argparse
from bisect import bisect_left, bisect_right
from contextlib import ExitStack
from fractions import Fraction
import json
import math
from pathlib import Path
import statistics
import time
from io import BytesIO

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


FPS = 20
VIEW_SIZE = (512, 640)
CANVAS_SIZE = (1072, 784)
LEFT, RIGHT, TOP = 16, 544, 128


def read_json(path):
    return json.loads(Path(path).read_text())


class Camera:
    """Read original JPEG packets in timestamp order, decoding selected frames only."""
    def __init__(self, run, key, spec, stack):
        self.key, self.spec = key, spec
        self.rows = [json.loads(line) for line in
                     (run / 'cameras' / (key + '.timestamps.jsonl')).read_text().splitlines()]
        self.times = [row['source_time_ns'] for row in self.rows]
        if not self.times or any(a >= b for a, b in zip(self.times, self.times[1:])):
            raise ValueError('Missing or non-increasing camera timestamps: ' + key)
        if [row['frame'] for row in self.rows] != list(range(len(self.rows))):
            raise ValueError('Non-contiguous camera frame index: ' + key)
        source = stack.enter_context(av.open(str(run / 'cameras' / (key + '.mkv'))))
        self.packets = (packet for packet in source.demux(video=0) if packet.size)
        self.index, self.image = -1, None
        self.max_sample_error_ns = 0

    def at(self, stamp):
        after = bisect_left(self.times, stamp)
        choices = [i for i in (after - 1, after) if 0 <= i < len(self.times)]
        chosen = min(choices, key=lambda i: (abs(self.times[i] - stamp), self.times[i]))
        self.max_sample_error_ns = max(self.max_sample_error_ns, abs(self.times[chosen] - stamp))
        if chosen == self.index:
            return self.image
        if chosen < self.index:
            raise ValueError('Camera playback must not move backwards')
        packet = None
        while self.index < chosen:
            packet = next(self.packets)
            self.index += 1
        with Image.open(BytesIO(bytes(packet))) as raw:
            if list(raw.size) != self.spec['raw_wh']:
                raise ValueError('Unexpected raw camera dimensions: ' + self.key)
            image = raw.convert('RGB').rotate(self.spec['live_to_training_rotation_deg'], expand=True)
            image = image.crop(self.spec['crop_xyxy'])
            size = tuple(self.spec['output_wh'])
            self.image = (ImageOps.pad(image, size, method=Image.Resampling.BILINEAR,
                                      color=(0, 0, 0), centering=(.5, .5))
                          if self.key == 'head' else image.resize(size, Image.Resampling.BILINEAR))
        return self.image

    def verify(self):
        count = self.index + 1 + sum(1 for _ in self.packets)
        if count != len(self.rows):
            raise ValueError(f'{self.key}: {count} video packets, {len(self.rows)} timestamp records')


def title_lines(text, font, width):
    lines, line = [], ''
    for word in text.split():
        candidate = (line + ' ' + word).strip()
        if line and font.getlength(candidate) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    if len(lines) > 2:
        raise ValueError('Instruction requires more than two title lines')
    return lines


def load_chunks(run, head, clock_offset, raw_overrides=None):
    raw_overrides = raw_overrides or {}
    timeline = read_json(run / 'imagined/timeline.json')
    by_id = {entry['request_id']: entry for entry in timeline['chunks']}
    forwarded = []
    for line in (run / 'events.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event.get('event') == 'forwarded':
            forwarded.append(event)
    # Camera sidecars link the robot's source clock to 5090's monotonic clock.
    camera_offset = int(statistics.median(row['source_time_ns'] - row['received_monotonic_ns']
                                          for row in head.rows))
    chunks = []
    for event in forwarded:
        ident = event['request_id']
        prediction = run / 'predictions' / ident / 'prediction.json'
        reply = prediction.stat().st_mtime_ns - clock_offset + camera_offset
        observed = event['source_time_ns']
        extra = (reply - observed) / 1e9 - event['roundtrip_seconds']
        if not 0 <= reply - observed or abs(extra) > .25:
            raise ValueError(f'Cannot reliably reconstruct reply time for {ident}: {extra:.3f}s discrepancy')
        if chunks and reply < chunks[-1]['reply_source_time_ns'] + 800_000_000:
            raise ValueError('Overlapping forwarded chunks in a synchronous recording')
        path = (Path(raw_overrides[ident]) if ident in raw_overrides else
                run / 'imagined' / by_id[ident]['directory'] / 'decoded_rollout.npy')
        if not path.is_file():
            raise ValueError('Missing raw imagined frames: ' + str(path))
        chunks.append({'request_id': ident, 'observation_source_time_ns': observed,
                       'reply_source_time_ns': reply, 'raw_frames': str(path),
                       'roundtrip_seconds': event['roundtrip_seconds'],
                       'reply_time_check_difference_seconds': extra})
    if not chunks:
        raise ValueError('No forwarded chunks to compare')
    return chunks, len(timeline['chunks'])


def render(job, runs, output, profile, font_path):
    run = runs / job['run']
    destination = output / job['file']
    if destination.exists():
        raise FileExistsError(destination)
    metadata = read_json(run / 'run.json')
    if metadata.get('instruction') != job['instruction']:
        raise ValueError('Instruction does not match saved run: ' + str(run))
    if metadata.get('episode_state') != 'ended':
        raise ValueError('Only completed recordings may be rendered')
    clock_offset = time.time_ns() - time.monotonic_ns()
    started = time.monotonic()
    title_font = ImageFont.truetype(str(font_path), 30)
    label_font = ImageFont.truetype(str(font_path), 24)
    planning_font = ImageFont.truetype(str(font_path), 32)
    template = Image.new('RGB', CANVAS_SIZE, '#101722')
    draw = ImageDraw.Draw(template)
    lines = title_lines(job['instruction'], title_font, CANVAS_SIZE[0] - 40)
    y = 34 if len(lines) == 1 else 20
    for line in lines:
        draw.text((CANVAS_SIZE[0] // 2, y), line, font=title_font, fill='white', anchor='mm')
        y += 37
    draw.text((LEFT + 256, 103), 'Actual cameras', font=label_font, fill='#e8eef7', anchor='mm')
    draw.text((RIGHT + 256, 103), 'Imagined rollout', font=label_font, fill='#e8eef7', anchor='mm')
    partial = destination.with_name('.' + destination.stem + '.tmp.mp4')
    planning_frames = 0
    try:
        with ExitStack() as stack:
            cameras = {key: Camera(run, key, profile['cameras'][key], stack)
                       for key in ('head', 'wrist_left', 'wrist')}
            chunks, generated = load_chunks(run, cameras['head'], clock_offset,
                                            job.get('raw_frame_overrides'))
            replies = [entry['reply_source_time_ns'] for entry in chunks]
            origin = min(camera.times[0] for camera in cameras.values())
            end = max(camera.times[-1] + int(statistics.median(
                b-a for a, b in zip(camera.times, camera.times[1:]))) for camera in cameras.values())
            frames = math.ceil((end - origin) * FPS / 1e9)
            initial = run / 'predictions' / chunks[0]['request_id'] / 'input_composite.png'
            with Image.open(initial) as image:
                imagined = image.convert('RGB').copy()
            if imagined.size != VIEW_SIZE:
                raise ValueError('Unexpected conditioned composite shape')
            active, raw = -1, None
            outgoing = stack.enter_context(av.open(str(partial), 'w', format='mp4',
                                                   options={'movflags': '+faststart'}))
            stream = outgoing.add_stream('libx264', rate=FPS,
                                         options={'crf': '18', 'preset': 'veryfast'})
            stream.width, stream.height = CANVAS_SIZE
            stream.pix_fmt = 'yuv420p'
            stream.codec_context.thread_count = 3
            for index in range(frames):
                stamp = origin + index * 1_000_000_000 // FPS
                actual = Image.new('RGB', VIEW_SIZE)
                actual.paste(cameras['head'].at(stamp), (0, 0))
                actual.paste(cameras['wrist_left'].at(stamp), (0, 384))
                actual.paste(cameras['wrist'].at(stamp), (256, 384))
                chosen = bisect_right(replies, stamp) - 1
                planning = chosen < 0 or stamp - replies[chosen] >= 800_000_000
                if chosen >= 0:
                    if chosen != active:
                        raw = np.load(chunks[chosen]['raw_frames'], mmap_mode='r', allow_pickle=False)
                        if raw.shape != (17, 640, 512, 3) or raw.dtype != np.uint8:
                            raise ValueError('Unexpected raw WAM rollout shape/dtype')
                        active = chosen
                    future = min(15, (stamp - replies[chosen]) // 50_000_000)
                    imagined = Image.fromarray(raw[1 + future])
                panel = imagined
                if planning:
                    panel = imagined.copy()
                    overlay = ImageDraw.Draw(panel, 'RGBA')
                    overlay.rounded_rectangle((116, 274, 396, 350), radius=14, fill=(8, 15, 28, 220))
                    overlay.text((256, 312), 'Planning...', font=planning_font,
                                 fill=(255, 255, 255, 255), anchor='mm')
                    planning_frames += 1
                canvas = template.copy()
                canvas.paste(actual, (LEFT, TOP))
                canvas.paste(panel, (RIGHT, TOP))
                if index == min(FPS, frames - 1):
                    canvas.save(output / (destination.stem + '_planning.jpg'), quality=92)
                if index == min(frames - 1, max(0, (replies[0] - origin) * FPS // 1_000_000_000 + 7)):
                    canvas.save(output / (destination.stem + '_preview.jpg'), quality=92)
                frame = av.VideoFrame.from_image(canvas)
                frame.pts, frame.time_base = index, Fraction(1, FPS)
                for packet in stream.encode(frame):
                    outgoing.mux(packet)
                if index % (FPS * 15) == 0:
                    print(json.dumps({'event': 'rendering', 'file': destination.name,
                                      'seconds': index / FPS, 'total_seconds': frames / FPS}), flush=True)
            for packet in stream.encode():
                outgoing.mux(packet)
            for camera in cameras.values():
                camera.verify()
        with av.open(str(partial)) as check:
            if (check.streams.video[0].width, check.streams.video[0].height) != CANVAS_SIZE:
                raise ValueError('Rendered dimensions changed')
            decoded = sum(1 for _ in check.decode(video=0))
        if decoded != frames:
            raise ValueError('Rendered frame count mismatch')
        partial.rename(destination)
        report = {'file': destination.name, 'instruction': job['instruction'], 'source_run': run.name,
                  'recorded_episode_id': metadata.get('episode_id'), 'fps': FPS,
                  'frames': frames, 'duration_seconds': frames / FPS, 'width': CANVAS_SIZE[0],
                  'height': CANVAS_SIZE[1], 'camera_playback_speed': 1,
                  'camera_layout': 'head above left wrist (left) and right wrist (right); training crop/rotation',
                  'planning_overlay': 'Planning...', 'planning_frames': planning_frames,
                  'forwarded_chunks': len(chunks), 'recorded_imagined_chunks': generated,
                  'recovered_raw_chunk_overrides': job.get('raw_frame_overrides', {}),
                  'timing_method': 'Prediction JSON save time mapped from 5090 wall time through monotonic camera receipt/source timestamps. '
                                   'Approximate response availability, not an exact FFW acceptance timestamp. '
                                   'Each forwarded forecast plays for 0.8 s; final frame holds while Planning. '
                                   'Results not forwarded before episode shutdown are not played.',
                  'source_origin_ns': origin, 'source_end_ns': end,
                  'wall_minus_monotonic_ns_at_render': clock_offset,
                  'camera_max_sampling_error_ms': {key: camera.max_sample_error_ns / 1e6
                                                   for key, camera in cameras.items()},
                  'chunk_timing': chunks, 'bytes': destination.stat().st_size,
                  'render_seconds': time.monotonic() - started, 'verified': True}
        (output / (destination.stem + '.json')).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({key: report[key] for key in ('file', 'frames', 'duration_seconds', 'render_seconds', 'verified')}), flush=True)
        return report
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--runs-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = read_json(args.jobs)
    profile = read_json(args.profile)
    reports = [render(job, args.runs_dir, args.output, profile, args.font) for job in jobs]
    (args.output / 'manifest.json').write_text(json.dumps({'videos': reports,
        'source_recordings_modified': False, 'hardware_connections': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
