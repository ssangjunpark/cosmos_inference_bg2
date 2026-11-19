#!/usr/bin/env python3
"""Render a saved policy run or v2/v1 sequence; inputs are always read-only.

Requires PyAV, NumPy and Pillow (available in cosmos-bg2-gateway:local).
The top strip is the physical head camera. The policy strips are imagined.
"""
import argparse
from bisect import bisect_right
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import av
from PIL import Image, ImageDraw, ImageFont

from render_trial_comparisons import Camera, read_json


FPS = 30
SIZE = (1280, 1600)
FORECAST_NS = 800_000_000
STEP_NS = 50_000_000
THEMES = {
    'dark': {'background': '#101722', 'muted': '#adbacd', 'text': '#ffffff',
             'overlay': (8, 15, 28), 'badge_text': '#e8eef7', 'extra_text': '#f4d17b'},
    'white': {'background': '#ffffff', 'muted': '#536174', 'text': '#17212f',
              'overlay': (255, 255, 255), 'badge_text': '#17212f', 'extra_text': '#805500'},
}


@dataclass(frozen=True)
class Forecast:
    request_id: str
    observed_ns: int
    ready_ns: int
    first_frame: int
    kind: str


def sample(forecasts, stamp):
    """Return movie frame/status without exposing any forecast before its reply."""
    if stamp < forecasts[0].observed_ns:
        return None, 'waiting', False
    chosen = bisect_right([entry.ready_ns for entry in forecasts], stamp) - 1
    if chosen < 0:
        return None, 'inference', False
    entry = forecasts[chosen]
    elapsed = stamp - entry.ready_ns
    frame = entry.first_frame + min(15, elapsed // STEP_NS)
    following = forecasts[chosen + 1] if chosen + 1 < len(forecasts) else None
    inference = bool(following and following.observed_ns <= stamp < following.ready_ns)
    if elapsed < FORECAST_NS:
        return frame, entry.kind, inference
    if inference:
        return frame, 'inference', True
    return frame, 'complete' if following is None else 'holding', False


def load_forecasts(run, version, server_outputs, wall_to_source, *, include_pause_extra=True):
    timeline = read_json(run / 'imagined/timeline.json')
    if timeline.get('partial') or timeline.get('pending'):
        raise ValueError('Imagined recording is incomplete: ' + str(run))
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
    forwarded = {e['request_id']: e for e in events if e.get('event') == 'forwarded'}
    forecasts, evidence = [], []
    for entry in timeline['chunks']:
        ident = entry['request_id']
        candidates = [run / 'predictions' / ident / 'prediction.json']
        for server in ('server-wam-edge-' + version, 'server-wam-edge'):
            candidates.append(server_outputs / server / 'runs' / run.name / 'requests' /
                              entry['epoch'] / ident / 'prediction.json')
        prediction = next((path for path in candidates if path.is_file()), None)
        if prediction is None:
            raise FileNotFoundError('No saved prediction metadata for ' + ident)
        response = read_json(prediction)
        if response['request_id'] != ident or response['source_time_ns'] != entry['source_time_ns']:
            raise ValueError('Prediction identity mismatch: ' + ident)
        ready = prediction.stat().st_mtime_ns - wall_to_source
        check = None
        if ident in forwarded:
            check = (ready - entry['source_time_ns']) / 1e9 - forwarded[ident]['roundtrip_seconds']
            if abs(check) > .25:
                raise ValueError(f'Unreliable prediction save time for {ident}: {check:.3f}s')
        if ready < entry['source_time_ns']:
            raise ValueError('Prediction predates its observation')
        kind = 'rollout' if ident in forwarded else 'stopped'
        forecasts.append(Forecast(ident, entry['source_time_ns'], ready, entry['first_frame'], kind))
        evidence.append({'request_id': ident, 'prediction_file': str(prediction),
                         'forwarded_to_ffw': ident in forwarded,
                         'reply_time_check_difference_seconds': check})
    diagnostic = {'not_before_source_ns': None}
    if include_pause_extra:
        diagnostic = read_json(run / 'diagnostics/pause/diagnostic.json')
        if (diagnostic['state'] != 'complete' or diagnostic['executed'] or
                diagnostic['submitted_to_ffw'] or not diagnostic['diagnostic_only']):
            raise ValueError('Expected a completed, unexecuted pause diagnostic')
        prediction = run / 'diagnostics/pause/prediction.json'
        forecasts.append(Forecast(diagnostic['request_id'], diagnostic['source_time_ns'],
                                  prediction.stat().st_mtime_ns - wall_to_source,
                                  timeline['frame_count'], 'extra'))
    if any(b.ready_ns < a.ready_ns + FORECAST_NS for a, b in zip(forecasts, forecasts[1:])):
        raise ValueError('Forecasts overlap; cannot show all saved frames at real-time speed')
    return forecasts, evidence, diagnostic


class Movie:
    """Sequentially decode the selected imagined export, retaining one frame."""
    def __init__(self, path, expected_frames, stack):
        self.path = path
        source = stack.enter_context(av.open(str(path)))
        stream = source.streams.video[0]
        if stream.frames != expected_frames or stream.average_rate != 20:
            raise ValueError('Unexpected imagined frame count/rate: ' + str(path))
        self.frames = iter(source.decode(video=0))
        self.index, self.image = -1, None

    def at(self, index):
        if index < self.index:
            raise ValueError('Imagined movie cannot move backwards')
        while self.index < index:
            self.image = next(self.frames).to_image()
            self.index += 1
        return self.image

    def verify(self, expected_frames):
        if self.index + 1 != expected_frames or next(self.frames, None) is not None:
            raise ValueError('Not all imagined frames were presented: ' + str(self.path))


class PolicyStrip:
    def __init__(self, run, version, label, accent, forecasts, font, stack, theme):
        self.forecasts, self.version = forecasts, version
        self.label = label
        self.theme = theme
        self.has_extra = any(entry.kind == 'extra' for entry in forecasts)
        self.frames = forecasts[-1].first_frame + 16
        filenames = (('head_extra.mp4', 'wrist_left_extra.mp4', 'wrist_right_extra.mp4') if self.has_extra else
                     ('head.mp4', 'wrist_left.mp4', 'wrist_right.mp4'))
        self.movies = [Movie(run / 'imagined' / filename, self.frames, stack) for filename in
                       filenames]
        self.fonts = {size: ImageFont.truetype(str(font), size) for size in (17, 20, 24, 28)}
        self.template = Image.new('RGB', (1280, 416), theme['background'])
        draw = ImageDraw.Draw(self.template)
        draw.rounded_rectangle((8, 0, 60, 415), radius=12, fill=accent)
        text = Image.new('RGBA', (380, 48))
        ImageDraw.Draw(text).text((190, 24), label, font=self.fonts[28], fill='white', anchor='mm')
        self.template.paste(text.rotate(90, expand=True), (10, 18), text.rotate(90, expand=True))
        for x, title in ((76, 'IMAGINED HEAD  ·  ' + version.upper()),
                         (604, 'LEFT WRIST'), (932, 'RIGHT WRIST')):
            draw.text((x, 11), title, font=self.fonts[17], fill=theme['muted'], anchor='lm')
        self.cached_index, self.cached_panel = None, self.template.copy()
        self.counts = {}
        self.seen_frames = set()

    def at(self, stamp):
        index, state, concurrent = sample(self.forecasts, stamp)
        self.counts[state] = self.counts.get(state, 0) + 1
        if index is not None:
            self.seen_frames.add(index)
        if index != self.cached_index:
            panel = self.template.copy()
            for movie, xy, size in zip(self.movies, ((76, 28), (604, 58), (932, 58)),
                                       ((512, 384), (324, 324), (324, 324))):
                panel.paste(movie.at(index).resize(size, Image.Resampling.BILINEAR), xy)
            self.cached_index, self.cached_panel = index, panel
        panel = self.cached_panel.copy()
        draw = ImageDraw.Draw(panel, 'RGBA')
        labels = {'waiting': 'Waiting for ' + self.label.lower(),
                  'inference': 'Inference…', 'rollout': 'Imagined rollout',
                  'stopped': 'Stopped forecast · not executed',
                  'extra': 'Extra forecast · not executed',
                  'complete': 'Complete · ' + ('extra' if self.has_extra else 'final') + ' forecast held',
                  'holding': 'Waiting for next observation'}
        if state in ('waiting', 'inference'):
            draw.rectangle((76, 28, 1256, 412), fill=(*self.theme['overlay'], 125))
            draw.rounded_rectangle((92, 178, 572, 258), radius=14, fill=(*self.theme['overlay'], 235))
            draw.text((332, 218), labels[state], font=self.fonts[24], fill=self.theme['text'], anchor='mm')
        else:
            color = self.theme['extra_text'] if state in ('stopped', 'extra', 'complete') else self.theme['badge_text']
            width = math.ceil(self.fonts[20].getlength(labels[state])) + 24
            draw.rounded_rectangle((88, 40, 88 + width, 76), radius=8, fill=(*self.theme['overlay'], 230))
            draw.text((100, 58), labels[state], font=self.fonts[20], fill=color, anchor='lm')
        if concurrent and state != 'inference':
            draw.rounded_rectangle((812, 386, 1256, 414), radius=6, fill=(*self.theme['overlay'], 235))
            draw.text((1034, 400), 'Inference… (next forecast)', font=self.fonts[17],
                      fill=self.theme['text'], anchor='mm')
        return panel

    def verify(self):
        if self.seen_frames != set(range(self.frames)):
            raise ValueError('Some selected imagined frames were skipped')
        for movie in self.movies:
            movie.verify(self.frames)


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def render(args):
    single = args.run is not None
    sequence = (args.run if single else args.sequence).resolve()
    size = (1280, 1176) if single else SIZE
    output = args.output.resolve()
    if output == sequence or sequence in output.parents:
        raise ValueError('Save the comparison outside the original sequence directory')
    output.mkdir(parents=True, exist_ok=True)
    destination, partial = output / 'comparison.mp4', output / '.comparison.tmp.mp4'
    if destination.exists() or partial.exists():
        raise FileExistsError(destination)
    if single:
        recorded = read_json(sequence / 'run.json')
        if recorded.get('episode_state') != 'ended':
            raise ValueError('Only completed recordings may be rendered')
        version = recorded.get('model_id')
        if version is None:
            version = next((v for v in ('v1', 'v2') if ('-' + v + '-') in sequence.name), None)
        if version not in ('v1', 'v2'):
            raise ValueError('Cannot identify saved policy version')
        metadata = {'recorder': '.', 'runs': {version: '.'}}
    else:
        metadata = read_json(sequence / 'sequence.json')
        if metadata['state'] != 'complete':
            raise ValueError('Only completed sequences may be rendered')
    source_videos = sorted(sequence.rglob('*.mp4')) + sorted(sequence.rglob('*.mkv'))
    originals = {str(p.relative_to(sequence)): digest(p) for p in source_videos}
    theme = THEMES[args.theme]
    reference = read_json(args.timing_reference) if args.timing_reference else None
    if reference and (not reference.get('verified') or reference['source_video_sha256'] != originals or
                      reference['fps'] != FPS or (reference['width'], reference['height']) != size):
        raise ValueError('Timing reference must match these original recordings and output format')
    font = ImageFont.truetype(str(args.font), 30)
    small = ImageFont.truetype(str(args.font), 22)
    started = time.monotonic()
    previews, reports = {}, {}
    try:
        with ExitStack() as stack:
            recorder = sequence / metadata['recorder']
            spec = {'raw_wh': [672, 376], 'live_to_training_rotation_deg': 0,
                    'crop_xyxy': [0, 0, 672, 376], 'output_wh': [1126, 630]}
            camera = Camera(recorder, 'head', spec, stack)
            source_minus_mono = int(statistics.median(row['source_time_ns'] - row['received_monotonic_ns']
                                                      for row in camera.rows))
            wall_minus_mono = time.time_ns() - time.monotonic_ns()
            wall_to_source = wall_minus_mono - source_minus_mono
            origin = camera.times[0]
            camera_end = camera.times[-1] + int(statistics.median(b-a for a, b in zip(camera.times, camera.times[1:])))
            end = camera_end
            if reference and (origin != reference['source_origin_ns'] or
                              camera_end != reference.get('camera_source_end_ns', reference['source_end_ns'])):
                raise ValueError('Timing reference uses a different camera interval')
            strips = []
            for version, label, accent in [('v2', 'Original policy', '#246aa0'),
                                            ('v1', 'Recovery policy', '#437959')]:
                if version not in metadata['runs']:
                    continue
                run = sequence / metadata['runs'][version]
                if reference:
                    previous = reference['policies'][version]
                    if previous['source_run'] != run.name:
                        raise ValueError('Timing reference uses a different policy run')
                    forecasts = [Forecast(**entry) for entry in previous['forecasts']]
                    evidence = previous['timing_evidence']
                    diagnostic = {'not_before_source_ns': previous['pause_source_time_ns']}
                else:
                    forecasts, evidence, diagnostic = load_forecasts(run, version, args.server_outputs, wall_to_source,
                                                                      include_pause_extra=not (single or args.no_pause_extra))
                excluded = []
                if args.no_pause_extra:
                    excluded = [entry.request_id for entry in forecasts if entry.kind == 'extra']
                    forecasts = [entry for entry in forecasts if entry.kind != 'extra']
                if not forecasts:
                    raise ValueError('No original imagined forecasts to show')
                if single:
                    end = max(end, forecasts[-1].ready_ns + FORECAST_NS)
                elif forecasts[-1].ready_ns + FORECAST_NS > end:
                    raise ValueError('Continuous recording ends before the final forecast can be shown')
                strip = PolicyStrip(run, version, label, accent, forecasts, args.font, stack, theme)
                strips.append(strip)
                reports[version] = {'source_run': run.name, 'label': label,
                                    'imagined_frames_per_view': strip.frames,
                                    'pause_extra_included': strip.has_extra,
                                    'excluded_pause_forecast_request_ids': excluded,
                                    'forecasts': [vars(entry) for entry in forecasts],
                                    'timing_evidence': evidence,
                                    'pause_source_time_ns': diagnostic['not_before_source_ns']}
                previews[version + '_rollout'] = math.ceil((forecasts[0].ready_ns - origin) * FPS / 1e9) + 6
                previews[version + ('_extra' if strip.has_extra else '_final')] = math.ceil(
                    (forecasts[-1].ready_ns - origin) * FPS / 1e9) + 6
            previews['inference'] = math.ceil((strips[0].forecasts[0].observed_ns - origin) * FPS / 1e9) + 6
            frames = math.ceil((end-origin) * FPS / 1e9)
            if reference and (end != reference['source_end_ns'] or frames != reference['frames']):
                raise ValueError('Timing reference uses a different playback interval')
            movie = stack.enter_context(av.open(str(partial), 'w', format='mp4', options={'movflags': '+faststart'}))
            stream = movie.add_stream('libx264', rate=FPS, options={'crf': '18', 'preset': 'veryfast'})
            stream.width, stream.height, stream.pix_fmt = *size, 'yuv420p'
            stream.codec_context.thread_count = 2
            for index in range(frames):
                stamp = origin + index * 1_000_000_000 // FPS
                canvas = Image.new('RGB', size, theme['background'])
                draw = ImageDraw.Draw(canvas)
                draw.text((640, 32), 'Move the yellow stick into the basket.', font=font, fill=theme['text'], anchor='mm')
                actual_label = ('REAL HEAD CAMERA · recording ended; final frame held' if stamp >= camera_end else
                                'REAL HEAD CAMERA')
                draw.text((76, 78), actual_label, font=small, fill=theme['muted'], anchor='lm')
                draw.text((1204, 78), f'{index/FPS:05.1f} s', font=small, fill=theme['muted'], anchor='rm')
                canvas.paste(camera.at(min(stamp, camera.times[-1]) if single else stamp), (77, 100))
                for strip, y in zip(strips, (752, 1176)):
                    canvas.paste(strip.at(stamp), (0, y))
                for label, preview_index in previews.items():
                    if index == preview_index:
                        canvas.save(output / ('preview_' + label + '.jpg'), quality=93)
                frame = av.VideoFrame.from_image(canvas)
                frame.pts, frame.time_base = index, Fraction(1, FPS)
                for packet in stream.encode(frame):
                    movie.mux(packet)
                if index % (FPS*10) == 0:
                    print(json.dumps({'event': 'rendering', 'seconds': index/FPS,
                                      'total_seconds': frames/FPS}), flush=True)
            for packet in stream.encode():
                movie.mux(packet)
            camera.verify()
            for strip in strips:
                strip.verify()
                reports[strip.version]['output_frame_states'] = strip.counts
        with av.open(str(partial)) as check:
            video = check.streams.video[0]
            if (video.width, video.height) != size or sum(1 for _ in check.decode(video=0)) != frames:
                raise ValueError('Comparison encoding verification failed')
        for name, expected in originals.items():
            if digest(sequence / name) != expected:
                raise ValueError('Source video changed: ' + name)
        report = {'sequence': str(sequence), 'video': destination.name, 'fps': FPS,
                  'theme': args.theme, 'timing_reference': str(args.timing_reference) if args.timing_reference else None,
                  'width': size[0], 'height': size[1], 'frames': frames, 'strip_count': 1 + len(strips),
                  'duration_seconds': frames/FPS, 'playback_speed': 1,
                  'source_origin_ns': origin, 'source_end_ns': end,
                  'camera_source_end_ns': camera_end,
                  'camera_duration_seconds': (camera_end-origin)/1e9,
                  'camera_final_frame_hold_seconds': max(0, end-camera_end)/1e9,
                  'timing_method': 'Camera source timestamps; forecast availability from original prediction JSON '
                                   'save times mapped through camera source/monotonic receipts and host wall/monotonic '
                                   'offset. Forwarded roundtrip durations cross-check within 250 ms. Approximate '
                                   'response availability, not exact FFW acceptance. Every saved 16-frame forecast '
                                   'plays at 20 fps, with inference gaps and final-frame holds.',
                  'wrist_views': 'imagined', 'head_top': 'continuous physical camera, full frame',
                  'pause_extra_included': any(strip.has_extra for strip in strips),
                  'diagnostic_forecasts_executed': False, 'policies': reports,
                  'camera_max_sampling_error_ms': camera.max_sample_error_ns/1e6,
                  'source_video_sha256': originals, 'source_videos_unchanged': True,
                  'bytes': partial.stat().st_size, 'render_seconds': time.monotonic()-started,
                  'verified': True}
        (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
        partial.rename(destination)
        print(json.dumps({'event': 'comparison_saved', 'directory': str(output), 'frames': frames,
                          'duration_seconds': frames/FPS, 'verified': True}), flush=True)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--sequence', type=Path)
    source.add_argument('--run', type=Path, help='Single-policy recording; two strips, original forecasts only')
    parser.add_argument('--server-outputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--theme', choices=THEMES, default='dark')
    parser.add_argument('--no-pause-extra', action='store_true',
                        help='Play full original forecasts and exclude the additional pause diagnostic')
    parser.add_argument('--timing-reference', type=Path,
                        help='Reuse exact forecast timing from an earlier verified comparison.json')
    parser.add_argument('--font', type=Path, default=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    render(parser.parse_args())


if __name__ == '__main__':
    main()
