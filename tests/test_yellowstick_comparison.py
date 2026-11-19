"""Saved-video timing: preserve every forecast, avoid leaking future images."""
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'render'))
from render_yellowstick_comparison import Forecast, load_forecasts, sample


class ComparisonTimingTests(unittest.TestCase):
    def setUp(self):
        self.forecasts = [
            Forecast('normal', 1_000_000_000, 2_000_000_000, 0, 'rollout'),
            Forecast('stopped', 3_000_000_000, 4_000_000_000, 16, 'stopped'),
            Forecast('diagnostic', 3_500_000_000, 5_000_000_000, 32, 'extra'),
        ]

    def test_inactive_recovery_does_not_show_future_imagined_frames(self):
        self.assertEqual(sample(self.forecasts, 500_000_000), (None, 'waiting', False))
        self.assertEqual(sample(self.forecasts, 1_500_000_000), (None, 'inference', False))

    def test_inference_freezes_previous_forecast_until_reply(self):
        self.assertEqual(sample(self.forecasts, 2_900_000_000), (15, 'holding', False))
        self.assertEqual(sample(self.forecasts, 3_200_000_000), (15, 'inference', True))
        self.assertEqual(sample(self.forecasts, 4_000_000_000), (16, 'stopped', True))

    def test_stopped_and_extra_forecasts_are_preserved_and_labelled(self):
        self.assertEqual(sample(self.forecasts, 4_250_000_000), (21, 'stopped', True))
        self.assertEqual(sample(self.forecasts, 4_900_000_000), (31, 'inference', True))
        self.assertEqual(sample(self.forecasts, 5_250_000_000), (37, 'extra', False))
        self.assertEqual(sample(self.forecasts, 6_000_000_000), (47, 'complete', False))

    def test_30fps_output_presents_all_20fps_forecast_frames_in_order(self):
        samples = [sample(self.forecasts, index * 1_000_000_000 // 30)[0] for index in range(190)]
        shown = [index for index in samples if index is not None]
        self.assertEqual(set(shown), set(range(48)))
        self.assertEqual(shown, sorted(shown))

    def test_original_only_finishes_full_stopped_forecast_without_extra_inference(self):
        original = self.forecasts[:-1]
        samples = [sample(original, index * 1_000_000_000 // 30)[0] for index in range(190)]
        self.assertEqual({index for index in samples if index is not None}, set(range(32)))
        self.assertEqual(sample(original, 4_250_000_000), (21, 'stopped', False))
        self.assertEqual(sample(original, 5_250_000_000), (31, 'complete', False))

    def test_older_original_only_recording_loads_stopped_forecast_from_legacy_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'gateway/single-v1'
            (run / 'imagined').mkdir(parents=True)
            chunk = {'request_id': 'stopped', 'epoch': 'epoch', 'source_time_ns': 1_000_000_000,
                     'first_frame': 0, 'last_frame': 15}
            (run / 'imagined/timeline.json').write_text(json.dumps({'chunks': [chunk], 'frame_count': 16}))
            (run / 'events.jsonl').write_text('')
            prediction = root / 'server-wam-edge/runs/single-v1/requests/epoch/stopped/prediction.json'
            prediction.parent.mkdir(parents=True)
            prediction.write_text(json.dumps({'request_id': 'stopped', 'source_time_ns': 1_000_000_000}))
            os.utime(prediction, ns=(3_000_000_000, 3_000_000_000))
            forecasts, evidence, diagnostic = load_forecasts(run, 'v1', root, 1_000_000_000,
                                                             include_pause_extra=False)
            self.assertEqual(forecasts, [Forecast('stopped', 1_000_000_000, 2_000_000_000, 0, 'stopped')])
            self.assertEqual(evidence[0]['prediction_file'], str(prediction))
            self.assertFalse(evidence[0]['forwarded_to_ffw'])
            self.assertEqual(diagnostic['not_before_source_ns'], None)


if __name__ == '__main__':
    unittest.main()
