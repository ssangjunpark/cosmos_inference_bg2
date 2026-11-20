"""Observed camera history sampled on the same clock/grid as the SG2 future."""

import numpy as np
from PIL import Image

from .core import require
from .observations import Observations, decode_view


class HistoryObservations(Observations):
    def __init__(
        self,
        *args,
        history_frames=2,
        frame_interval_ns=50_000_000,
        video_view="head",
        video_size=(256, 192),
        history_required=None,
        **kwargs,
    ):
        require(
            type(history_frames) is int
            and 0 < history_frames <= 48
            and history_frames % 2 == 0,
            "History must contain 2..48 frames in complete JEPA tubelets",
        )
        kwargs.setdefault("max_samples", max(64, history_frames * 3))
        super().__init__(*args, **kwargs)
        require(frame_interval_ns == 50_000_000, "SG2 WAM currently predicts at 20 fps")
        require(
            video_view in self.cameras
            and len(video_size) == 2
            and 0 < min(video_size) <= max(video_size) <= 512,
            "Invalid JEPA view or size",
        )
        self.history_frames, self.interval = history_frames, frame_interval_ns
        self.history_required = history_required or (lambda: True)
        self.video_view, self.video_size = video_view, tuple(video_size)

    def snapshot(
        self,
        instruction,
        now_ns,
        *,
        receipt_now_ns=None,
        not_before_ns=0,
        include_raw=False,
    ):
        snapshot = super().snapshot(
            instruction,
            now_ns,
            receipt_now_ns=receipt_now_ns,
            not_before_ns=not_before_ns,
            include_raw=include_raw,
        )
        result, raw = snapshot if include_raw else (snapshot, None)
        source = int(result["source_time_ns"])
        tolerance = min(self.interval // 2 - 1, 25_000_000)
        with self.lock:
            available = [
                frame
                for frame in self.cameras[self.video_view]
                if frame[0] >= not_before_ns
            ]
        selected = []
        count = self.history_frames if self.history_required() else 1
        for offset in range(count - 1, -1, -1):
            target = source - offset * self.interval
            require(
                target >= not_before_ns and available,
                "Waiting for camera history after the hold",
            )
            frame = min(available, key=lambda row: (abs(row[0] - target), row[0]))
            require(
                abs(frame[0] - target) <= tolerance,
                "Camera history has a gap or wrong frame rate",
            )
            selected.append(frame)
        require(
            len({frame[0] for frame in selected}) == count,
            "Camera history must use distinct frames",
        )
        spec = self.profile["cameras"][self.video_view]
        past = [
            np.asarray(
                Image.fromarray(decode_view(frame[2], self.video_view, spec)).resize(
                    self.video_size, Image.Resampling.BILINEAR
                )
            )
            for frame in selected
        ]
        result.update(
            past=np.stack(past),
            frame_interval_ns=np.asarray(self.interval),
            past_source_time_ns=np.asarray(
                [frame[0] for frame in selected], dtype=np.int64
            ),
        )
        return (result, raw) if include_raw else result
