"""Best-effort live decision dashboard. Telemetry errors never decide robot actions."""

import json
import logging
from pathlib import Path

from .artifact_protocol import atomic_json
from .sg2_replay import animation


class SelectionMonitor:
    def __init__(self, output):
        self.output = Path(output).expanduser()
        self.output.mkdir(parents=True, exist_ok=True)
        self.gate = None
        self.report = {}

    def __call__(self, event):
        try:
            report = {**self.report, **event}
            if event["state"] == "inferring":
                report.pop("predicted_video", None)
                report.pop("observed_video", None)
            if (
                self.gate is not None
                and self.gate.last_observation is not None
                and event["state"] == "selected"
            ):
                report["observed_video"] = animation(self.gate.last_observation["past"])
                report["predicted_video"] = animation(self.gate.last_result["future"])
            self.report = report
            with (self.output / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            atomic_json(self.output / "report.json", report)
            data = json.dumps(report, allow_nan=False).replace("<", "\\u003c")
            html = """<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="2">
<title>SG2 CM1 / CM2 · JEPA</title><style>body{font:18px system-ui;background:#111827;color:#eee;margin:30px}img{width:40%;max-width:512px}pre{white-space:pre-wrap}</style>
<h1>SG2 CM1 / CM2 · JEPA</h1><p>Observed history / selected forecast</p>
<img id="past"><img id="future"><pre id="status"></pre><script>const r=REPORT;
if(r.observed_video)document.getElementById('past').src=r.observed_video;
if(r.predicted_video)document.getElementById('future').src=r.predicted_video;
const {observed_video,predicted_video,...status}=r;document.getElementById('status').textContent=JSON.stringify(status,null,2);</script>""".replace(
                "REPORT", data
            )
            temporary = self.output / ".index.html.tmp"
            temporary.write_text(html)
            temporary.replace(self.output / "index.html")
        except Exception as error:  # noqa: BLE001 -- telemetry must not change action selection
            logging.getLogger(__name__).warning("Selection telemetry failed: %s", error)
