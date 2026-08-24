"""Prepare and serve a back-to-back pupil-tuning queue."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from analysis.batch_auxiliary import manifest_rows
from analysis.session.auxiliary import discover_pupil_videos
from analysis.session.pupil import (
    PupilTuningGUI,
    camera_frame_times,
    count_csv_frames,
    find_frametimes_csv,
    iter_gray_frames,
    load_pupil_tuning,
    video_recording_timestamp,
)
from analysis.session.respiration import find_behavior_sync
from analysis.session.sync import (
    frame_onset_samples,
    group_frames_into_acquisitions,
    open_sync,
)


def representative_frames(video_paths, sync_path, *, samples_per_video=8):
    """Decode once, validate camera counts, and retain active dim/bright frames."""
    sync = open_sync(sync_path)
    camera_blocks = group_frames_into_acquisitions(
        frame_onset_samples(sync, channel="cameraFrameSync"), rate_hz=sync.rate_hz
    )
    if len(camera_blocks) != len(video_paths):
        raise ValueError(
            f"{len(video_paths)} videos but {len(camera_blocks)} camera clock blocks"
        )
    imaging_blocks = group_frames_into_acquisitions(
        frame_onset_samples(sync), rate_hz=sync.rate_hz
    )
    csv_paths = [find_frametimes_csv(path) for path in video_paths]
    counts = [count_csv_frames(csv_path) if csv_path is not None else len(block)
              for csv_path, block in zip(csv_paths, camera_blocks)]
    video_stamps = [video_recording_timestamp(path) for path in video_paths]
    camera_time_s, alignment_qc = camera_frame_times(
        video_paths, video_stamps, camera_blocks, sync, counts,
        frametimes_paths=csv_paths,
    )
    active = np.zeros(len(camera_time_s), dtype=bool)
    for block in imaging_blocks:
        lo = np.searchsorted(camera_time_s, block[0] / sync.rate_hz, side="left")
        hi = np.searchsorted(camera_time_s, block[-1] / sync.rate_hz, side="right")
        active[lo:hi] = True

    retained = []
    offset = 0
    for path, expected in zip(video_paths, counts):
        local_active = np.flatnonzero(active[offset:offset + expected])
        if local_active.size == 0:
            raise ValueError(f"No acquisition-active camera frames in {path}")
        targets = set(local_active[np.linspace(
            0, local_active.size - 1,
            min(int(samples_per_video), local_active.size), dtype=int,
        )].tolist())
        count = 0
        for index, frame in enumerate(iter_gray_frames(path)):
            if index in targets:
                retained.append(np.asarray(frame))
            count = index + 1
        if count != expected:
            raise ValueError(
                f"{Path(path).name}: {count:,} decoded frames but "
                f"{expected:,} frametimes rows"
            )
        offset += expected
    if not retained:
        raise ValueError("No representative pupil frames were retained")
    means = np.asarray([np.mean(frame) for frame in retained])
    return (retained[int(np.argmin(means))], retained[int(np.argmax(means))],
            counts, alignment_qc)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def prepare_queue(rows, *, imaging_root, scratch_root, refresh=False,
                  samples_per_video=8):
    """Build/update the queue, keeping failures visible and successful caches reusable."""
    scratch_root = Path(scratch_root)
    queue_path = scratch_root / "pupil_tuning" / "queue.json"
    old = {}
    if queue_path.exists() and not refresh:
        old = {int(item["group_id"]): item for item in json.loads(queue_path.read_text())["items"]}
    items = []
    for row in rows:
        group_id = int(row["group_id"])
        cached = old.get(group_id)
        if cached and cached.get("status") == "prepared" and Path(cached["cache"]).exists():
            items.append(cached)
            continue
        item = {"group_id": group_id, "mouse": row["mouse"], "date": row["date"],
                "objective": row["objective"], "status": "failed"}
        try:
            exp_dir = Path(imaging_root) / row["date"] / row["mouse"] / row["exp"]
            exp_name = f"{row['date']}_{row['mouse']}_{row['exp']}"
            videos = discover_pupil_videos(exp_dir)
            sync_path = find_behavior_sync(exp_dir)
            dim, bright, counts, alignment_qc = representative_frames(
                videos, sync_path, samples_per_video=samples_per_video
            )
            cache = scratch_root / "pupil_tuning" / f"group{group_id}_frames.npz"
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, dim=dim, bright=bright)
            output = (exp_dir / "processed" / "python" / "aux" /
                      f"group{group_id}_{exp_name}_pupil_tuning.json")
            item.update(status="prepared", exp_name=exp_name,
                        exp_dir=str(exp_dir), videos=list(map(str, videos)),
                        sync=str(sync_path), video_counts=counts,
                        camera_alignment=alignment_qc,
                        cache=str(cache), output=str(output),
                        tuned=output.exists())
        except Exception as error:
            item["error"] = f"{type(error).__name__}: {error}"
        items.append(item)
        _atomic_json(queue_path, {"version": 1, "items": items})
        print(f"group {group_id}: {item['status']}" +
              (f" — {item['error']}" if "error" in item else ""), flush=True)
    payload = {"version": 1, "items": items}
    _atomic_json(queue_path, payload)
    return queue_path, payload


class TuningQueueApp:
    def __init__(self, doc, queue_path, *, include_complete=False):
        from bokeh.models import Button, Div
        from bokeh.layouts import row

        self.doc = doc
        self.queue_path = Path(queue_path)
        self.include_complete = bool(include_complete)
        payload = json.loads(self.queue_path.read_text())
        self.items = [item for item in payload["items"] if item["status"] == "prepared"]
        if not include_complete:
            self.items = [item for item in self.items if not Path(item["output"]).exists()]
        self.index = 0
        self.header = Div(width=1000)
        self.previous = Button(label="Previous", width=120)
        self.skip = Button(label="Skip", button_type="warning", width=120)
        self.previous.on_click(lambda: self.move(-1))
        self.skip.on_click(lambda: self.move(1))
        self.navigation = row(self.previous, self.skip)
        self.render()

    def move(self, amount):
        if self.items:
            self.index = min(max(self.index + amount, 0), len(self.items) - 1)
        self.render()

    def saved(self, _path):
        self.items[self.index]["tuned"] = True
        payload = json.loads(self.queue_path.read_text())
        for item in payload["items"]:
            if int(item["group_id"]) == int(self.items[self.index]["group_id"]):
                item["tuned"] = True
        _atomic_json(self.queue_path, payload)
        if self.include_complete:
            self.doc.add_next_tick_callback(lambda: self.move(1))
        else:
            self.items.pop(self.index)
            self.index = min(self.index, max(len(self.items) - 1, 0))
            self.doc.add_next_tick_callback(self.render)

    def render(self):
        self.doc.clear()
        if not self.items:
            self.doc.add_root(self.header)
            self.header.text = "<h2>No prepared pupil-tuning sessions are pending.</h2>"
            return
        item = self.items[self.index]
        output = Path(item["output"])
        settings = load_pupil_tuning(output) if output.exists() else {}
        frames = np.load(item["cache"])
        self.header.text = (
            f"<h2>Pupil tuning {self.index + 1}/{len(self.items)} — group "
            f"{item['group_id']}</h2><p>{item['date']} · {item['mouse']} · "
            f"{item['objective']} · {item['exp_name']}</p>"
        )
        self.doc.add_root(self.header)
        gui = PupilTuningGUI(
            frames["dim"], frames["bright"], save_path=output,
            roi=settings.get("roi"),
            threshold_offset=settings.get("threshold_offset", 0.0),
            bright_percentile=settings.get("bright_percentile", 97.0),
            on_save=self.saved,
        )
        gui.modify_doc(self.doc)
        self.doc.add_root(self.navigation)
        self.doc.title = f"Pupil tuning — group {item['group_id']}"


def serve_queue(queue_path, *, port=0, include_complete=False):
    from bokeh.application import Application
    from bokeh.application.handlers.function import FunctionHandler
    from bokeh.server.server import Server

    app = Application(FunctionHandler(
        lambda doc: TuningQueueApp(doc, queue_path, include_complete=include_complete)
    ))
    server = Server({"/": app}, port=int(port), allow_websocket_origin=["localhost:*", "127.0.0.1:*"])
    server.start()
    print(f"Tuning queue: http://localhost:{server.port}/", flush=True)
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="analysis/stage0/ketxyl_16odor_session_manifest.csv")
    parser.add_argument("--imaging-root", default=os.environ.get("ODYN_IMAGING_ROOT", "/Volumes/MossLab/ImagingData"))
    parser.add_argument("--scratch-root", default=os.environ.get("ODYN_SCRATCH_ROOT", str(Path.home() / "odyn_scratch")))
    parser.add_argument("--groups", nargs="*", type=int, default=[])
    parser.add_argument("--samples-per-video", type=int, default=8)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--include-complete", action="store_true")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    queue_path = Path(args.scratch_root) / "pupil_tuning" / "queue.json"
    if args.prepare:
        rows = manifest_rows(args.manifest, args.groups)
        queue_path, _ = prepare_queue(
            rows, imaging_root=args.imaging_root, scratch_root=args.scratch_root,
            refresh=args.refresh, samples_per_video=args.samples_per_video,
        )
    if args.serve:
        if not queue_path.exists():
            parser.error("No queue exists; run with --prepare first")
        serve_queue(queue_path, port=args.port, include_complete=args.include_complete)
    if not args.prepare and not args.serve:
        parser.error("Choose --prepare, --serve, or both")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
