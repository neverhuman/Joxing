"""Decode the preregistered diagnostic MP4, with zero new physics."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from joxing.motor_storage import RawCredit, write_json


def decode_bounded(reader, maximum=26):
    """Avoid reader.__len__/length_hint, which may advertise sys.maxsize."""
    first, last, count = None, None, 0
    for frame in reader:
        count += 1
        if count > maximum:
            raise ValueError("Video exceeds registered frame cap")
        if frame.shape != (576, 1024, 3) or frame.dtype != np.uint8:
            raise ValueError("Foreign decoded frame shape/precision")
        if first is None:
            first = frame.copy()
        last = frame.copy()
    if count != maximum:
        raise ValueError("Video is missing registered frames")
    return first, last, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--raw-budget-bytes", type=int)
    args = parser.parse_args()
    p = json.loads((args.run / "protocol.json").read_text())
    limit = (
        p["raw_bytes"]
        if args.raw_budget_bytes is None
        else min(p["raw_bytes"], args.raw_budget_bytes)
    )
    credit = RawCredit(args.run, limit)
    path = args.run / "first-diagnostic.mp4"
    with imageio.get_reader(path, input_params=["-threads", "1"]) as reader:
        meta = reader.get_meta_data()
        first, last, count = decode_bounded(reader)
    if (
        count != 26
        or abs(meta["fps"] - 25) > 1e-9
        or abs(meta["duration"] - 1.04) > 0.001
    ):
        raise ValueError("Encoded video frame/timing dimensions differ")
    if np.array_equal(first, last):
        raise ValueError("No changing video state")
    render = json.loads(path.with_suffix(".json").read_text())
    if render["new_physics_steps"] != 0 or render["frames"] != 26:
        raise ValueError("Foreign render receipt")
    write_json(
        args.run / "video-validation.json",
        {
            "status": "passed",
            "frames": 26,
            "fps": 25,
            "encoded_duration_s": meta["duration"],
            "simulated_duration_s": 1.0,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "new_physics": 0,
            "classification": p["recording"]["classification"],
            "visual_inspection_pending": True,
        },
    )
    for name, frame in [("first-frame.png", first), ("last-frame.png", last)]:
        credit.check(args.run / name, frame.nbytes + 65536)
        imageio.imwrite(args.run / name, frame)


if __name__ == "__main__":
    main()
