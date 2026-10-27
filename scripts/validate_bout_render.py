"""Decode the preregistered diagnostic MP4, with zero new physics."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from joxing.motor_storage import RawCredit, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads((args.run / "protocol.json").read_text())
    credit = RawCredit(args.run, p["raw_bytes"])
    path = args.run / "first-diagnostic.mp4"
    with imageio.get_reader(path) as reader:
        meta = reader.get_meta_data()
        frames = list(reader)
    if (
        len(frames) != 26
        or any(f.shape != (576, 1024, 3) for f in frames)
        or abs(meta["fps"] - 25) > 1e-9
        or abs(meta["duration"] - 1.04) > 0.001
    ):
        raise ValueError("Encoded video frame/timing dimensions differ")
    if np.array_equal(frames[0], frames[-1]):
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
    for name, frame in [("first-frame.png", frames[0]), ("last-frame.png", frames[-1])]:
        credit.check(args.run / name, frame.nbytes + 65536)
        imageio.imwrite(args.run / name, frame)


if __name__ == "__main__":
    main()
