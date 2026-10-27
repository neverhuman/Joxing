"""Bounded decode of every frame in the first fixed pretrained trial."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from validate_bout_render import decode_bounded

from joxing.motor_storage import RawCredit, write_json
from joxing.shared_motor_proof import RUN, file_sha, read_packet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    p = read_packet(root, args.protocol_sha256, args.source_sha)
    out = root / RUN
    credit = RawCredit(out, p["raw_bytes"] - 512 * 1024)
    video = out / "first-pretrained-trial.mp4"
    receipt = json.loads(video.with_suffix(".json").read_text())
    if (
        receipt["frames"] != 89
        or receipt["new_render_physics_steps"] != 0
        or file_sha(video) != receipt["sha256"]
    ):
        raise ValueError("Pretrained render receipt/digest/frame count")
    with imageio.get_reader(video, input_params=["-threads", "1"]) as reader:
        meta = reader.get_meta_data()
        first, last, count = decode_bounded(reader, maximum=89)
    if (
        abs(meta["fps"] - 25) > 1e-9
        or abs(meta["duration"] - 89 / 25) > 0.001
        or np.array_equal(first, last)
    ):
        raise ValueError("Pretrained video clock or unchanging state")
    for name, frame in [("first-frame.png", first), ("last-frame.png", last)]:
        credit.check(out / name, frame.nbytes + 65536)
        imageio.imwrite(out / name, frame)
    write_json(
        out / "video-validation.json",
        {
            "status": "passed",
            "frames": count,
            "fps": 25,
            "encoded_duration_s": meta["duration"],
            "simulated_duration_s": 3.52,
            "sha256": file_sha(video),
            "bytes": video.stat().st_size,
            "new_physics": 0,
            "visual_inspection_pending": True,
            "classification": "Frozen pretrained locomotion transfer diagnostic",
        },
    )


if __name__ == "__main__":
    main()
