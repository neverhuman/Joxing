"""Zero-physics display of the first audited trial from an INCOMPLETE proof.

This never changes the failed proof or grants the missing replay/motor gate.
"""

import argparse
import json
import resource
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from joxing.env import INTEGRATION
from joxing.evidence_storage import read_arrays, read_json
from joxing.motor_storage import RawCredit, write_json
from joxing.provenance import digest
from joxing.shared_motor_audit import byte_sha
from joxing.shared_motor_proof import RUN, file_sha, read_packet
from joxing.shared_motor_saved_audit import require
from joxing.shared_motor_task import build_motor_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--proof-root", type=Path, required=True)
    parser.add_argument("--display-protocol", type=Path, required=True)
    parser.add_argument("--display-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.proof_root.resolve()
    display = json.loads(args.display_protocol.read_text())
    require(digest(display) == args.display_sha256, "Frozen display protocol")
    require(
        file_sha(Path(__file__)) == display["renderer_sha256"],
        "Frozen display renderer",
    )
    p = read_packet(root, args.protocol_sha256, args.source_sha)
    run = root / RUN
    audited = json.loads((run / "independent-audit.json").read_text())
    result = json.loads((run / "result.json").read_text())
    validate_partial_display(result, audited)
    for name, expected in display["inputs"].items():
        path = root / name
        require(
            path.stat().st_size == expected["bytes"]
            and file_sha(path) == expected["sha256"],
            "Display input drift: " + name,
        )
    require(
        args.out.resolve() != run.resolve(), "Display must have a separate output owner"
    )
    args.out.mkdir(parents=True, exist_ok=False)
    scene = build_motor_scene()
    folder = run / p["recording_source"]
    trace, tape = read_json(folder / "trace.json"), read_arrays(folder / "tape.npz")
    if trace["contract"]["scene_hash"] != scene.report["scene_hash"]:
        raise ValueError("Render scene drift")
    target = args.out / "first-pretrained-trial.mp4"
    if target.exists():
        raise FileExistsError("Never overwrite the first trial clip")
    credit = RawCredit(args.out, display["raw_bytes"])
    credit.check(target, 4 * 1024**2 + 65536)
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024**2, 4 * 1024**2))
    data = mujoco.MjData(scene.model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 5.5, 115, -12
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    pelvis = [scene.model.body(a + "_pelvis").id for a in ("A", "B")]
    with (
        mujoco.Renderer(scene.model, height=576, width=1024) as renderer,
        imageio.get_writer(
            target,
            fps=25,
            codec="libx264",
            quality=8,
            ffmpeg_params=["-threads", "1", "-maxrate", "3M", "-bufsize", "3M"],
        ) as writer,
    ):
        for tick, state in zip(tape["video_ticks"], tape["video_states"], strict=True):
            if tick and byte_sha(state) != trace["state_sha256"][int(tick) - 1]:
                raise ValueError("Recorded frame/engine drift")
            mujoco.mj_setState(scene.model, data, state, INTEGRATION)
            mujoco.mj_forward(scene.model, data)  # Scratch visual data only.
            camera.lookat[:] = [*np.mean(data.xpos[pelvis, :2], axis=0), 0.9]
            renderer.update_scene(data, camera=camera, scene_option=option)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 1024, 75), fill=(16, 24, 39))
            draw.text(
                (14, 10),
                "FROZEN PRETRAINED LOCOMOTION - INCOMPLETE FOUR-SEED PROOF",
                fill="white",
            )
            draw.text(
                (14, 31),
                f"First registered trial 71000 | t={tick / 1000:.3f}s | 700 muscles per athlete | No training updates",
                fill="white",
            )
            draw.text(
                (14, 52),
                "Audited first trial only; full replay/motor gate NOT passed; boxing points unavailable",
                fill="white",
            )
            draw.rectangle((0, 530, 1024, 576), fill=(16, 24, 39))
            for x, a, color in [(14, "A", "#ff8576"), (520, "B", "#83baff")]:
                physical, tracking = (
                    trace["physical_failure"][a],
                    trace["tracking_failure"][a],
                )
                status = (
                    physical["reason"]
                    if physical and tick >= physical["tick"]
                    else "no physical latch yet"
                )
                if tracking and tick >= tracking["tick"]:
                    status += " | tracking failed"
                draw.text((x, 540), a + ": " + status, fill=color)
            writer.append_data(np.asarray(frame))
    write_json(
        args.out / "first-pretrained-trial.json",
        {
            "source_commit": args.source_sha,
            "protocol_sha256": args.protocol_sha256,
            "sha256": file_sha(target),
            "bytes": target.stat().st_size,
            "frames": len(tape["video_ticks"]),
            "fps": 25,
            "new_render_physics_steps": 0,
            "seed": 71000,
            "selection": "first fixed trial, complete retained horizon",
            "short_motor_gate_passed": False,
            "full_proof_passed": False,
            "display_protocol_sha256": args.display_sha256,
        },
    )

    # Decode every frame; no allocation from a reader's untrusted length_hint.
    with imageio.get_reader(target, input_params=["-threads", "1"]) as reader:
        meta = reader.get_meta_data()
        first, last, count = None, None, 0
        for frame in reader:
            count += 1
            require(
                count <= 89
                and frame.shape == (576, 1024, 3)
                and frame.dtype == np.uint8,
                "Partial display decoded frame bound",
            )
            if first is None:
                first = frame.copy()
            last = frame.copy()
    require(
        count == 89
        and abs(meta["fps"] - 25) < 1e-9
        and abs(meta["duration"] - 89 / 25) < 0.001,
        "Partial display complete clock",
    )
    require(not np.array_equal(first, last), "Partial display changing state")
    for name, frame in [("first-frame.png", first), ("last-frame.png", last)]:
        credit.check(args.out / name, frame.nbytes + 65536)
        imageio.imwrite(args.out / name, frame)
    for name, expected in display["inputs"].items():
        require(file_sha(root / name) == expected["sha256"], "Display input changed")
    write_json(
        args.out / "video-validation.json",
        {
            "status": "passed",
            "frames": count,
            "fps": 25,
            "encoded_duration_s": meta["duration"],
            "simulated_duration_s": 3.52,
            "sha256": file_sha(target),
            "bytes": target.stat().st_size,
            "new_physics": 0,
            "visual_inspection_pending": True,
            "full_proof_passed": False,
            "short_motor_gate_passed": False,
            "scope": "First audited fixed source only; incomplete four-world/replay proof",
        },
    )


def validate_partial_display(result, audit):
    require(
        result["status"] == "halted" and len(result["rows"]) == 5,
        "Partial proof denominator/status",
    )
    first = result["rows"][0]
    require(
        first["id"] == "source-71000"
        and first["seed"] == 71000
        and first["status"] == "completed"
        and first["physics_attempted"] == 3520,
        "Only first registered completed source may be displayed",
    )
    require(
        audit["passed"] is False and audit["short_motor_gate_passed"] is False,
        "Do not relabel incomplete qualification",
    )
    finding = audit["findings"][0]
    require(
        finding["id"] == first["id"]
        and finding["passed"] is True
        and finding["policy_inference_audited"] is True
        and finding["measured_intervals"] == 3520
        and finding["video_frames"] == 89,
        "First source independent measurement/policy audit",
    )
    require(
        audit["new_audit_physics"] == 0 and audit["exact_replay"]["passed"] is False,
        "No hidden replay or motor promotion",
    )


if __name__ == "__main__":
    main()
