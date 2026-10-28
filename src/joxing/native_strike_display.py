"""Zero-step video of the first fixed native diagnostic, including zero points."""

import resource

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .boxing_evidence import array_sha
from .env import INTEGRATION
from .evidence_storage import read_arrays, read_json, sha_file
from .motor_storage import RawCredit, write_json
from .native_strike_qualification import RUN, controllers, environment


def points_at_tick(events, tick):
    eligible = [
        e
        for e in events
        if e["finalized_tick"] < tick
        or (e["reason"] == "bell" and e["finalized_tick"] == tick)
    ]
    return [
        sum(
            e["points"]
            for e in eligible
            if e["event_id"].split("/")[1].startswith(a + "_")
        )
        for a in "AB"
    ]


def render(root, p, source_sha, protocol_sha):
    a = read_json(root / "runs/v0.0.11/scored-contact-supervision-01/result.json")
    b = read_json(root / "runs/v0.0.11/native-strike-supervision-01/result.json")
    proof = read_json(root / RUN / "result.json")
    audited = read_json(root / RUN / "independent-audit.json")
    if (
        b["status"] not in ("running", "completed")
        or any(
            row["status"] != "completed"
            for row in b["stages"]
            if row["stage"] in ("proof", "audit")
        )
        or proof["status"] != "passed"
        or audited["status"] != "passed"
        or any(
            r["source_commit"] != source_sha or r["protocol_sha256"] != protocol_sha
            for r in (b, proof, audited)
        )
        or p["prior_postgate_upper_s"] + a["post_gate_wall_s"] + p["post_gate_wall_s"]
        > 900
    ):
        raise ValueError(
            "Native fixed display qualification/remaining time prerequisite"
        )
    out = root / RUN
    credit = RawCredit(out, p["raw_bytes"])
    tape = read_arrays(out / "source/tape.npz")
    trace = read_json(out / "source/trace.json")
    ticks, states = tape["video_ticks"], tape["video_states"]
    if ticks.tolist() != list(range(0, 3001, 40)) or states.shape[0] != 76:
        raise ValueError("First fixed native video extent")
    initial = read_json(out / "source/initial.json")
    if (
        states[0].tobytes() != np.array(initial["engine"], np.float64).tobytes()
        or not np.isfinite(states).all()
    ):
        raise ValueError("Native initial frame/state precision")
    target = out / "fixed-native-interaction.mp4"
    if target.exists():
        raise FileExistsError("Never overwrite a fixed source recording")
    credit.check(target, 2 * 1024**2 + 65536)
    env = environment(controllers(root, p))
    data = mujoco.MjData(env.model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 4.5, 115, -12
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    pelvis = [env.model.body(a + "_pelvis").id for a in "AB"]
    events = [e for row in trace["intervals"] for e in row["events"]]
    original_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024**2, original_limit[1]))
    try:
        with (
            mujoco.Renderer(env.model, height=448, width=800) as renderer,
            imageio.get_writer(
                target,
                fps=25,
                codec="libx264",
                quality=7,
                ffmpeg_params=["-threads", "1", "-maxrate", "2M", "-bufsize", "2M"],
            ) as writer,
        ):
            for tick, state in zip(ticks, states, strict=True):
                if (
                    tick
                    and array_sha(state)
                    != trace["intervals"][int(tick) - 1]["engine_sha256"]
                ):
                    raise ValueError(
                        "Native recorded frame is not the actual source state"
                    )
                mujoco.mj_setState(env.model, data, state, INTEGRATION)
                mujoco.mj_forward(env.model, data)  # Scratch visual data only.
                camera.lookat[:] = [*np.mean(data.xpos[pelvis, :2], axis=0), 0.9]
                renderer.update_scene(data, camera=camera, scene_option=options)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 800, 64), fill=(16, 24, 39))
                draw.text(
                    (12, 9),
                    "FROZEN MUSCLE CONTROLLERS - DIAGNOSTIC, NOT QUALIFIED BOXING",
                    fill="white",
                )
                draw.text(
                    (12, 31),
                    f"A: v0.0.21 | B: v0.0.19 | t={tick / 1000:.3f}s | Legal candidate points {points_at_tick(events, int(tick))}",
                    fill="white",
                )
                draw.rectangle((0, 418, 800, 448), fill=(16, 24, 39))
                draw.text(
                    (12, 426),
                    "700 muscles each | No assistance or new learning | Original rules and first fixed seed",
                    fill="white",
                )
                writer.append_data(np.asarray(frame))
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, original_limit)
    count, stills = 0, {}
    with imageio.get_reader(target, input_params=["-threads", "1"]) as reader:
        metadata = reader.get_meta_data()
        for frame in reader:
            if count >= 76 or frame.shape != (448, 800, 3):
                raise ValueError("Native decoded video bound")
            if count in (0, 38, 75):
                stills[count] = frame.copy()
            count += 1
    if (
        count != 76
        or abs(metadata["fps"] - 25) > 1e-9
        or abs(metadata["duration"] - 76 / 25) > 0.001
        or np.array_equal(stills[0], stills[75])
    ):
        raise ValueError("Native recording decode/clock/state change")
    for index, frame in stills.items():
        credit.check(out / f"frame-{index:03d}.png", frame.nbytes + 65536)
        imageio.imwrite(out / f"frame-{index:03d}.png", frame)
    result = {
        "source_commit": source_sha,
        "protocol_sha256": protocol_sha,
        "classification": "First fixed frozen muscle-controller diagnostic; not qualified boxing",
        "frames": 76,
        "fps": 25,
        "simulated_seconds": 3.0,
        "encoded_seconds": 76 / 25,
        "points": proof["rows"][0]["points"],
        "sha256": sha_file(target),
        "bytes": target.stat().st_size,
        "new_physics": 0,
        "selection": "fixed seed10001/A; whole recording retained regardless of score",
    }
    write_json(out / "display.json", result)
    return result
