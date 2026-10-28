"""Bounded recovery of complete ZIP-local members and zero-step partial display.

Does not recover missing force values or qualify the failed native/replay proof.
"""

import argparse
import ast
import hashlib
import json
import lzma
import math
import resource
import struct
import time
import zlib
from pathlib import Path

import numpy as np

ORDER = [
    "commands.npy",
    "observations.npy",
    "decision_engines.npy",
    "contact_offsets.npy",
    "contact_geoms.npy",
    "contacts.npy",
    "video_ticks.npy",
    "video_states.npy",
]


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def read_complete_prefix(path):
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError("Partial ZIP input bound")
    arrays = {}
    receipts = []
    total = 0
    incomplete = None
    with path.open("rb") as f:
        for expected_name in ORDER:
            header = f.read(30)
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                raise ValueError("Missing local ZIP header")
            h = struct.unpack("<4s5H3I2H", header)
            if h[-2] > 128 or h[-1] > 64:
                raise ValueError("Local header metadata bound")
            name = f.read(h[-2]).decode("ascii")
            f.read(h[-1])
            if name != expected_name or h[2] & ~(2 | 0x800) or h[3] != 14:
                raise ValueError("Unexpected ZIP member or flags")
            if h[7] == h[8] == 0:
                incomplete = name
                break
            if not 0 < h[7] <= 8 * 1024**2 or not 0 < h[8] <= 8 * 1024**2:
                raise ValueError("ZIP member size bound")
            total += h[8]
            if total > 12 * 1024**2:
                raise ValueError("ZIP decoded total bound")
            compressed = f.read(h[7])
            if (
                len(compressed) != h[7]
                or len(compressed) < 9
                or compressed[:2] != b"\x09\x04"
                or int.from_bytes(compressed[2:4], "little") != 5
            ):
                raise ValueError("Incomplete ZIP LZMA payload")
            prop = compressed[4]
            dictionary = int.from_bytes(compressed[5:9], "little")
            lc = prop % 9
            lp = (prop // 9) % 5
            pb = prop // 45
            if prop >= 225 or lc + lp > 4 or not 4096 <= dictionary <= 8 * 1024**2:
                raise ValueError("ZIP LZMA filter bound")
            decoder = lzma.LZMADecompressor(
                format=lzma.FORMAT_RAW,
                filters=[
                    {
                        "id": lzma.FILTER_LZMA1,
                        "dict_size": dictionary,
                        "lc": lc,
                        "lp": lp,
                        "pb": pb,
                    }
                ],
            )
            raw = decoder.decompress(compressed[9:], max_length=h[8] + 1)
            if (
                len(raw) != h[8]
                or not decoder.eof
                or decoder.unused_data
                or zlib.crc32(raw) & 0xFFFFFFFF != h[6]
            ):
                raise ValueError("ZIP CRC/stream extent differs")
            if raw[:6] != b"\x93NUMPY" or raw[6:8] not in (b"\x01\x00", b"\x02\x00"):
                raise ValueError("NPY member schema")
            size = 2 if raw[6:8] == b"\x01\x00" else 4
            length = int.from_bytes(raw[8 : 8 + size], "little")
            if length > 4096:
                raise ValueError("NPY header bound")
            meta = ast.literal_eval(raw[8 + size : 8 + size + length].decode("latin1"))
            if (
                set(meta) != {"descr", "fortran_order", "shape"}
                or meta["fortran_order"]
                or meta["descr"] not in ("<f4", "<f8", "<i8")
                or not isinstance(meta["shape"], tuple)
                or not 1 <= len(meta["shape"]) <= 3
                or any(type(n) is not int or n < 0 for n in meta["shape"])
            ):
                raise ValueError("Unsupported NPY layout")
            payload = raw[8 + size + length :]
            dtype = np.dtype(meta["descr"])
            if math.prod(meta["shape"]) * dtype.itemsize != len(payload):
                raise ValueError("NPY shape/byte extent")
            array = np.frombuffer(payload, dtype=dtype).reshape(meta["shape"]).copy()
            arrays[name[:-4]] = array
            receipts.append(
                {
                    "member": name,
                    "crc32": h[6],
                    "compressed_bytes": h[7],
                    "dtype": meta["descr"],
                    "shape": list(meta["shape"]),
                    "payload_bytes": len(payload),
                    "payload_sha256": array_sha(array),
                }
            )
    return arrays, {
        "complete": False,
        "members": receipts,
        "first_incomplete_member": incomplete,
        "missing_fields_not_padded": True,
    }


def main():
    began = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.manifest.read_text())
    if (
        digest(p) != args.manifest_sha256
        or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        != p["renderer_sha256"]
    ):
        raise ValueError("Partial display source/protocol identity")
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw

    from joxing.env import INTEGRATION
    from joxing.evidence_storage import read_json
    from joxing.motor_storage import RawCredit, write_json
    from joxing.native_strike_display import points_at_tick
    from joxing.provenance import ROOT
    from joxing.scene import build_scene
    from joxing.score_diagnostic import audit_saved_trace

    if str(ROOT) != p["core_root"]:
        raise ValueError("Qualified core import root differs")
    for n, d in p["inputs"].items():
        f = Path(n)
        if (
            f.stat().st_size != d["bytes"]
            or hashlib.sha256(f.read_bytes()).hexdigest() != d["sha256"]
        ):
            raise ValueError("Partial display input bytes differ")
    source = Path(p["source_folder"])
    initial, final, trace, contract = [
        read_json(source / (n + ".json"))
        for n in ("initial", "final", "trace", "contract")
    ]
    for state in (initial, final):
        if digest({k: v for k, v in state.items() if k != "sha256"}) != state[
            "sha256"
        ] or state["contract_hash"] != digest(contract):
            raise ValueError("Actual source snapshot identity")
    journal = [json.loads(s) for s in Path(p["journal"]).read_text().splitlines()]
    if journal != [
        {"row": "fixed-source-10001-A", "phase": phase, "tick": tick}
        for tick in range(3000)
        for phase in ("entered", "completed")
    ]:
        raise ValueError("Actual source engine denominator")
    arrays, prefix = read_complete_prefix(source / "tape.npz.tmp")
    if (
        set(arrays)
        != {
            "commands",
            "observations",
            "decision_engines",
            "contact_offsets",
            "contact_geoms",
        }
        or prefix["first_incomplete_member"] != "contacts.npy"
    ):
        raise ValueError("Actual incomplete source membership")
    if (
        arrays["commands"].shape != (150, 2, 700)
        or arrays["observations"].shape != (150, 2, 1970)
        or arrays["decision_engines"].shape != (150, 4555)
        or arrays["contact_offsets"].shape != (3001,)
        or arrays["contact_geoms"].shape != (int(arrays["contact_offsets"][-1]), 2)
        or np.any(np.diff(arrays["contact_offsets"]) < 0)
    ):
        raise ValueError("Actual source typed extent")
    if any(
        not np.isfinite(arrays[n]).all()
        for n in ("commands", "observations", "decision_engines")
    ) or np.any(np.abs(arrays["commands"]) > 1):
        raise ValueError("Actual source nonfinite or command bound")
    if [d["tick"] for d in trace["decisions"]] != list(range(0, 3000, 20)) or len(
        trace["intervals"]
    ) != 3000:
        raise ValueError("Actual source decision/physical clock")
    score = audit_saved_trace(
        initial["application"]["referee"],
        final["application"]["referee"],
        trace["intervals"],
        3000,
        maximum_intervals=3000,
    )
    events = [e for row in trace["intervals"] for e in row["events"]]
    from collections import Counter

    failed = Counter()
    onsets = []
    head_ticks = {a: set() for a in "AB"}
    for row in score["explanations"]:
        for sample in row["samples"]:
            t = sample["measured"]
            if (
                t["normal_force_N"] > 0
                and t["zone"] in ("head", "torso")
                and t["approach_mps"] >= 0.5
                and t["active_speed_mps"] >= 0.5
            ):
                head_ticks[row["glove"][0]].add(row["tick"])
            if sample["first_zone_onset"]:
                onsets.append({"glove": row["glove"], "tick": row["tick"], **sample})
                failed.update(
                    k
                    for k in ("approach", "active_speed", "face_cos", "striking_face")
                    if not sample["instantaneous_predicates"][k]
                )
    summary = {
        "status": "passed",
        "scope": "Independent score-only reconstruction of actual saved staged inputs; missing full signed contact tape and physical continuation remain unqualified",
        "intervals": 3000,
        "events": events,
        "candidate_legal_points": [
            sum(
                e["points"]
                for e in events
                if e["event_id"].split("/")[1].startswith(a + "_")
            )
            for a in "AB"
        ],
        "effective_sample_rejections": score["rejections"],
        "first_direct_onsets": onsets,
        "first_onset_failed_predicates": dict(failed),
        "staged_head_torso_speed_ticks": {a: len(v) for a, v in head_ticks.items()},
        "new_physics": 0,
    }
    ticks = list(range(0, 3001, 40))
    states = []
    for tick in ticks:
        state = (
            arrays["decision_engines"][tick // 20]
            if tick < 3000
            else np.array(final["engine"], np.float64)
        )
        expected = (
            array_sha(np.array(initial["engine"], np.float64))
            if tick == 0
            else trace["intervals"][tick - 1]["engine_sha256"]
        )
        if array_sha(state) != expected:
            raise ValueError("Recovered actual video engine hash differs")
        states.append(state)
    args.out.mkdir(parents=True, exist_ok=False)
    credit = RawCredit(args.out, p["raw_bytes"])
    write_json(args.out / "prefix-audit.json", prefix)
    write_json(args.out / "score-audit.json", summary)
    scene = build_scene()
    if (
        scene.report["scene_hash"] != contract["scene_hash"]
        or scene.model.nu != 1400
        or scene.model.na != 1400
    ):
        raise ValueError("Actual intact native model identity")
    data = mujoco.MjData(scene.model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 4.5, 115, -12
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    pelvis = [scene.model.body(a + "_pelvis").id for a in "AB"]
    for state in arrays["decision_engines"]:
        mujoco.mj_setState(scene.model, data, state, INTEGRATION)
        if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            raise ValueError("Actual decision contains external assistance")
    temporary = args.out / "first-native-partial.tmp.mp4"
    credit.check(temporary, p["video_max_bytes"] + 65536)
    oldlimit = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (p["video_max_bytes"], oldlimit[1]))
    try:
        with (
            mujoco.Renderer(scene.model, height=360, width=640) as renderer,
            imageio.get_writer(
                temporary,
                fps=25,
                macro_block_size=1,
                codec="libx264",
                quality=7,
                ffmpeg_params=["-threads", "1", "-maxrate", "1M", "-bufsize", "1M"],
            ) as writer,
        ):
            for tick, state in zip(ticks, states, strict=True):
                if (
                    time.monotonic() - began > p["wall_limit_s"]
                    or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                    > p["maximum_rss_bytes"]
                ):
                    raise TimeoutError("Partial display first limit")
                mujoco.mj_setState(scene.model, data, state, INTEGRATION)
                mujoco.mj_forward(scene.model, data)
                camera.lookat[:] = [*np.mean(data.xpos[pelvis, :2], axis=0), 0.9]
                renderer.update_scene(data, camera=camera, scene_option=options)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 54), fill=(16, 24, 39))
                draw.text(
                    (10, 8),
                    "FROZEN MUSCLE CONTROLLERS - PARTIAL NATIVE PROOF",
                    fill="white",
                )
                draw.text(
                    (10, 29),
                    f"A v21 / B v19 | t={tick / 1000:.3f}s | Candidate legal points {points_at_tick(events, tick)}",
                    fill="white",
                )
                draw.rectangle((0, 334, 640, 360), fill=(16, 24, 39))
                draw.text(
                    (10, 342),
                    "700 muscles each | Force tape / replay incomplete | No new learning",
                    fill="white",
                )
                writer.append_data(np.asarray(frame))
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, oldlimit)
    stills = {}
    count = 0
    with imageio.get_reader(temporary, input_params=["-threads", "1"]) as reader:
        metadata = reader.get_meta_data()
        for frame in reader:
            if count >= 76 or frame.shape != (360, 640, 3):
                raise ValueError("Partial native decoded frame bound")
            if count in (0, 38, 75):
                stills[count] = frame.copy()
            count += 1
    if (
        count != 76
        or abs(metadata["fps"] - 25) > 1e-9
        or abs(metadata["duration"] - 3.04) > 0.001
        or np.array_equal(stills[0], stills[75])
    ):
        raise ValueError("Partial native video extent/state")
    target = args.out / "first-native-partial.mp4"
    temporary.rename(target)
    for index, frame in stills.items():
        credit.check(args.out / f"frame-{index:03d}.png", frame.nbytes + 65536)
        imageio.imwrite(args.out / f"frame-{index:03d}.png", frame)
    result = {
        "schema": "joxing-native-source-partial-display-v1",
        "status": "passed",
        "source_commit": p["source_commit"],
        "manifest_sha256": args.manifest_sha256,
        "frames": 76,
        "fps": 25,
        "simulated_s": 3.0,
        "encoded_s": 3.04,
        "candidate_points": summary["candidate_legal_points"],
        "source_native_steps": 3000,
        "new_display_physics": 0,
        "full_native_proof_passed": False,
        "missing": [
            "complete signed contacts array",
            "original video array",
            "actual mid continuation",
        ],
        "display_states": "75actual decision engines plus actual final engine; all76complete engine hashes match actual source",
        "video_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "video_bytes": target.stat().st_size,
        "wall_s": time.monotonic() - began,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    write_json(args.out / "result.json", result)


if __name__ == "__main__":
    main()
