"""Read release ZIP/JSON metadata without deserializing a model or normalizer."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

from joxing.provenance import manifest, write_json
from joxing.stance_train import file_hash

RELEASE_URL = (
    "https://github.com/LNSGroup/msgym/releases/download/Checkpoints/LocomotionFull.zip"
)
RELEASE_SHA256 = "2b2f0f1007d530db76463e7d3d5fb309b52c7b2a53649920ddfd9cf8a6880c57"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    assert file_hash(args.archive) == RELEASE_SHA256
    with zipfile.ZipFile(args.archive) as outer:
        members = [
            {
                "path": i.filename,
                "bytes": i.file_size,
                "compressed_bytes": i.compress_size,
            }
            for i in outer.infolist()
        ]
        config = json.loads(outer.read("LocomotionFull/locomotionFull.json"))
        normalizer = outer.read("LocomotionFull/checkpoint/best_env.zip")
        payload = outer.read("LocomotionFull/checkpoint/best_model.zip")
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        checkpoint_members = [
            {"path": i.filename, "bytes": i.file_size} for i in inner.infolist()
        ]
        data = json.loads(inner.read("data"))
        system_info = inner.read("system_info.txt").decode()
    plain = {
        k: v
        for k, v in data.items()
        if isinstance(v, (str, int, float, bool, type(None))) and len(str(v)) < 2000
    }
    groups = data["policy_kwargs"]["dynsyn"]
    assert sorted(m for group in groups for m in group) == list(range(700))
    schedule = min(
        max(0.0, data["dynsyn_k"] * (data["num_timesteps"] - data["dynsyn_a"])), 0.1
    )
    receipt = manifest() | {
        "script_sha256": file_hash(Path(__file__)),
        "url": RELEASE_URL,
        "release_asset_sha256": RELEASE_SHA256,
        "bytes": args.archive.stat().st_size,
        "scope": "Archive and plain JSON inspection only. No model, tensor archive, cloudpickle or normalizer is deserialized; no upstream motor or transfer performance is measured.",
        "members": members,
        "release_config": config,
        "checkpoint_members": checkpoint_members,
        "checkpoint_system_info": system_info,
        "checkpoint_plain_metadata": plain,
        "interface": {
            "observation_shape": data["observation_space"]["_shape"],
            "normalized_action_shape": data["action_space"]["_shape"],
            "synergy_groups": len(groups),
            "native_muscles_covered_once": 700,
            "policy_architecture": data["policy_kwargs"]["net_arch"],
            "saved_top_level_dynsyn_weight_amp": data["dynsyn_weight_amp"],
            "amplitude_from_saved_training_step_schedule": schedule,
        },
        "normalizer": {
            "bytes": len(normalizer),
            "sha256": hashlib.sha256(normalizer).hexdigest(),
            "header_hex": normalizer[:8].hex(),
            "is_zip": zipfile.is_zipfile(io.BytesIO(normalizer)),
            "loaded": False,
        },
        "limitations": [
            "Saved training counters describe the upstream checkpoint, not Joxing compute or a replicated result.",
            "The saved normalizer and exact trajectory/model/observation/action/decoder contract are necessary for evaluation.",
            "Joxing stance has a different observation and action interface; direct checkpoint loading is incompatible.",
            "Pinned SAC_DynSyn._setup_model and DynSynLayer source indicate that loading saved amplitude=None can disable adaptation. Reproduce and restore intended amplitude explicitly before interpreting a transfer test.",
        ],
    }
    write_json(args.out, receipt)
    print(json.dumps(receipt["interface"], indent=2))


if __name__ == "__main__":
    main()
