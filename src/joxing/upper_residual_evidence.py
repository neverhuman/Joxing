"""Independent scalar oracle for the opt-in compact arm residual contract."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def descriptor_check(descriptor, *, source_root, checkpoint_groups):
    root = Path(source_root)
    names, arms, sources = [], set(), {}
    main = root / "MS-Human-700.xml"
    sources[main.name] = hashlib.sha256(main.read_bytes()).hexdigest()
    for node in ET.parse(main).getroot().findall("include"):
        relative = node.get("file", "")
        if not relative.startswith("Muscle/"):
            continue
        file = root / relative
        sources[relative] = hashlib.sha256(file.read_bytes()).hexdigest()
        current = [m.attrib["name"] for m in ET.parse(file).getroot().find("actuator")]
        names.extend(current)
        if relative in ("Muscle/Muscle_Arm_l.xml", "Muscle/Muscle_Arm_r.xml"):
            arms.update(current)
    selected = [i for i, n in enumerate(names) if n in arms]
    groups = descriptor["groups"]
    if (
        descriptor["variant"] != "native700-arm-file-residual-v1"
        or names != descriptor["actuator_names"]
        or len(names) != 700
        or len(arms) != 122
        or selected != descriptor["selected_ids"]
        or sources != descriptor["source_hashes"]
        or groups != checkpoint_groups
        or len(groups) != 138
        or sorted(i for g in groups for i in g) != list(range(700))
        or descriptor["amplitude"] != 0.1
    ):
        raise ValueError("Independent arm file/group/decoder binding")
    chosen = set(selected)
    active = [j for j, group in enumerate(groups) if chosen.intersection(group)]
    if (
        active != descriptor["active_groups"]
        or len(active) != descriptor["action_dimension"]
        or [i for i in range(700) if i not in chosen] != descriptor["protected_ids"]
        or [j for j in active if not set(groups[j]) <= chosen]
        != descriptor["mixed_groups"]
    ):
        raise ValueError("Independent compact arm action ownership")
    return {
        "muscles": 700,
        "selected": len(selected),
        "protected": 700 - len(selected),
        "action_dimension": len(active),
    }


def expected_command(teacher, residual, descriptor):
    """Per-muscle scalar f32 arithmetic; no candidate decoder/helper call."""
    if (
        teacher.dtype != np.float32
        or teacher.shape != (700,)
        or residual.dtype != np.float32
        or residual.shape != (descriptor["action_dimension"],)
    ):
        raise ValueError("Independent command oracle dtype/shape")
    if (
        not np.isfinite(teacher).all()
        or not np.isfinite(residual).all()
        or np.any(abs(teacher) > 1)
        or np.any(abs(residual) > 1)
    ):
        raise ValueError("Independent command oracle numeric bounds")
    ownership = {}
    chosen = set(descriptor["selected_ids"])
    for j, group_id in enumerate(descriptor["active_groups"]):
        for i in descriptor["groups"][group_id]:
            if i in chosen:
                ownership[i] = j
    output = teacher.copy()
    for i, j in ownership.items():
        step = np.float32(0.1) * residual[j]
        if step != 0:
            value = np.float32(teacher[i] + step)
            output[i] = np.float32(min(max(value, -1), 1))
    return output


def inspect_snapshot(
    snapshot, descriptor, teacher, residual, command, *, actor, tick, game_id
):
    """Check the saved record directly, without candidate restore/compose."""
    canonical = lambda x: hashlib.sha256(
        json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if (
        snapshot["schema"] != "joxing-held-upper-residual-v1"
        or snapshot["actor"] != actor
        or snapshot["game_id"] != game_id
        or snapshot["sampled_tick"] != tick
        or snapshot["physics_dt_s"] != 0.001
        or snapshot["hold_ticks"] != 20
        or snapshot["spec_sha256"] != canonical(descriptor)
        or snapshot["sha256"]
        != canonical({k: v for k, v in snapshot.items() if k != "sha256"})
    ):
        raise ValueError("Independent held record owner/clock/binding")
    for key, expected in (
        ("teacher", teacher),
        ("residual", residual),
        ("command", command),
    ):
        actual = np.asarray(snapshot[key], np.float32)
        if (
            actual.shape != expected.shape
            or actual.tobytes() != expected.tobytes()
            or hashlib.sha256(actual.tobytes()).hexdigest() != snapshot[key + "_sha256"]
        ):
            raise ValueError("Independent held record payload")
