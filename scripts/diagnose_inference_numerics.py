"""Localize a retained inference fault without changing the frozen H0 gate."""

import argparse
import fcntl
import json
import os
import platform
import resource
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from joxing.native_transfer import load_controller
from joxing.provenance import digest, manifest, write_json
from joxing.stance_train import file_hash

PROTOCOL_SHA = "1e5ddb681c9d05b594ff79714a64e575386ce95e19016524d05fc5f32ee9f00e"


def error_summary(reference, actual):
    a, b = np.asarray(reference), np.asarray(actual)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Unmatched row/channel shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite trace output")
    # Float64 subtraction measures float32 differences without extra rounding.
    delta = abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "shape": list(a.shape),
        "exact": bool(np.array_equal(a, b)),
        "max_absolute_error": float(delta.max()),
        "max_relative_error": float((delta / np.maximum(abs(a), 1e-30)).max()),
        "changed_per_row": np.count_nonzero(delta, axis=1).tolist(),
        "channels_over_1e_6_per_row": np.count_nonzero(delta > 1e-6, axis=1).tolist(),
        "worst_row_channel": list(map(int, np.unravel_index(delta.argmax(), a.shape))),
    }


@contextmanager
def output_trace(actor):
    """Clone hook outputs immediately; downstream in-place operations cannot alter them."""
    handles, events, counts = [], [], {}

    def hook(name):
        def capture(module, inputs, output):
            if not hasattr(output, "detach") or output.ndim != 2:
                raise ValueError(f"Unknown module output: {name}")
            count = counts.get(name, 0)
            counts[name] = count + 1
            events.append((f"{name}#{count}", output.detach().cpu().numpy().copy()))
            # Returning None preserves PyTorch's original output object.

        return capture

    try:
        for name, module in actor.named_modules():
            handles.append(module.register_forward_hook(hook(name or "<actor>")))
        yield events
    finally:
        for handle in handles:
            handle.remove()


def predict_normalized(model, values, mode):
    if mode == "scalar":
        rows = [model.predict(row, deterministic=True) for row in values]
        if any(memory is not None for _, memory in rows):
            raise RuntimeError("Unexpected actor memory")
        return np.stack([action for action, _ in rows])
    if mode != "batched":
        raise ValueError("Unknown prediction mode")
    action, memory = model.predict(values, deterministic=True)
    if memory is not None:
        raise RuntimeError("Unexpected actor memory")
    return action


def trace_forward(model, values, mode):
    if mode == "batched":
        with output_trace(model.actor) as events:
            action = predict_normalized(model, values, mode)
        return action, events
    traces, actions = [], []
    for row in values:
        with output_trace(model.actor) as events:
            actions.append(predict_normalized(model, row[None], "scalar")[0])
        traces.append(events)
    names = [name for name, _ in traces[0]]
    if any([name for name, _ in trace] != names for trace in traces):
        raise RuntimeError("Scalar module call order changed")
    return np.stack(actions), [
        (name, np.concatenate([trace[i][1] for trace in traces]))
        for i, name in enumerate(names)
    ]


def compare_traces(scalar, batched):
    if [n for n, _ in scalar] != [n for n, _ in batched]:
        raise RuntimeError("Batch module call order differs")
    return [
        {"module_call": name, **error_summary(a, b)}
        for (name, a), (_, b) in zip(scalar, batched, strict=True)
    ]


def run(protocol_path, inputs, fault_path, out):
    p = json.loads(protocol_path.read_text())
    if digest(p) != PROTOCOL_SHA:
        raise ValueError("Registered numerical protocol changed")
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".numerics.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    rows = [{"variant": name, "status": "not_run"} for name in p["variants"]]
    report = manifest() | {
        "scope": p["scope"],
        "status": "running",
        "cases": rows,
        "protocol_sha256": digest(p),
        "script_sha256": file_hash(Path(__file__)),
        "physics_steps": 0,
        "boxing_games": 0,
        "backend_qualified": False,
    }
    write_json(out / "result.json", report)
    write_json(out / "protocol.json", p)
    active = None

    def budget():
        if time.perf_counter() - started > p["caps"]["wall_s"]:
            raise TimeoutError("Numerics wall cap")
        if (
            sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
            > p["caps"]["raw_bytes"]
        ):
            raise TimeoutError("Numerics raw cap")

    try:
        import torch

        from joxing.wave import canonical_actor_hash, check_resources

        check_resources(out, p["caps"], True)
        for path, key in [
            (fault_path, "fault_sha256"),
            (inputs / "model.zip", "model_sha256"),
            (inputs / "normalizer.pkl", "normalizer_sha256"),
        ]:
            if file_hash(path) != p[key]:
                raise ValueError(f"Changed diagnostic input: {key}")
        with np.load(fault_path, allow_pickle=False) as saved:
            raw, saved_scalar, saved_batch = (
                saved[k].copy() for k in ("raw", "reference", "actual")
            )
        if raw.shape != (2, 3601) or not np.isfinite(raw).all():
            raise ValueError("Unknown fault observations")
        torch.set_num_threads(1)
        model, normalizer = load_controller(inputs)
        model.policy.set_training_mode(False)
        actor_hash = canonical_actor_hash(model.actor)
        values = np.stack([normalizer.apply(row) for row in raw])
        if not np.array_equal(values, np.stack([normalizer.apply(row) for row in raw])):
            raise RuntimeError("Normalizer is not repeatable")
        report.update(
            torch=torch.__version__,
            torch_config=torch.__config__.show(),
            threads=torch.get_num_threads(),
            platform=platform.platform(),
            thread_environment={k: os.environ.get(k) for k in p["thread_environment"]},
            actor_hash=actor_hash,
            normalizer_stats_hash=normalizer.fingerprint,
            normalization_exact=True,
            module_graph=[
                {
                    "name": n or "<actor>",
                    "class": type(m).__module__ + "." + type(m).__name__,
                }
                for n, m in model.actor.named_modules()
            ],
        )
        required = {
            "features_extractor",
            "latent_pi",
            "mu",
            "dynsyn_layer",
            "dynsyn_layer.mu",
        }
        if not required.issubset(dict(model.actor.named_modules())):
            raise RuntimeError("Unknown loaded actor graph")
        default_mkldnn = torch.backends.mkldnn.enabled
        with torch.inference_mode():
            for row in rows:
                active = row
                budget()
                row["status"] = "running"
                write_json(out / "result.json", report)
                try:
                    torch.backends.mkldnn.enabled = (
                        default_mkldnn if row["variant"] == "default_float32" else False
                    )
                    row["mkldnn_enabled"] = torch.backends.mkldnn.enabled
                    before = {
                        mode: predict_normalized(model, values, mode)
                        for mode in ("scalar", "batched")
                    }
                    if row["variant"] == "default_float32" and not (
                        np.array_equal(before["scalar"], saved_scalar)
                        and np.array_equal(before["batched"], saved_batch)
                    ):
                        raise RuntimeError("Saved H0 fault did not reproduce exactly")
                    scalar_action, scalar = trace_forward(model, values, "scalar")
                    batch_action, batched = trace_forward(model, values, "batched")
                    for mode, hooked in [
                        ("scalar", scalar_action),
                        ("batched", batch_action),
                    ]:
                        after = predict_normalized(model, values, mode)
                        if not np.array_equal(
                            before[mode], hooked
                        ) or not np.array_equal(before[mode], after):
                            raise RuntimeError("Hooks or baseline changed output")
                    comparisons = compare_traces(scalar, batched)
                    arrays = {
                        f"{mode}_{i}": a
                        for mode, trace in [("scalar", scalar), ("batched", batched)]
                        for i, (_, a) in enumerate(trace)
                    }
                    arrays.update(
                        normalized=values,
                        scalar_action=scalar_action,
                        batched_action=batch_action,
                    )
                    target = out / (row["variant"] + ".npz")
                    np.savez_compressed(target, **arrays)
                    normalizer.check()
                    if canonical_actor_hash(model.actor) != actor_hash:
                        raise RuntimeError("Actor tensors changed")
                    summary = error_summary(scalar_action, batch_action)
                    row.update(
                        status="completed",
                        modules=comparisons,
                        first_divergent_module=next(
                            (r["module_call"] for r in comparisons if not r["exact"]),
                            None,
                        ),
                        action_error=summary,
                        old_h0_gate_met=summary["max_absolute_error"]
                        <= p["original_h0_action_max_error"],
                        hooks_output_preserving=True,
                        unchanged_actor_normalizer=True,
                        arrays_sha256=file_hash(target),
                    )
                    budget()
                finally:
                    torch.backends.mkldnn.enabled = default_mkldnn
                write_json(out / "result.json", report)
        report["status"] = "completed"
    except Exception as error:  # noqa: BLE001 -- retain failures and every planned row
        if active is not None:
            active.update(
                status="incomplete" if isinstance(error, TimeoutError) else "invalid",
                reason=str(error),
            )
        report.update(
            status="halted", error=str(error), traceback=traceback.format_exc()
        )
    finally:
        for row in rows:
            if row["status"] == "not_run":
                row["reason"] = "halted_before_case"
        report.update(
            wall_s=time.perf_counter() - started,
            max_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            rss_units="KiB on Linux, bytes on macOS",
        )
        write_json(out / "result.json", report)
        lease.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for flag in ("protocol", "inputs", "fault", "out"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    result = run(args.protocol, args.inputs, args.fault, args.out)
    print(
        json.dumps({k: result.get(k) for k in ("status", "wall_s", "error")}, indent=2)
    )
    if result["status"] != "completed":
        raise SystemExit(1)
