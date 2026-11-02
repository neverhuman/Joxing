"""Opt-in closed-world graph release. Stdlib only before watcher readiness."""

import gc
import json
import time
import weakref
from pathlib import Path


def live_rss(bound, proc_root=Path("/proc")):
    folder = proc_root / str(bound["pid"])

    def checked_stat():
        fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        if (
            fields[19] != str(bound["start"])
            or fields[0] in ("Z", "X")
            or int(fields[6]) & 4
        ):
            raise RuntimeError("Release memory identity/live-state drift")
        return {
            "pid": bound["pid"],
            "start": fields[19],
            "state": fields[0],
            "flags": int(fields[6]),
        }

    before = checked_stat()
    fields = {
        k: v.strip()
        for line in (folder / "status").read_text().splitlines()
        if ":" in line
        for k, v in [line.split(":", 1)]
    }
    rss = fields.get("VmRSS", "").split()
    if len(rss) != 2 or rss[1] != "kB" or int(rss[0]) <= 0:
        raise RuntimeError("Release live RSS unavailable")
    after = checked_stat()
    return {"before": before, "after": after, "rss_bytes": int(rss[0]) * 1024}


def release_wrapper_class(
    base,
    receipt_path,
    *,
    state_hash,
    bound,
    read_memory=live_rss,
    ref_factory=weakref.ref,
    collect=gc.collect,
    clock=time.monotonic,
):
    """Drop only the closed inner graph; do not alter an evaluated native world."""
    path = Path(receipt_path)

    class ReleaseAfterClose(base):
        def __init__(self, env):
            super().__init__(env)
            self._release_state = "open"

        def _require_open(self):
            if self._release_state != "open":
                raise RuntimeError("Released world cannot be used")

        def reset(self, **kwargs):
            self._require_open()
            return self.env.reset(**kwargs)

        def step(self, action):
            self._require_open()
            return self.env.step(action)

        def snapshot(self):
            self._require_open()
            return self.env.unwrapped.snapshot()

        def _emit(self, phase, **values):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "phase": phase,
                            "identity": bound,
                            "monotonic_s": clock(),
                            **values,
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )

        def close(self):
            if self._release_state == "released":
                return
            self._require_open()
            self._release_state = "closing"
            try:
                core = self.env.unwrapped
                refs = {
                    "env": ref_factory(core),
                    "model": ref_factory(core.model),
                    "data": ref_factory(core.data),
                }
                initial_hash = state_hash(core.snapshot())
                memory_before = read_memory(bound)
                self._emit(
                    "before_close",
                    state_hash=initial_hash,
                    memory=memory_before,
                    muscles=int(core.model.nu),
                )
                del core
                self.env.close()
                closed_hash = state_hash(self.env.unwrapped.snapshot())
                self._emit("inner_closed", state_hash=closed_hash)
                if initial_hash != closed_hash:
                    raise ValueError("Native state changed during inner close")
                # No bound methods, native views or strong temporary core refs remain.
                self.env = None
                collected = collect()
                dead = {name: ref() is None for name, ref in refs.items()}
                memory_after = read_memory(bound)
                self._emit(
                    "graph_checked",
                    weakrefs_dead=dead,
                    collected_objects=collected,
                    memory=memory_after,
                    observed_rss_change_bytes=memory_after["rss_bytes"]
                    - memory_before["rss_bytes"],
                )
                if not all(dead.values()):
                    raise RuntimeError("Closed native graph still referenced")
                self._release_state = "released"
                self._emit("released")
            except BaseException as error:
                self._release_state = "failed"
                try:
                    self._emit("fault", error=type(error).__name__ + ": " + str(error))
                except OSError:
                    pass  # Preserve the originating fault; parent also records failure.
                raise

    return ReleaseAfterClose
