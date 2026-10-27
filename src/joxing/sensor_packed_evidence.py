"""Explicit lossless save-only variant; frozen SensorTape.choose/capture inherited.

Payload construction mirrors the frozen D0/D1 writers. Tests compare every
logical file and dtype/shape/byte against those writers, including held replay.
"""

from pathlib import Path

import numpy as np

from .boxing_evidence import CONTACT_COLUMNS
from .evidence_storage import save_npz, write_json
from .sensor_evidence import BaseTapeView, SensorTape


class PackedSensorTape(SensorTape):
    def _save_base(self, env, folder, *, quarantined=False):
        folder = Path(folder)
        final = None if quarantined else env.snapshot()
        write_json(folder / "initial.json", self.initial)
        if final is not None:
            write_json(folder / "final.json", final)
        write_json(
            folder / "trace.json",
            {
                "schema": "joxing-shared-evidence-v1",
                "decisions": self.decisions,
                "rngs_before": self.rngs,
                "intervals": self.intervals,
                "state_sha256": self.hashes,
                "contact_columns": CONTACT_COLUMNS,
                "transition": {"invalid_reason": env.invalid_reason}
                if quarantined
                else {k: v for k, v in env.transition().items() if k != "observations"},
            },
        )
        width = len(env.state())
        save_npz(
            folder / "tape.npz",
            applied=np.asarray(self.applied, np.float32).reshape(-1, 2, 700),
            excitation=np.asarray(self.excitations, np.float64).reshape(-1, 2, 700),
            packets=np.asarray(self.packets, np.int64).reshape(-1, 2, 2),
            contact_offsets=np.asarray(self.offsets, np.int64),
            contact_geoms=np.asarray(self.geom_ids, np.int64).reshape(-1, 2),
            contacts=np.asarray(self.contacts, np.float64).reshape(-1, 18),
            decision_states=np.asarray(self.decision_states, np.float64).reshape(
                -1, width
            ),
            observations=np.asarray(self.observations, np.float32).reshape(-1, 2, 1970),
            histories=np.asarray(self.histories, np.float32).reshape(-1, 2, 8, 1970),
            history_ticks=np.asarray(self.history_ticks, np.int64).reshape(-1, 2, 8),
            history_received=np.asarray(self.history_received, np.int64).reshape(
                -1, 2, 8
            ),
            history_valid=np.asarray(self.history_valid, bool).reshape(-1, 2, 8),
            chosen=np.asarray(self.chosen, np.float32).reshape(-1, 2, 700),
            decision_ticks=np.asarray(self.decision_ticks, np.int64),
        )
        return {
            "physics": len(self.intervals),
            "final_sha256": None if final is None else final["sha256"],
            "positive_glove_touches": sum(
                t["normal_force_N"] > 0
                for row in self.intervals
                for g in row["gloves"].values()
                for t in g["touches"]
            ),
        }

    def save(self, env, folder, *, quarantined=False):
        from pathlib import Path

        folder = Path(folder)
        result = self._save_base(BaseTapeView(env), folder, quarantined=quarantined)
        write_json(folder / "sensor-initial.json", self.sensor_initial)
        if not quarantined:
            write_json(folder / "sensor-final.json", env.snapshot())
        write_json(
            folder / "sensor-trace.json",
            {
                "schema": "joxing-sensor-evidence-v1",
                "frames": self.frames,
                "rngs_after": self.after_rngs,
                "events": self.sensor_events,
                "sample_points": [s["points"] for s in self.samples],
                "initial_events_in_snapshot": True,
            },
        )
        width = len(env.state())
        save_npz(
            folder / "sensor-tape.npz",
            sample_ticks=np.asarray([s["tick"] for s in self.samples], np.int64),
            sample_engines=np.asarray(
                [s["engine"] for s in self.samples], np.float64
            ).reshape(-1, width),
            sample_commands=np.asarray(
                [s["commands"] for s in self.samples], np.float32
            ).reshape(-1, 2, 700),
            sample_values=np.asarray(
                [s["values"] for s in self.samples], np.float32
            ).reshape(-1, 2, 1970),
            history_sample_ages=np.asarray(self.sample_ages, np.int64).reshape(
                -1, 2, 8
            ),
            history_receipt_ages=np.asarray(self.receipt_ages, np.int64).reshape(
                -1, 2, 8
            ),
        )
        return result | {"sensor_samples": len(self.samples)}
