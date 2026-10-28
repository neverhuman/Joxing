"""Rigid physical adapter control, explicitly not an assisted athlete result."""

from dataclasses import asdict
from types import SimpleNamespace

import mujoco
import numpy as np

from .physics import step_contacts
from .referee import MatchReferee
from .score_diagnostic import audit_saved_trace


def run_fixture(zone, reverse=False, *, step_receipt=None):
    if zone not in ("head", "torso") or type(reverse) is not bool:
        raise ValueError("Unknown physical scoring control")
    # The moving sphere's LOCAL -Y face points WORLD +X, as in the adapter.
    # Only reset-time velocity is specified; no pose writes/applied forces later.
    moving = '<body name="moving" pos="-.3 0 0" quat=".7071067811865476 0 0 .7071067811865476"><freejoint/><geom name="A_glove_r" type="sphere" size=".062" mass="1"/></body>'
    target = '<geom name="B_target" type="sphere" size=".1"/>'
    xml = (
        """<mujoco><option timestep=".001" gravity="0 0 0" integrator="implicitfast"/>
    <default><geom condim="1" friction="0 0 0" solref=".015 1"/></default>
    <worldbody><body name="A_thoracic12" pos="-.4 0 0"/>
    <body name="B_thoracic12" pos=".2 0 0"/>
    <geom name="A_glove_l" type="sphere" pos="0 8 0" size=".062" contype="0" conaffinity="0"/>
    <geom name="B_glove_l" type="sphere" pos="0 9 0" size=".062" contype="0" conaffinity="0"/>
    <geom name="B_glove_r" type="sphere" pos="0 10 0" size=".062" contype="0" conaffinity="0"/>
    """
        + (moving + target if reverse else target + moving)
        + "</worldbody></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    registry = [
        {"name": n, "id": model.geom(n).id, "actor": n[0], "role": "glove"}
        for n in ("A_glove_l", "A_glove_r", "B_glove_l", "B_glove_r")
    ]
    registry.append(
        {
            "name": "B_target",
            "id": model.geom("B_target").id,
            "actor": "B",
            "role": zone,
        }
    )
    scene = SimpleNamespace(model=model, report={"physical_geoms": registry})
    referee = MatchReferee(scene, f"rigid-adapter-{zone}-{reverse}")
    initial = referee.snapshot()
    data.qvel[0] = 2.0
    intervals = []
    positive = 0
    for tick in range(250):
        before = data.qvel.copy()
        if step_receipt is not None:
            step_receipt("entered", tick)
        contacts = step_contacts(model, data)
        if step_receipt is not None:
            step_receipt("completed", tick)
        staged = referee.stage(data, contacts, before)
        events = referee.commit(staged, tick, bell=tick == 249)
        intervals.append(
            {
                "tick": tick,
                "gloves": {name: asdict(g) for name, g in staged.items()},
                "events": events,
            }
        )
        positive += sum(
            t.normal_force_N > 0 for g in staged.values() for t in g.touches
        )
        if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
            raise ValueError("Unexpected diagnostic external force")
    audited = audit_saved_trace(initial, referee.snapshot(), intervals, 250)
    expected = 2 if zone == "head" else 1
    if referee.points != (expected, 0) or not positive:
        raise ValueError(
            f"Physical adapter positive control did not score: {referee.points}"
        )
    return {
        "classification": "rigid force-producing adapter fixture, not native athlete boxing",
        "zone": zone,
        "reverse_geom_order": reverse,
        "physics_steps": 250,
        "native_physics_steps": 0,
        "positive_force_touches": positive,
        "points": list(referee.points),
        "initial_referee": initial,
        "final_referee": referee.snapshot(),
        "intervals": intervals,
        "independent_audit": audited,
        "model_xml": xml,
    }
