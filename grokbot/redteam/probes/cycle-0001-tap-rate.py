"""Referee-only ceiling on points/s for a minimal "feather tap" cycle (cycle 0001, F4).

Drives one candidate HitTracker with a synthetic per-tick stream: `contact` ticks on
the head at `force` N with the minimum approach/active speed and face cosine, then
`clear` ticks at the minimum clearance with 5 cm retraction, repeated for 60 s.
No physics: this bounds what the rules pay, not what a muscle body can perform.

    uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0001-tap-rate.py
"""

from joxing.rules import HitTracker, Rules, Touch

ROUND_TICKS = 60_000
EXTENDED_M = 0.30


def tap_round(force, contact, clear, rules):
    tracker = HitTracker("probe", "A_glove_r", rules)
    tick = points = events = 0
    touch = rules.approach_min_mps, rules.active_speed_min_mps, rules.face_cos_min
    while tick < ROUND_TICKS:
        for phase, length in (("contact", contact), ("clear", clear)):
            for _ in range(length):
                if tick >= ROUND_TICKS:
                    break
                if phase == "contact":
                    emitted = tracker.update(
                        tick,
                        [Touch(str(tick), "head", force, *touch)],
                        clearance_m=0.0,
                        extension_m=EXTENDED_M,
                    )
                else:
                    emitted = tracker.update(
                        tick,
                        [],
                        clearance_m=rules.clearance_m,
                        extension_m=EXTENDED_M - rules.retraction_m - 0.02,
                        bell=tick == ROUND_TICKS - 1,
                    )
                points += sum(e["points"] for e in emitted)
                events += len(emitted)
                tick += 1
    return events, points


def main():
    rules = Rules()
    print(f"rules_hash {rules.hash}")
    for force, contact, clear in [
        (251, 1, 30),
        (26, 10, 30),
        (6, 50, 30),
        (26, 10, 90),
        (251, 1, 29),
    ]:
        events, points = tap_round(force, contact, clear, rules)
        print(
            f"{force:>4} N x {contact:>2} ms ({force * contact / 1000:.3f} N s), "
            f"clear {clear:>3} ms -> {events:5d} events, {points:5d} points/60 s/glove "
            f"({points / 60:.1f}/s)"
        )


if __name__ == "__main__":
    main()
