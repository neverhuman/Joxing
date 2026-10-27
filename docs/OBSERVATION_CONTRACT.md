# Diagnostic actor observation v1

Implemented for the v11shared-loop prototype, not connected to a trained boxer.
Source: src/joxing/observations.py; semantic measurement adapter in
src/joxing/bodies/native.py. Current schema has1,970real-valuedfloat32features;
all named units/slices and content hash are persisted in
artifacts/v0.0.11/development/observations.json. No normalization is applied.
This is incompatible with the frozen native3,601feature walking teacher;
its input/cache/normalizer contract was not changed.

| Feature group | Count | Ownership/meaning |
|---|---:|---|
| Intrinsic joint positions and velocities | 158 | Own79non-root scalar coordinates each; six root coordinates excluded from this block |
| Muscle activation | 700 | Own native muscles, unchanged anatomy |
| Own visible body kinematics | 198 | Eleven position/linear velocity/orientation/angular velocity anchors |
| Opponent visible body kinematics | 198 | Same eleven anchors; no opponent private joint/activation state |
| Clock, own/opponent points, own pelvis height | 4 | Public game state and physical height |
| Ring vertices in ego frame | 12 | Four actual supplied vertices, ordered by ego bearing |
| Previous applied command | 700 | Own normalized command; pending/opponent commands excluded |

Anchors: pelvis, head_neck, thoracic12, hand_r/l, humerus_r/l, ulna_r/l,
calcn_r/l. Body rotations/linear and angular velocities are real engine readouts
at a single completed state. BodyAdapter verifies exact native scalar-root order,
85joints/700activations, semantic actuator ownership; it provides owned readonly
copies. Measurements themselves include private data and must never be passed
wholesale to the opponent's actor.

All positions are relative to the observing pelvis; vectors and orientations
rotate into its projected forward-yaw frame. Height, gravity and tilt remain
observable: this is not a full-pelvis transform that makes a fall appear upright.
Undefined heading/nonfinite data, mismatched corners or physical sample clocks
reject instead of inventing values. Intrinsic scalar joint coordinates keep
semantic left/right identity; no muscle permutation occurs on a180degree swap.
Opponent velocity is absolute measured velocity expressed in the ego yaw frame,
not an unlabelled relative-velocity quantity.

ActorObservation carries game/player/sampletick/schema metadata plus its owned
feature vector. Both body timestamps must equal the declared publictick×dt.
There is no critic payload in actor assembly. CriticState is a separate future
type. Opponent commands, goals, activation, joint arrays, RNG, hidden state and
bundle identities never enter this feature path. Previous command means applied,
not an action still waiting in a delay queue.

TimedHistory stores already filtered actor vectors; it does not sanitize private
BodyMeasurements. It selects only packets with sample<=receipt<=decision,
returns validity/sample-age/receipt-age and padding, and isolates game/player/
schema. Its snapshots preserve delay/reordering history. Action queues and the
shared physics loop remain next; no current policy uses this history yet.

Validation: final11body observation/body/history subset7passed5.78s; source and
schema hashes are in the receipt. Tests check neutral corner-frame agreement,
real limb velocity response, private opponent field exclusion, shared timestamp
and ownership rejection, public-state copying, and global yaw/translation frame
arithmetic. This is not dynamic motor equivariance, camera/occlusion realism,
sensor calibration, self-play or boxing competence. Ideal kinematics are an
explicit diagnostic sensor contract. The full201-test development suite passed
before this observation extension; these7targeted tests cover the new changes.

Next add timing.py (`CommandPacket`, `ActionQueue`), its pure tests and then wire
observations/history/queues into env.py. Freeze the final observation/cadence/
command/model/rules contract before formal v11physics, and retain complete
application/contact evidence in addition to sampled visual states.
