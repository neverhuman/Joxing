# Engineering Specification: Dual Muscle-Driven Boxing Agents with MS-Human-700

**Version:** 1.0  
**Date:** 2026-10-26  
**Target System:** MS-Human-700 (MuJoCo musculoskeletal model) + msgym + hierarchical self-play RL

---

## 1. Goals and Scope

Produce two full-body MS-Human-700 agents that:

- Track relative movement and inter-agent contacts in real time.
- Measure impact forces (via MuJoCo contact forces) and award points based on force magnitude + anatomical location (head > torso > limbs) + ring position.
- Improve via competitive self-play reinforcement learning, using the existing hierarchical low-dimensional muscle control machinery (DynSyn / TSHT-style synergies or latents).

### In Scope
- Modular MJCF scene composition
- Custom Gymnasium multi-agent environment
- Force and location-based reward system
- Hierarchical latent self-play pipeline
- MuJoCo Warp scaling support

### Out of Scope (Approximations)
- Full soft-tissue deformation
- Muscle fatigue models
- Photorealistic vision (can be added later)
- Real-time human control interfaces

---

## 2. Repository and File Structure

Extend the existing repositories rather than forking everything:

```
msgym/                          # existing repository
├── msgym/envs/
│   ├── boxing_dual_v1.py       # NEW: DualBoxingEnv
│   └── ...
├── boxing/                     # NEW package
│   ├── __init__.py
│   ├── scene_builder.py        # MJCF composition utilities
│   ├── contact_tracker.py      # Force & location extraction
│   ├── rewards.py              # Modular reward terms
│   └── latent_decoder.py       # Synergy / latent → muscle excitations
├── models/boxing/
│   ├── dual_boxers.xml         # Composed scene
│   ├── ring.xml
│   └── fist_geoms.xml
└── DynSyn-SAC/                 # existing training scripts (extend configs)
    └── configs/
        └── boxing_selfplay.json
```

### Dependencies
- Existing `msgym` installation
- `dm_control` (for MJCF composition)
- `mujoco`
- `gymnasium`
- `stable-baselines3` (or custom PPO/SAC implementation)
- Optional: MuJoCo Warp for GPU-parallel simulation

---

## 3. Scene / Model Composition (Detailed MJCF)

The primary model is modular (`MS-Human-700.xml` includes Asset_*, Contact_*, Equality_*, Tendon_*, Muscle_*, Body_Primary/* nested under a `pelvis` root body). Use this modularity.

### Composition Strategy

Implement in `scene_builder.py` or as a static dual XML:

1. Load two copies of the primary model (or hybrid primary + manipulation hands) via `dm_control.mjcf` or string-based prefixing.
2. Prefix every name: `agent0_` and `agent1_` for bodies, joints, geoms, tendons, actuators, sites, and sensors.
3. Root structure example:

```xml
<mujoco model="dual_boxers">
  <include file="ring.xml"/>
  <worldbody>
    <body name="agent0_pelvis" pos="0 0.8 0.95" quat="0.707107 0.707107 0 0">
      <freejoint name="agent0_root"/>
      <!-- include prefixed Body_Primary/* etc. -->
    </body>
    <body name="agent1_pelvis" pos="0 -0.8 0.95" quat="0.707107 -0.707107 0 0">
      <freejoint name="agent1_root"/>
      <!-- include prefixed assets -->
    </body>
  </worldbody>
  <!-- duplicated Muscle_*, Tendon_*, Equality_* with prefixes -->
  <!-- Contact excludes duplicated + inter-agent rules -->
</mujoco>
```

### Contact Setup
- Retain existing self-collision `<exclude>` pairs (from `Contact_Lowerbody.xml` / `Contact_Upperbody.xml`) under each prefix.
- Explicitly allow inter-agent contacts by setting appropriate `contype`/`conaffinity` on hand, head, torso, and arm geoms (or add `<pair>` elements for high-priority fist-to-head/torso contacts).
- Add simplified fist/glove geoms (capsules or low-poly meshes) attached to hand bodies for stable punch contacts.
- Prefer the detailed finger meshes from the Manipulation model for one or both hands if compute allows; otherwise close the fist via equality constraints or fixed joint poses.

### Sensors
- Add force/touch sensors or sites on: both hands (fist center), head, upper torso, abdomen.
- Optional: relative site sensors for opponent distance.

### Ring
- Floor plane + optional cylindrical or box walls with friction.
- Define scoring zones via sites or post-processing.

### Validation Criteria
- Model loads without errors in MuJoCo viewer.
- No immediate self-penetration.
- Agents can contact each other.
- Muscle actuators are correctly duplicated (~1400 total).

---

## 4. Custom Gymnasium Environment (`DualBoxingEnv`)

Subclass `gymnasium.Env` (or extend existing `msgym` base if available).

### Key Methods

#### `__init__`
- Load composed MJCF via `mujoco.MjModel.from_xml_path`.
- Create two independent action/observation mappings.
- Set control timestep (recommend 0.01–0.02 s effective, with internal substeps).

#### `reset`
- Randomize starting positions within ring (facing each other ± small noise).
- Initialize muscle activations near 0.
- Ensure safe pelvis heights.
- Return initial observations.

#### `step(actions)`
1. Map high-level actions → full muscle excitations (via latent decoder or synergy matrix).
2. Apply to `data.ctrl` for both agents.
3. Call `mujoco.mj_step`.
4. Extract contacts and compute rewards.
5. Check termination conditions.
6. Return observation, reward, terminated, truncated, info.

#### `render`
- Standard MuJoCo viewer or offscreen rendering.

### Parallelism Design
Structure the environment so multiple instances can be vectorized. Later port the inner loop to MuJoCo Warp for GPU-scale self-play.

---

## 5. Observation and Action Spaces

### Observation Space (per agent)
Recommended size: ~200–400 dimensions after reduction.

**Components:**
- **Own state**: pelvis height/orientation/velocity, key joint qpos/qvel (or reduced), muscle lengths/forces/activations (subset or averaged), recent action history.
- **Relative state**: opponent pelvis position/velocity in own frame, opponent head/hand sites, relative orientation.
- **Contact summary**: last N frames of incoming force magnitude on head/torso (or binary hit flags + location one-hot).
- **Ring context**: own position relative to ring center, distance to boundaries.
- **Optional**: short history stack or LSTM-friendly features.

Use `Dict` spaces initially, then flatten for training. Provide a privileged full-state critic observation during training if using actor-critic methods.

### Action Space
**Preferred:** Low-dimensional latent or synergy codes (e.g., 32–64 dimensions per agent, matching DynSyn practice). Decoder maps to 700 muscle excitations in [0, 1].

**Fallback:** Direct muscle subset or full 700 (not recommended for self-play stability).

---

## 6. Contact Force and Location Tracking

Implement in `contact_tracker.py` (called every step):

```python
def extract_impacts(model, data, agent_prefix, opponent_prefix):
    impacts = []
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1, geom2 = contact.geom1, contact.geom2
        body1 = model.geom_bodyid[geom1]
        body2 = model.geom_bodyid[geom2]
        name1 = model.body(body1).name
        name2 = model.body(body2).name
        
        # Filter inter-agent only
        if ((name1.startswith(agent_prefix) and name2.startswith(opponent_prefix)) or
            (name1.startswith(opponent_prefix) and name2.startswith(agent_prefix))):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            magnitude = np.linalg.norm(force[:3])
            location = classify_location(name1, name2)
            impacts.append({
                "force": magnitude,
                "location": location,
                "point": contact.pos.copy(),
                "geom1": geom1,
                "geom2": geom2
            })
    return impacts
```

### Location Classification
Use body name substrings or pre-defined site groups:
- `head`
- `upper_torso`
- `lower_torso`
- `arms`
- `legs`

Aggregate per step: maximum force, total impulse, and location-weighted score.

---

## 7. Reward Function Specification

Implement modular terms in `rewards.py` (weights tunable via configuration).

### Positive Terms (for agent i vs j)
- **Impact score**:  
  \(\sum_k \text{force}_k \times w_{\text{loc}}(\text{location}_k) \times \text{clean_hit_factor}_k\)

  Recommended weights:
  - \(w_{\text{head}} = 3.0\)
  - \(w_{\text{torso}} = 1.5\)
  - \(w_{\text{limbs}} = 0.5\)

  Clean-hit factor uses relative velocity alignment and contact duration.

- **Engagement / ring control**: Bonus for being closer to center while facing opponent; small bonus for successful slips (low incoming force while opponent is attacking).

### Negative Terms
- **Received damage**: Symmetric negative of opponent’s impact score (or scaled).
- **Fall / knockdown**: Large penalty when pelvis height < threshold or orientation exceeds tilt limit; terminal if sustained.
- **Energy**: Small penalty on \(\|u\|^2\) or muscle force sum.
- **Illegal / self**: Penalty for self-contact forces above threshold or out-of-ring position.

### Terminal Conditions
- Knockdown (pelvis height + duration threshold)
- Cumulative damage threshold
- Timeout
- Mutual fall

**Zero-sum structure**: Winner receives +1 (or scaled remaining HP), loser receives –1.

### Curriculum Progression
1. Dense solo rewards (punch tracking and footwork)
2. Freeze one agent (scripted or pretrained opponent)
3. Full competitive self-play

---

## 8. Hierarchical RL Pipeline

Three stages, building on DynSyn-SAC and TSHT patterns already present in the ecosystem.

### Stage 1 – Single-Agent Skill Pretraining
- Extend existing locomotion/manipulation environments or create a single-boxer punching/footwork imitation environment.
- Use motion-capture boxing clips (or procedurally generated punch trajectories) as reference.
- Reward = joint tracking + stability + simple punch success.
- Train with DynSyn-SAC or SAC on synergy/latent actions.
- **Output**: A low-level decoder (synergy matrix or small MLP) that maps ~32–64 dimensional codes → 700 muscle excitations.

### Stage 2 – Latent Manifold Regularization
- Collect trajectories from Stage 1.
- Train an autoencoder or hypersphere-constrained VAE on the action codes so the latent space only contains physically plausible motions.
- Freeze the decoder.

### Stage 3 – Competitive Self-Play
- High-level policy outputs latent codes.
- **Algorithm options**:
  - Latent-Space Neural Fictitious Self-Play (LS-NFSP)
  - Population-based self-play with opponent pooling
- Warm-up against scripted or Stage-1 opponents, then full co-evolution.
- Use vectorized environments + MuJoCo Warp for thousands of parallel rollouts.
- Shared policy with self-play or independent policies both viable; start with shared + opponent snapshots.

Training scripts can reuse and extend the existing `DynSyn-SAC/SB3-Scripts/train.py` pattern with new configuration files.

---

## 9. Training Infrastructure and Scaling

### Base Algorithms
- Stable-Baselines3 SAC/PPO, or
- Custom implementation matching DynSyn-SAC patterns.

### Parallelism Path
1. Start with CPU vectorized Gymnasium environments.
2. Migrate inner physics loop to MuJoCo Warp (already demonstrated by the original authors for high-fidelity tracking) for self-play scale.

### Logging Requirements
- Contact force histograms
- Location hit rates (head vs torso vs limbs)
- Knockdown frequency
- Win rates versus frozen opponents
- Energy expenditure trends

### Checkpointing
- Save both high-level policies and the frozen low-level decoder.
- Version control latent space dimension and decoder weights.

---

## 10. Approximations, Risks, and Mitigations

| Gap / Risk                        | Mitigation                                                                 |
|-----------------------------------|----------------------------------------------------------------------------|
| Rigid bodies only                 | Tune `solimp`/`solref` for softer contacts; add glove geoms with higher damping |
| Fist modeling                     | Use Manipulation model hands or fixed closed-fist equality constraints + capsules |
| 1400-actuator stiffness           | Strict hierarchical control; never train end-to-end in raw muscle space for self-play |
| Contact explosions                | Conservative initial contact parameters; curriculum on force magnitude     |
| No native multi-agent support    | Custom environment (this specification)                                    |
| High compute requirements         | MuJoCo Warp + latent actions                                               |
| Numerical instability             | Hierarchical control + careful contact tuning + curriculum                 |

---

## 11. Implementation Milestones

| Milestone | Description                                      | Target Duration | Success Criteria                                      |
|-----------|--------------------------------------------------|-----------------|-------------------------------------------------------|
| **M1**    | Dual MJCF loads and agents can be pose-controlled independently | 1–2 weeks      | Model loads; basic inter-agent contacts visible       |
| **M2**    | DualBoxingEnv with force extraction, location classification, and dense reward | 1–2 weeks | Single-agent punching works; force values logged      |
| **M3**    | Stage-1 skill pretraining produces plausible punches and footwork | 2–3 weeks     | Hierarchical control produces stable punch motions    |
| **M4**    | Latent distillation + initial self-play against frozen opponent | 2 weeks       | Improving hit rates observed                          |
| **M5**    | Full competitive self-play with measurable strategy emergence | 3–4 weeks     | Ring control, head targeting, and defensive slips emerge |

---

## 12. Next Steps and Validation Plan

1. Implement and validate dual MJCF composition (M1).
2. Build and unit-test `contact_tracker.py` and `rewards.py`.
3. Create `DualBoxingEnv` and verify observation/action spaces.
4. Run Stage-1 pretraining using existing DynSyn-SAC infrastructure.
5. Iterate on contact parameters and reward weights using logged metrics.
6. Scale to self-play with MuJoCo Warp once single-agent skills are stable.

---

## Appendix A: Key MuJoCo APIs

- `mujoco.mj_contactForce(model, data, contact_id, force)` — retrieve 6D contact wrench
- `data.contact[i].geom1 / geom2` — identify colliding geometries
- `model.body(body_id).name` — resolve body names for location classification
- Free joints or equivalent 6-DoF roots for independent agent movement

## Appendix B: Recommended Starting Configuration

```yaml
control_timestep: 0.02
latent_dim: 48
head_weight: 3.0
torso_weight: 1.5
limb_weight: 0.5
fall_height_threshold: 0.4
max_episode_steps: 500
```

---

*This specification is designed to be directly implementable on top of the modular MS-Human-700 XML structure and the existing msgym / DynSyn-SAC training infrastructure.*
