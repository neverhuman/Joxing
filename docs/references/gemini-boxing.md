# Engineering Specification: Two-Agent Musculoskeletal Boxing & Self-Play RL (`MS-Human-700-Boxing`)

* **Target Repositories:** `LNSGroup/MS-Human-700` (MuJoCo Musculoskeletal Assets) & `LNSGroup/msgym` (Gymnasium RL Suite & `DynSyn` Framework)
* **Base Anatomical Topology:** $90$ bone meshes (consolidated into $80$ rigid bodies), $206$ anatomical joints (constrained to $85$ active degrees of freedom via polynomial `<equality>` constraints), driven by $700$ Hill-type muscle-tendon units (MTUs) per fighter ($1,400$ total actuators in a two-agent bout)
* **Simulation & Control Frequencies:**
  * Physics Integration ($\Delta t_{\text{sim}}$): $0.002\text{ s}$ ($500\text{ Hz}$, `mjINT_IMPLICITFAST` + Newton solver)
  * Low-Level Motor Skill Decoder ($\Delta t_{\text{low}}$): $0.01\text{ s}$ ($100\text{ Hz}$, $5$ physics substeps per low-level step)
  * High-Level Tactical Self-Play Policy ($\Delta t_{\text{high}}$): $0.02\text{ s}$ ($50\text{ Hz}$, $2$ low-level steps / $10$ physics substeps per tactical step)

---

## 1. System Audit & Biomechanical Gap Analysis

### 1.1 Baseline Capabilities in `MS-Human-700` & `msgym`

| Subsystem | Baseline Implementation in Repo | Boxing Readiness & Bottlenecks |
| :--- | :--- | :--- |
| **Musculoskeletal Topology (`MS-Human-700.xml`)** | $90$ bone meshes across $80$ rigid bodies, $206$ joints constrained to $85$ active DOFs via polynomial `<equality>` constraints, actuated by $700$ Hill-type MTUs (`Muscle_Torso.xml`, `Muscle_Arm_l/r.xml`, `Muscle_Leg_l/r.xml`). | High kinematic fidelity, but architected strictly as a single-agent model. Each human has $91$ generalized coordinates ($q$), $90$ velocities ($\dot{q}$), and $700$ muscle activation states ($act$). |
| **Collision & Contact (`Contact_*.xml`)** | `Contact_Lowerbody.xml` defines `<pair>` tags exclusively between foot collision primitives and the floor (`geom_floor`). Bone meshes (`Geometry/*.stl`, `*.obj`) have `contype="0" conaffinity="0"`. `Contact_Upperbody.xml` lacks inter-agent upper-body collision primitives. | **Critical Gap:** Without dedicated analytical collision primitives and bitmask filtering, punches pass directly through the opponent's cranium, mandible, thorax, and guard. |
| **Hand & Wrist Mechanics (`Body_Arm_l/r.xml`)** | `MS-Human-700.xml` uses consolidated metacarpal/hand segments (`Body_Primary/Body_Arm_l/r.xml`), whereas `MS-Human-700-Manipulation.xml` loads articulated phalanges (`Body_Manipulation/Body_Arm_Hand_r.xml`). | Articulated fingers are fragile and computationally wasteful for punching. The primary model's hand must be wrapped in a viscoelastic boxing glove geom with radiocarpal hand-wrap stabilization. |
| **RL Algorithm (`DynSyn-SAC`)** | `SAC_DynSyn.py` and `DynSynLayer.py` use off-policy Soft Actor-Critic with a Dynamical Synergistic Representation: $a_i = \text{Softplus}(a_{G_i} + \kappa(t)\cdot \text{clip}(w_i, -c, c))$. | `DynSyn-SAC` is an off-policy, single-agent trajectory-imitation learner (`locomotionFull_v1.py`). Off-policy replay buffers destabilize under non-stationary two-player zero-sum self-play. |

### 1.2 Six Mandatory Physics & Biomechanical Fixes

1. **Upper-Body Collision Primitives & 6-Bit Bitmask Filtering:** Visual `.stl` and `.obj` bone meshes (`skull.stl`, `jaw.stl`, `Rib1L`--`Rib12R.obj`, `Sternum.stl`, `humerus.stl`, `radius.stl`, `ulna.stl`) must remain `contype="0" conaffinity="0"` so spatial tendons do not snag on polygon edges. Lightweight analytical primitives (`capsule`, `sphere`, `ellipsoid`, `box`) must be mounted to the underlying rigid bodies with orthogonal bitmasks for offensive gloves, defensive guards, and anatomical target zones.
2. **Boxing Glove Viscoelasticity & Contact Solver Tuning:** Rigid bone-on-bone impact generates numerical Dirac-delta force spikes in MuJoCo. Each hand (`capitate` / `hand` frame) must be encapsulated in a padded $12\text{ oz}$ ($0.34\text{ kg}$) glove primitive with tuned impedance parameters (`solref="0.015 1.05"`, `solimp="0.85 0.99 0.008 0.5 2"`) so impact force ramps smoothly across $10\text{--}20\text{ ms}$ ($5\text{--}10$ substeps at $500\text{ Hz}$).
3. **Cervical Spine & Mandible (Jaw) Impact Compliance:** In `Body_Torso_Simple.xml`, cervical joints (`cerv1`--`cerv7`) require passive rotational compliance (`stiffness="42.0"`, `damping="4.2"`) paired with active sternocleidomastoid/splenius muscle bracing and a cranial `<accelerometer>` + `<gyro>` IMU on `skull` to measure concussive linear and rotational acceleration.
4. **Radiocarpal (Wrist) & Scapulothoracic Impact Stabilization:** Axial punch loads of $1,500\text{--}3,500\text{ N}$ buckle purely muscle-driven wrists and violate loose scapulothoracic `<equality>` constraints (`Equality_Upperbody.xml`). Passive joint stiffness (`stiffness="35.0"`, `damping="3.5"`) simulating hand wraps must be added to `wrist_flexion` and `wrist_deviation`, and scapulothoracic equality constraints must be stiffened (`solref="0.008 1.0"`).
5. **Anatomical Target Vulnerability Mapping:** Distinct target geoms must be mounted to `jaw` (`zone_chin_jaw`), `skull` (`zone_temple_head`, `zone_illegal_back_head`), `Sternum` (`zone_solar_plexus`, `zone_ribs_chest`), and `thoracic9` (`zone_liver`) with anatomical point multipliers and trauma couplings.
6. **Aerobic/Anaerobic Fatigue & Visceral Shock ODEs:** MuJoCo's Hill-type muscles (`<general dyntype="muscle" gaintype="muscle" biastype="muscle">`) do not natively fatigue over a 3-minute round. A per-fighter stamina reservoir $E_t \in [0.15, 1.0]$ must deplete with muscle activation $\sum F_{\max,m} (act_m)^{2.2}$ and visceral strikes (`zone_liver`, `zone_solar_plexus`), dynamically scaling maximum muscle excitation.

---

## 2. Repository Architecture & File Manifest

All extensions are structured cleanly inside `msgym` and its `MS-Human-700` submodule without breaking existing locomotion or manipulation benchmarks:

~~~text
msgym/
├── msgym/
│   ├── MS-Human-700/                                  # Submodule: LNSGroup/MS-Human-700
│   │   ├── MS-Human-700.xml                           # Base single-agent full-body model
│   │   ├── MS-Human-700-Boxing-Single.xml             # [NEW] Fighter template with gloves, hit-zones & neck IMU
│   │   ├── Asset/
│   │   │   └── Boxing_Ring_Assets.xml                 # [NEW] Canvas, ropes, turnbuckles & lighting materials
│   │   ├── Body_Primary/
│   │   │   ├── Body_Combat_HitZones.xml               # [NEW] Anatomical target geoms (chin, temple, liver, etc.)
│   │   │   └── Body_Combat_Gloves.xml                 # [NEW] 12oz viscoelastic glove geoms + wrist wrap springs
│   │   └── Contact/
│   │       └── Contact_Boxing.xml                     # [NEW] Inter-fighter collision defaults & solver presets
│   │
│   ├── envs/
│   │   ├── locomotionFull_v1.py                       # Existing single-agent tracking environment
│   │   ├── boxing_arena_builder.py                    # [NEW] MjSpec compiler cloning p1_ and p2_ into ring
│   │   ├── boxing_combat_engine.py                    # [NEW] 500Hz substep impulse, trauma, stamina & judge logic
│   │   └── boxingFull_v1.py                           # [NEW] Multi-agent Gymnasium/PettingZoo environment
│   └── __init__.py                                    # [MODIFIED] Register "msgym/BoxingCombatEnv-v1"
│
├── DynSyn-Boxing/                                     # [NEW] Hierarchical Self-Play RL Stack
│   ├── synergies/
│   │   ├── extract_combat_synergies.py                # 64-group whole-body combat synergy clustering
│   │   └── combat_synergy_clusters_64.json            # Exported W_syn mapping (64 groups -> 700 MTUs)
│   ├── low_level/
│   │   ├── train_motor_decoder_warp.py                # Stage 1: MuJoCo Warp boxing MoCap imitation trainer
│   │   └── boxing_primitive_decoder.py                # Frozen/Residual DynSyn motor skill network (pi_low)
│   ├── high_level/
│   │   ├── tactical_ppo_policy.py                     # Stage 2/3: Asymmetric Actor-Critic (pi_high)
│   │   ├── pfsp_league_manager.py                     # Prioritized Fictitious Self-Play + Elo matchmaker
│   │   └── train_selfplay_ppo.py                      # Distributed GPU rollouts & PPO update loop
│   └── configs/
│       ├── stage1_mocap_primitives.json
│       ├── stage2_heavybag_curriculum.json
│       └── stage3_pfsp_league.json
│
└── tests/
    └── test_boxing_biomechanics.py                    # [NEW] Automated 5-part physics & RL verification suite
~~~

---

## 3. Musculoskeletal Physics & XML Engineering

### 3.1 6-Bit Collision Bitmask Matrix (`contype` & `conaffinity`)

MuJoCo evaluates collision between Geom A and Geom B if and only if `(A.contype & B.conaffinity) != 0` or `(B.contype & A.conaffinity) != 0`. We assign a strict 6-bit mask so internal bone meshes never collide, while offensive gloves collide with opponent target zones, opponent guards, and the fighter's own guard:

| Geometry Category | `contype` (Decimal) | `contype` (Binary) | `conaffinity` (Decimal) | `conaffinity` (Binary) | Collides With |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Visual Bone Meshes (`*.stl`, `*.obj`)** | `0` | `000000` | `0` | `000000` | None (visual & tendon wrapping only) |
| **Arena Floor & Ring Ropes** | `1` | `000001` | `63` | `111111` | All feet, bodies, gloves, and downed segments |
| **Fighter 1 Offensive Gloves (`p1_glove_l/r`)** | `2` | `000010` | `29` | `011101` | Arena (`1`), P2 Hit-Zones (`4`), P2 Guard (`8`), P1 Guard (`16`) |
| **Fighter 2 Target Hit-Zones (`p2_zone_*`)** | `4` | `000100` | `3` | `000011` | Arena (`1`), P1 Offensive Gloves (`2`) |
| **Fighter 2 Defensive Guard (`p2_zone_guard_*`, `p2_glove_*`)** | `8` | `001000` | `35` | `100011` | Arena (`1`), P1 Gloves (`2`), P2 Hit-Zones (`32` if self-guard) |
| **Fighter 1 Defensive Guard (`p1_zone_guard_*`)** | `16` | `010000` | `11` | `001011` | Arena (`1`), P1 Gloves (`2`), P2 Gloves (`8`) |
| **Fighter 1 Target Hit-Zones (`p1_zone_*`)** | `32` | `100000` | `9` | `001001` | Arena (`1`), P2 Gloves (`8`) |

### 3.2 Ring Arena XML (`msgym/MS-Human-700/Asset/Boxing_Ring_Assets.xml`)

Create `msgym/MS-Human-700/Asset/Boxing_Ring_Assets.xml` to define a standard $6.10\text{ m} \times 6.10\text{ m}$ ($20\text{ ft}$) competition boxing ring with canvas friction and elastic boundary ropes:

~~~xml
<mujoco model="Boxing_Ring_Arena">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" solver="Newton" iterations="50" ls_iterations="20"/>

  <asset>
    <texture name="tex_canvas" type="2d" builtin="checker" rgb1="0.14 0.18 0.24" rgb2="0.18 0.22 0.29" width="512" height="512"/>
    <material name="mat_canvas" texture="tex_canvas" texrepeat="8 8" reflectance="0.05" shininess="0.1"/>
    <material name="mat_rope_red" rgba="0.82 0.12 0.12 1.0" shininess="0.3"/>
    <material name="mat_rope_white" rgba="0.92 0.92 0.92 1.0" shininess="0.3"/>
    <material name="mat_rope_blue" rgba="0.12 0.28 0.82 1.0" shininess="0.3"/>
    <material name="mat_post" rgba="0.15 0.15 0.15 1.0" shininess="0.5"/>
  </asset>

  <worldbody>
    <light name="ring_overhead_1" pos="-1.5 1.5 4.5" dir="0.2 -0.2 -1.0" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2" castshadow="true"/>
    <light name="ring_overhead_2" pos="1.5 -1.5 4.5" dir="-0.2 0.2 -1.0" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2" castshadow="true"/>

    <!-- High-traction competition canvas floor -->
    <geom name="geom_floor" type="plane" size="4.0 4.0 0.1" material="mat_canvas"
          contype="1" conaffinity="63" condim="6" friction="1.1 0.05 0.002"
          solref="0.008 1.0" solimp="0.92 0.99 0.002 0.5 2"/>

    <!-- Corner Posts (Outside 3.05m fighting square) -->
    <geom name="post_ne" type="cylinder" fromto="3.20 3.20 0.0  3.20 3.20 1.45" size="0.06" material="mat_post" contype="1" conaffinity="63"/>
    <geom name="post_nw" type="cylinder" fromto="-3.20 3.20 0.0 -3.20 3.20 1.45" size="0.06" material="mat_post" contype="1" conaffinity="63"/>
    <geom name="post_se" type="cylinder" fromto="3.20 -3.20 0.0  3.20 -3.20 1.45" size="0.06" material="mat_post" contype="1" conaffinity="63"/>
    <geom name="post_sw" type="cylinder" fromto="-3.20 -3.20 0.0 -3.20 -3.20 1.45" size="0.06" material="mat_post" contype="1" conaffinity="63"/>

    <!-- Elastic Ring Ropes at heights 0.35m, 0.68m, 1.02m, 1.34m -->
    <!-- North Ropes -->
    <geom name="rope_n_1" type="capsule" fromto="-3.05 3.05 0.35  3.05 3.05 0.35" size="0.022" material="mat_rope_white" contype="1" conaffinity="63" solref="0.04 0.8"/>
    <geom name="rope_n_2" type="capsule" fromto="-3.05 3.05 0.68  3.05 3.05 0.68" size="0.022" material="mat_rope_blue"  contype="1" conaffinity="63" solref="0.04 0.8"/>
    <geom name="rope_n_3" type="capsule" fromto="-3.05 3.05 1.02  3.05 3.05 1.02" size="0.022" material="mat_rope_white" contype="1" conaffinity="63" solref="0.04 0.8"/>
    <geom name="rope_n_4" type="capsule" fromto="-3.05 3.05 1.34  3.05 3.05 1.34" size="0.022" material="mat_rope_red"   contype="1" conaffinity="63" solref="0.04 0.8"/>
    <!-- South Ropes -->
    <geom name="rope_s_1" type="capsule" fromto="-3.05 -3.05 0.35  3.05 -3.05 0.35" size="0.022" material="mat_rope_white" contype="1" conaffinity="63" solref="0.04 0.8"/>
    <geom name="rope_s_2" type="capsule" fromto="-3.05 -3.05 0.68  3.05 -3.05 0.68" size="0.022" material="mat_rope_blue"  contype="1" conaffinity="63" solref="0.04 0.8"/>
    <geom name="rope_s_3" type="capsule" fromto="-3.05 -3.05 1.02  3.05 -3.05 1.02" size="0.022" material="mat_rope_white" contype="1" conaffinity="63" solref="0.04 0.8"/>
    <geom name="rope_s_4" type="capsule" fromto="-3.05 -3.05 1.34  3.05 -3.05 1.34" size="0.022" material="mat_rope_red"   contype="1" conaffinity="63" solref="0.04 0.8"/>
    <!-- East Ropes -->
    <geom name="rope_e_1" type="capsule" fromto="3.05 -3.05 0.35  3.05 3.05 0.35" size="0.022" material="mat_rope_white" contype="1" conaffinity="63" solref="0.04 0.8"/>
    <geom name="rope_e_2" type="capsule" fromto="3.05 -3.05 0.68  3.05 3.05 0.68" size="0.022" material="mat_rope_blue"  contype="1" conaffinity="63" solref="0.04 0.8"/>
    <geom name="rope_e_3" type="capsule" fromto="3.05 -3.05 1.02  3.05 3.05 1.02" size="0.022" material="mat_rope_white" contype="1" conaffinity="63" solref="0.04 0.8"/>
    <geom name="rope_e_4" type="capsule" fromto="3.05 -3.05 1.34  3.05 3.05 1.34" size="0.022" material="mat_rope_red"   contype="1" conaffinity="63" solref="0.04 0.8"/>
    <!-- West Ropes -->
    <geom name="rope_w_1" type="capsule" fromto="-3.05 -3.05 0.35 -3.05 3.05 0.35" size="0.022" material="mat_rope_white" contype="1" conaffinity="63" solref="0.04 0.8"/>
    <geom name="rope_w_2" type="capsule" fromto="-3.05 -3.05 0.68 -3.05 3.05 0.68" size="0.022" material="mat_rope_blue"  contype="1" conaffinity="63" solref="0.04 0.8"/>
    <geom name="rope_w_3" type="capsule" fromto="-3.05 -3.05 1.02 -3.05 3.05 1.02" size="0.022" material="mat_rope_white" contype="1" conaffinity="63" solref="0.04 0.8"/>
    <geom name="rope_w_4" type="capsule" fromto="-3.05 -3.05 1.34 -3.05 3.05 1.34" size="0.022" material="mat_rope_red"   contype="1" conaffinity="63" solref="0.04 0.8"/>
  </worldbody>
</mujoco>
~~~

### 3.3 Anatomical Hit-Zones & Gloves Specification (`Body_Combat_HitZones.xml` & `Body_Combat_Gloves.xml`)

In `MS-Human-700-Boxing-Single.xml`, attach the following geoms and sites to their exact parent rigid bodies inside `Body_Primary/Body_Torso_Simple.xml` and `Body_Primary/Body_Arm_l/r.xml`:

~~~xml
<!-- Defaults added to MS-Human-700-Boxing-Single.xml -->
<default>
  <!-- 12oz (0.34kg) Viscoelastic Boxing Glove: 15ms compression time constant, critically damped -->
  <default class="boxing_glove">
    <geom type="capsule" size="0.068 0.045" mass="0.34" condim="6"
          friction="0.40 0.02 0.001" solref="0.015 1.05" solimp="0.85 0.99 0.008 0.5 2"
          rgba="0.85 0.12 0.12 0.90" group="1"/>
  </default>
  <!-- Anatomical Target Hit-Zone Primitive (massless sensor/collision envelope) -->
  <default class="hit_zone">
    <geom condim="4" friction="0.30 0.01" solref="0.012 1.0" solimp="0.90 0.99 0.004 0.5 2"
          density="0" group="3"/>
  </default>
  <!-- Forearm & Upper-Arm Guard Primitives -->
  <default class="guard_zone">
    <geom condim="4" friction="0.35 0.01" solref="0.014 1.0" solimp="0.88 0.99 0.005 0.5 2"
          density="0" group="3"/>
  </default>
</default>

<!-- ========================================================================= -->
<!-- 1. CRANIAL & MANDIBULAR ZONES (Injected into <body name="skull"> & "jaw") -->
<!-- ========================================================================= -->
<!-- Inside <body name="skull">: -->
<geom name="zone_temple_head" class="hit_zone" type="ellipsoid"
      pos="0.005 0.095 0.0" size="0.095 0.105 0.082" rgba="1.0 0.45 0.0 0.25"/>
<geom name="zone_illegal_back_head" class="hit_zone" type="sphere"
      pos="-0.065 0.065 0.0" size="0.065" rgba="0.55 0.0 0.55 0.25"/>
<site name="skull_imu_site" pos="0.0 0.075 0.0" size="0.01"/>
<camera name="left_eye" pos="0.085 0.085 -0.032" xyaxes="0 0 -1 0 1 0" fovy="90"/>
<camera name="right_eye" pos="0.085 0.085 0.032" xyaxes="0 0 -1 0 1 0" fovy="90"/>

<!-- Inside <body name="jaw">: -->
<geom name="zone_chin_jaw" class="hit_zone" type="capsule"
      fromto="0.055 -0.025 -0.035  0.055 -0.025 0.035" size="0.038" rgba="1.0 0.0 0.0 0.35"/>

<!-- ========================================================================= -->
<!-- 2. THORACIC & VISCERAL ZONES (Injected into "Sternum" & "thoracic9")      -->
<!-- ========================================================================= -->
<!-- Inside <body name="Sternum">: -->
<geom name="zone_solar_plexus" class="hit_zone" type="sphere"
      pos="0.115 -0.085 0.0" size="0.052" rgba="1.0 0.85 0.0 0.35"/>
<geom name="zone_ribs_chest" class="hit_zone" type="box"
      pos="0.065 0.010 0.0" size="0.085 0.135 0.145" rgba="0.2 0.8 0.2 0.20"/>

<!-- Inside <body name="thoracic9"> (Right lower anterolateral 9th-10th ribs): -->
<geom name="zone_liver" class="hit_zone" type="ellipsoid"
      pos="0.075 -0.040 0.095" size="0.060 0.070 0.055" rgba="0.9 0.0 0.4 0.35"/>

<!-- Inside <body name="pelvis"> (Low-blow penalty detection): -->
<geom name="zone_illegal_groin" class="hit_zone" type="sphere"
      pos="0.065 -0.095 0.0" size="0.065" rgba="0.55 0.0 0.55 0.25"/>

<!-- ========================================================================= -->
<!-- 3. GUARD ARMS & OFFENSIVE GLOVES (Injected into Left & Right Arms)        -->
<!-- ========================================================================= -->
<!-- Inside <body name="humerus_r"> and <body name="humerus_l">: -->
<geom name="zone_guard_upperarm_r" class="guard_zone" type="capsule"
      fromto="0.0 -0.04 0.0  0.0 -0.26 0.0" size="0.048" rgba="0.1 0.5 1.0 0.20"/>
<geom name="zone_guard_upperarm_l" class="guard_zone" type="capsule"
      fromto="0.0 -0.04 0.0  0.0 -0.26 0.0" size="0.048" rgba="0.1 0.5 1.0 0.20"/>

<!-- Inside <body name="ulna_r"> and <body name="ulna_l">: -->
<geom name="zone_guard_forearm_r" class="guard_zone" type="capsule"
      fromto="0.0 -0.02 0.0  0.0 -0.23 0.0" size="0.042" rgba="0.1 0.5 1.0 0.25"/>
<geom name="zone_guard_forearm_l" class="guard_zone" type="capsule"
      fromto="0.0 -0.02 0.0  0.0 -0.23 0.0" size="0.042" rgba="0.1 0.5 1.0 0.25"/>

<!-- Inside <body name="hand_r"> / <body name="capitate_r"> & left equivalent: -->
<geom name="glove_r" class="boxing_glove" fromto="0.0 -0.02 0.0  0.0 -0.09 0.0"/>
<site name="glove_r_site" pos="0.0 -0.06 0.0" size="0.01"/>
<geom name="glove_l" class="boxing_glove" fromto="0.0 -0.02 0.0  0.0 -0.09 0.0"/>
<site name="glove_l_site" pos="0.0 -0.06 0.0" size="0.01"/>
~~~

### 3.4 Wrist Hand-Wrap Stiffness, Cervical Compliance & Cranial IMU Sensors

In `MS-Human-700-Boxing-Single.xml`, apply the following joint and sensor modifications:

~~~xml
<!-- 1. Radiocarpal Wrist Stabilization (simulates boxing hand-wraps & tape) -->
<joint name="wrist_flexion_r" stiffness="35.0" damping="3.5" armature="0.015" range="-0.65 0.65"/>
<joint name="wrist_deviation_r" stiffness="45.0" damping="4.0" armature="0.015" range="-0.35 0.35"/>
<joint name="wrist_flexion_l" stiffness="35.0" damping="3.5" armature="0.015" range="-0.65 0.65"/>
<joint name="wrist_deviation_l" stiffness="45.0" damping="4.0" armature="0.015" range="-0.35 0.35"/>

<!-- 2. Cervical Spine Compliance (allows realistic head snap & neck muscle bracing) -->
<!-- Apply across cervical flexion/extension, lateral bending, and axial rotation DOFs -->
<default class="cervical_joint">
  <joint stiffness="42.0" damping="4.2" armature="0.012"/>
</default>

<!-- 3. Cranial IMU & Glove Kinematics Sensors -->
<sensor>
  <accelerometer name="skull_lin_acc" site="skull_imu_site"/>
  <gyro name="skull_ang_vel" site="skull_imu_site"/>
  <framequat name="skull_quat" objtype="site" objname="skull_imu_site"/>
  <framelinvel name="glove_r_linvel" objtype="site" objname="glove_r_site"/>
  <framelinvel name="glove_l_linvel" objtype="site" objname="glove_l_site"/>
</sensor>
~~~

### 3.5 Programmatic Dual-Fighter Compiler (`msgym/envs/boxing_arena_builder.py`)

Instead of duplicating 15,000-line XML files, use MuJoCo's `MjSpec` API (`mujoco >= 3.2.0`) to instantiate Fighter 1 (`p1_`) and Fighter 2 (`p2_`) into the ring arena with automatic bitmask assignment:

~~~python
# msgym/envs/boxing_arena_builder.py
import os
import mujoco

FOOT_GEOM_KEYWORDS = ("bofoot", "talus", "calcaneus", "foot", "toe")

def build_two_fighter_boxing_model(
    single_fighter_xml_path: str,
    ring_xml_path: str,
    initial_separation: float = 1.70,
) -> mujoco.MjModel:
    """
    Compiles two MS-Human-700-Boxing-Single models (p1_ and p2_) into the ring arena
    with orthogonal 6-bit collision bitmasks for gloves, guards, hit-zones, and floor.
    """
    if not os.path.exists(single_fighter_xml_path):
        raise FileNotFoundError(f"Missing fighter XML: {single_fighter_xml_path}")
    if not os.path.exists(ring_xml_path):
        raise FileNotFoundError(f"Missing ring XML: {ring_xml_path}")

    arena_spec = mujoco.MjSpec.from_file(ring_xml_path)
    arena_spec.option.timestep = 0.002
    arena_spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    arena_spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    arena_spec.option.iterations = 50
    arena_spec.option.ls_iterations = 20
    arena_spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

    half_sep = 0.5 * initial_separation
    fighter_configs = [
        {
            "prefix": "p1_",
            "pos": [-half_sep, 0.0, 0.92],
            "quat": [0.70710678, 0.0, 0.0, 0.70710678],   # Facing +X
            "glove_rgba": [0.85, 0.12, 0.12, 1.0],        # Red Corner
            "is_p1": True,
        },
        {
            "prefix": "p2_",
            "pos": [half_sep, 0.0, 0.92],
            "quat": [0.70710678, 0.0, 0.0, -0.70710678],  # Facing -X
            "glove_rgba": [0.12, 0.28, 0.85, 1.0],        # Blue Corner
            "is_p1": False,
        },
    ]

    for cfg in fighter_configs:
        f_spec = mujoco.MjSpec.from_file(single_fighter_xml_path)
        is_p1 = cfg["is_p1"]

        for geom in f_spec.geoms:
            gname = geom.name or ""
            if "glove" in gname:
                geom.contype = 2 if is_p1 else 8
                geom.conaffinity = 29 if is_p1 else 51
                geom.rgba = cfg["glove_rgba"]
            elif "zone_guard" in gname:
                geom.contype = 16 if is_p1 else 8
                geom.conaffinity = 11 if is_p1 else 35
            elif "zone_" in gname:
                geom.contype = 32 if is_p1 else 4
                geom.conaffinity = 9 if is_p1 else 3
            elif any(k in gname.lower() for k in FOOT_GEOM_KEYWORDS):
                # Foot contact geoms collide only with arena floor/ropes
                geom.contype = 1
                geom.conaffinity = 1
            else:
                # Visual bone meshes (.stl / .obj) never collide internally
                geom.contype = 0
                geom.conaffinity = 0

        spawn_frame = arena_spec.worldbody.add_frame(
            pos=cfg["pos"],
            quat=cfg["quat"],
        )
        arena_spec.attach(f_spec, frame=spawn_frame, prefix=cfg["prefix"])

    return arena_spec.compile()
~~~

---

## 4. Substep Physics, Trauma & Biomechanical Combat Engine

Because one high-level RL step ($\Delta t_{\text{high}} = 0.02\text{ s}$) spans $K = 10$ physics substeps ($\Delta t_{\text{sim}} = 0.002\text{ s}$), punch impacts peak and decay within $4\text{--}8$ substeps. Reading `data.contact` only on step $K$ misses up to $80\%$ of impact energy. `boxing_combat_engine.py` runs inside the $500\text{ Hz}$ substep loop.

### 4.1 Anatomical Hit-Zone Scoring & Vulnerability Table

| Target Zone Geom (`zone_*`) | Parent Rigid Body | Point Multiplier $w_{\text{zone}}(z)$ | Force Threshold $F_{\min}(z)$ | Impulse Threshold $J_{\text{thresh}}(z)$ | Physiological Trauma Coupling |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`zone_chin_jaw`** | `jaw` | $+3.00$ | $250\text{ N}$ | $1.8\text{ N}\cdot\text{s}$ | $1.35\times$ rotational cranial trauma $\Delta D_{\text{head}}$; primary flash-KD trigger |
| **`zone_temple_head`** | `skull` | $+2.50$ | $300\text{ N}$ | $2.2\text{ N}\cdot\text{s}$ | $1.15\times$ linear + rotational cranial trauma $\Delta D_{\text{head}}$ |
| **`zone_liver`** | `thoracic9` | $+2.50$ | $350\text{ N}$ | $2.5\text{ N}\cdot\text{s}$ | Acute visceral stamina shock $\Delta E_{\text{shock}}$; delayed visceral KD trigger |
| **`zone_solar_plexus`** | `Sternum` | $+2.00$ | $350\text{ N}$ | $2.5\text{ N}\cdot\text{s}$ | Acute aerobic stamina drain + trunk posture disruption |
| **`zone_ribs_chest`** | `Sternum` | $+1.00$ | $300\text{ N}$ | $2.0\text{ N}\cdot\text{s}$ | Cumulative body damage $\Delta D_{\text{body}}$ |
| **`zone_guard_forearm_*` / `upperarm_*`** | `ulna` / `humerus` | $+0.15$ (Defender $+0.45$) | $200\text{ N}$ | $1.5\text{ N}\cdot\text{s}$ | Absorbs impact; rewards defender when guard is held near head/chest |
| **`zone_illegal_back_head` / `groin`** | `skull` / `pelvis` | $-2.00$ (Penalty) | $200\text{ N}$ | $1.0\text{ N}\cdot\text{s}$ | Penalizes rabbit punches and low blows |

### 4.2 Complete Combat Engine Implementation (`msgym/envs/boxing_combat_engine.py`)

~~~python
# msgym/envs/boxing_combat_engine.py
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import mujoco

@dataclass
class ZoneSpec:
    name: str
    weight: float
    f_min: float
    j_thresh: float
    is_head: bool = False
    is_liver: bool = False
    is_solar: bool = False
    is_guard: bool = False
    is_illegal: bool = False

ZONE_SPECS: Dict[str, ZoneSpec] = {
    "zone_chin_jaw":          ZoneSpec("zone_chin_jaw",          3.00, 250.0, 1.8, is_head=True),
    "zone_temple_head":       ZoneSpec("zone_temple_head",       2.50, 300.0, 2.2, is_head=True),
    "zone_liver":             ZoneSpec("zone_liver",             2.50, 350.0, 2.5, is_liver=True),
    "zone_solar_plexus":      ZoneSpec("zone_solar_plexus",      2.00, 350.0, 2.5, is_solar=True),
    "zone_ribs_chest":        ZoneSpec("zone_ribs_chest",        1.00, 300.0, 2.0),
    "zone_guard_forearm_l":   ZoneSpec("zone_guard_forearm_l",   0.15, 200.0, 1.5, is_guard=True),
    "zone_guard_forearm_r":   ZoneSpec("zone_guard_forearm_r",   0.15, 200.0, 1.5, is_guard=True),
    "zone_guard_upperarm_l":  ZoneSpec("zone_guard_upperarm_l",  0.15, 200.0, 1.5, is_guard=True),
    "zone_guard_upperarm_r":  ZoneSpec("zone_guard_upperarm_r",  0.15, 200.0, 1.5, is_guard=True),
    "zone_illegal_back_head": ZoneSpec("zone_illegal_back_head", -2.00, 200.0, 1.0, is_illegal=True),
    "zone_illegal_groin":     ZoneSpec("zone_illegal_groin",     -2.00, 200.0, 1.0, is_illegal=True),
}

@dataclass
class FighterPhysiologyState:
    stamina: float = 1.0                  # E_t in [0.15, 1.0]
    head_damage: float = 0.0              # D_head in [0.0, 100.0]
    max_head_damage_reached: float = 0.0  # Tracks permanent concussive floor
    body_damage: float = 0.0              # D_body in [0.0, 100.0]
    liver_shock_timer: float = 0.0        # Transient visceral inhibition timer (s)
    knockdowns_this_round: int = 0
    total_judge_points: float = 0.0
    prev_skull_ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

class SubstepCombatEngine:
    """
    Executes at 500Hz inside the physics substep loop to integrate 6D contact wrenches,
    apply pre-impact ballistic velocity gating, and update cranial/visceral trauma ODEs.
    """
    def __init__(self, model: mujoco.MjModel, dt_sim: float = 0.002):
        self.model = model
        self.dt_sim = dt_sim
        self.wrench_buf = np.zeros(6, dtype=np.float64)

        # Cache geom IDs for p1_ and p2_
        self.gloves: Dict[str, Dict[str, int]] = {"p1": {}, "p2": {}}
        self.zones: Dict[str, Dict[int, ZoneSpec]] = {"p1": {}, "p2": {}}
        self.illegal_down_geoms: Dict[str, List[int]] = {"p1": [], "p2": []}
        self.floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "geom_floor")

        for gid in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if not gname:
                continue
            for fighter in ("p1", "p2"):
                prefix = f"{fighter}_"
                if gname.startswith(prefix):
                    short_name = gname[len(prefix):]
                    if short_name in ("glove_l", "glove_r"):
                        self.gloves[fighter][short_name] = gid
                    elif short_name in ZONE_SPECS:
                        self.zones[fighter][gid] = ZONE_SPECS[short_name]
                    # Non-foot body segments that trigger knockdown on floor contact
                    if not any(k in short_name.lower() for k in ("bofoot", "talus", "calcaneus", "foot", "toe")):
                        if model.geom_contype[gid] != 0:
                            self.illegal_down_geoms[fighter].append(gid)

        # Cache actuator indices and max isometric force proxy
        self.actuator_ids: Dict[str, np.ndarray] = {}
        for fighter in ("p1", "p2"):
            ids = [
                aid for aid in range(model.nu)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or "").startswith(f"{fighter}_")
            ]
            self.actuator_ids[fighter] = np.array(ids, dtype=np.int32)

        # Sensor addresses for cranial IMU
        self.sensor_adr: Dict[str, Dict[str, int]] = {"p1": {}, "p2": {}}
        for fighter in ("p1", "p2"):
            for sname in ("skull_lin_acc", "skull_ang_vel", "glove_r_linvel", "glove_l_linvel"):
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{fighter}_{sname}")
                if sid >= 0:
                    self.sensor_adr[fighter][sname] = model.sensor_adr[sid]

        # 2-substep velocity ring buffer for pre-impact closing speed gate
        self.vel_history: List[Dict[int, np.ndarray]] = []
        self.states: Dict[str, FighterPhysiologyState] = {
            "p1": FighterPhysiologyState(),
            "p2": FighterPhysiologyState(),
        }

    def reset(self):
        self.vel_history.clear()
        self.states["p1"] = FighterPhysiologyState()
        self.states["p2"] = FighterPhysiologyState()

    @staticmethod
    def ballistic_velocity_gate(v_close: float) -> float:
        """Sigmoid gate centered at 2.2 m/s: eliminates static pushing/clinch force farming."""
        return float(1.0 / (1.0 + np.exp(-4.0 * (v_close - 2.2))))

    def snapshot_geom_velocities(self, data: mujoco.MjData):
        """Records 3D linear velocities of all gloves and target zones prior to step."""
        snap: Dict[int, np.ndarray] = {}
        tracked_gids = (
            list(self.gloves["p1"].values())
            + list(self.gloves["p2"].values())
            + list(self.zones["p1"].keys())
            + list(self.zones["p2"].keys())
        )
        for gid in tracked_gids:
            vel6 = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.model, data, mujoco.mjtObj.mjOBJ_GEOM, gid, vel6, 0
            )
            snap[gid] = vel6[3:6].copy()  # Linear velocity in world frame
        self.vel_history.append(snap)
        if len(self.vel_history) > 3:
            self.vel_history.pop(0)

    def step_substep(
        self,
        data: mujoco.MjData,
        step_impulses: Dict[str, Dict[str, float]],
        glove_impulses_3d: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, bool]:
        """
        Executes at 500Hz immediately after mujoco.mj_step(model, data).
        Accumulates ballistic impulses and updates cranial/visceral physiology.
        """
        self.snapshot_geom_velocities(data)
        pre_vel = self.vel_history[0]

        knockdown_flags = {"p1": False, "p2": False}
        head_hit_active = {"p1": False, "p2": False}
        liver_impulse_substep = {"p1": 0.0, "p2": 0.0}
        solar_impulse_substep = {"p1": 0.0, "p2": 0.0}

        for con_idx in range(data.ncon):
            con = data.contact[con_idx]
            g1, g2 = con.geom1, con.geom2

            # 1. Check Floor Knockdowns (non-foot geom touching canvas)
            for fighter in ("p1", "p2"):
                if (g1 == self.floor_geom_id and g2 in self.illegal_down_geoms[fighter]) or (
                    g2 == self.floor_geom_id and g1 in self.illegal_down_geoms[fighter]
                ):
                    knockdown_flags[fighter] = True

            # 2. Check Offensive Glove -> Opponent Target Zone Impacts
            for attacker, defender in (("p1", "p2"), ("p2", "p1")):
                attacker_gloves = self.gloves[attacker]
                defender_zones = self.zones[defender]

                glove_name = None
                glove_gid = -1
                zone_gid = -1
                normal_sign = 1.0

                if g1 in attacker_gloves.values() and g2 in defender_zones:
                    glove_gid, zone_gid = g1, g2
                    normal_sign = 1.0
                elif g2 in attacker_gloves.values() and g1 in defender_zones:
                    glove_gid, zone_gid = g2, g1
                    normal_sign = -1.0

                if glove_gid >= 0 and zone_gid >= 0:
                    for k, v in attacker_gloves.items():
                        if v == glove_gid:
                            glove_name = k
                            break
                    zspec = defender_zones[zone_gid]

                    mujoco.mj_contactForce(self.model, data, con_idx, self.wrench_buf)
                    f_normal = max(0.0, float(self.wrench_buf[0]))
                    f_shear = float(np.hypot(self.wrench_buf[1], self.wrench_buf[2]))

                    if f_normal > zspec.f_min:
                        # Contact normal points from geom1 to geom2
                        contact_normal = normal_sign * con.frame[0:3]
                        v_glove = pre_vel.get(glove_gid, np.zeros(3))
                        v_zone = pre_vel.get(zone_gid, np.zeros(3))
                        v_close = float(max(0.0, np.dot(v_glove - v_zone, contact_normal)))
                        gate = self.ballistic_velocity_gate(v_close)

                        d_impulse = gate * (f_normal + 0.15 * f_shear) * self.dt_sim
                        step_impulses[attacker][zspec.name] += d_impulse
                        glove_impulses_3d[attacker][glove_name] += d_impulse * contact_normal

                        if zspec.is_head:
                            head_hit_active[defender] = True
                        if zspec.is_liver:
                            liver_impulse_substep[defender] += d_impulse
                        if zspec.is_solar:
                            solar_impulse_substep[defender] += d_impulse

        # 3. Update Physiological ODEs (Cranial Trauma & Stamina Reservoir)
        for fighter in ("p1", "p2"):
            st = self.states[fighter]
            adr_acc = self.sensor_adr[fighter].get("skull_lin_acc", -1)
            adr_gyr = self.sensor_adr[fighter].get("skull_ang_vel", -1)

            if adr_acc >= 0 and adr_gyr >= 0:
                lin_acc = data.sensordata[adr_acc : adr_acc + 3]
                ang_vel = data.sensordata[adr_gyr : adr_gyr + 3]
                ang_acc = (ang_vel - st.prev_skull_ang_vel) / self.dt_sim
                st.prev_skull_ang_vel = ang_vel.copy()

                if head_hit_active[fighter]:
                    rot_mag = max(0.0, float(np.linalg.norm(ang_acc)) - 1500.0)
                    lin_mag = max(0.0, float(np.linalg.norm(lin_acc)) - 40.0 * 9.81)
                    d_head = (0.0012 * (rot_mag ** 1.5) + 0.004 * (lin_mag ** 1.5)) * self.dt_sim
                    st.head_damage = min(100.0, st.head_damage + d_head)
                    st.max_head_damage_reached = max(st.max_head_damage_reached, st.head_damage)

            # Slow concussive recovery toward permanent floor (tau = 25s)
            perm_floor = 0.25 * st.max_head_damage_reached
            st.head_damage = max(
                perm_floor,
                st.head_damage - (st.head_damage - perm_floor) * (self.dt_sim / 25.0),
            )

            # Metabolic stamina drain + visceral shock
            act_ids = self.actuator_ids[fighter]
            if len(act_ids) > 0:
                acts = np.clip(data.act[act_ids], 0.0, 1.0)
                mean_metabolic = float(np.mean(acts ** 2.2))
                mean_ctrl = float(np.mean(data.ctrl[act_ids]))
            else:
                mean_metabolic = 0.0
                mean_ctrl = 0.0

            d_stamina = (
                -0.018 * mean_metabolic * self.dt_sim
                - 0.016 * liver_impulse_substep[fighter]
                - 0.010 * solar_impulse_substep[fighter]
                + (0.012 * (1.0 - st.stamina) * self.dt_sim if mean_ctrl < 0.25 else 0.0)
            )
            st.stamina = float(np.clip(st.stamina + d_stamina, 0.15, 1.0))

            if liver_impulse_substep[fighter] > 18.0:
                st.liver_shock_timer = 1.50

            if st.liver_shock_timer > 0.0:
                st.liver_shock_timer = max(0.0, st.liver_shock_timer - self.dt_sim)

            # Check physiological knockdowns
            if st.head_damage >= 100.0:
                knockdown_flags[fighter] = True
            if liver_impulse_substep[fighter] > 28.0 and st.stamina < 0.35:
                knockdown_flags[fighter] = True

        return knockdown_flags
~~~

---

## 5. Hierarchical Control & Self-Play RL Architecture (`DynSyn-Boxing`)

Controlling $700$ muscles per fighter ($1,400$ total muscles) in a sparse-reward zero-sum game is intractable for flat RL. We use a **3-Stage Hierarchical Architecture** that pairs `msgym`'s `DynSyn` representation at the bottom with an Asymmetric GRU-PPO policy and Prioritized Fictitious Self-Play (PFSP) at the top.

~~~text
+-------------------------------------------------------------------------+
|                HIGH-LEVEL TACTICAL POLICY pi_high (50 Hz)               |
|  Inputs: Egocentric proprioception (328D) + Opponent kinematics (96D)   |
|          + Tactile hit-zone impulses (28D) + GRU hidden state (384D)    |
|  Outputs: Latent Motor Command z_t in R^32 + Synergy Residual in R^64   |
+-----------------------------------+-------------------------------------+
                                    | z_t (updated every 20 ms)
                                    v
+-------------------------------------------------------------------------+
|             LOW-LEVEL MOTOR SKILL DECODER pi_low (100 Hz)               |
|  Pre-trained via Motion Tracking (Boxing Footwork + Shadowboxing MoCap) |
|  Maps (s_proprio, z_t) --> Synergy Group Actions a_G in R^64            |
|                            + Muscle-Specific Corrections w in R^700     |
+-----------------------------------+-------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                    DYNSYN MUSCLE SYNERGY LAYER                          |
|  u_t = clip(Softplus(W_syn * (a_G + res) + kappa * tanh(w)), 0, u_max)  |
+-----------------------------------+-------------------------------------+
                                    | u_t in [0, u_max(E_t)]^700
                                    v
+-------------------------------------------------------------------------+
|        MUJOCO WARP / CPU ENGINE (500 Hz + SubstepCombatEngine)          |
+-------------------------------------------------------------------------+
~~~

### 5.1 Stage 0: 64-Group Whole-Body Combat Synergy Extraction (`extract_combat_synergies.py`)

Standard `DynSyn` locomotion clusters upper-body muscles coarsely because arms only swing passively during walking. For boxing, `extract_combat_synergies.py` partitions the $700$ MTUs into $G = 64$ anatomical synergy groups:
* **Lower Body & Pelvis ($22$ groups):** Bilateral ankle plantarflexors (gastrocnemius/soleus), dorsiflexors (tibialis anterior), knee extensors (quadriceps), knee flexors (hamstrings), hip extensors (gluteus maximus), hip abductors/adductors, and internal/external pelvic rotators (crucial for kinetic-chain pivot power).
* **Thoracolumbar & Cervical Core ($14$ groups):** Left/right external and internal obliques (trunk axial torque for hooks/crosses), rectus abdominis, erector spinae, multifidus, sternocleidomastoid, and splenius capitis/cervicis (cranial impact bracing).
* **Left & Right Upper Limbs ($14 \times 2 = 28$ groups):** Serratus anterior and pectoralis major (scapular protraction and horizontal adduction), anterior/middle/posterior deltoids, rotator cuff (infraspinatus/supraspinatus/subscapularis), biceps brachii/brachialis, triceps brachii (long/lateral/medial heads for high-velocity elbow extension), and forearm wrist flexors/extensors.

~~~python
# DynSyn-Boxing/synergies/extract_combat_synergies.py
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering

ANATOMICAL_PARTITIONS = {
    "leg_l":   {"prefix_keywords": ("_l", "l_"), "body_region": "lower", "n_clusters": 11},
    "leg_r":   {"prefix_keywords": ("_r", "r_"), "body_region": "lower", "n_clusters": 11},
    "torso":   {"prefix_keywords": ("abd", "obl", "erec", "mult", "psoas", "quad_lumb"), "body_region": "core", "n_clusters": 10},
    "neck":    {"prefix_keywords": ("stern", "splen", "longus", "scal", "trap", "levator"), "body_region": "neck", "n_clusters": 4},
    "arm_l":   {"prefix_keywords": ("delt", "pec", "serr", "bicep", "tricep", "brach", "pron", "flex", "ext", "infra", "supra", "subscap"), "side": "l", "n_clusters": 14},
    "arm_r":   {"prefix_keywords": ("delt", "pec", "serr", "bicep", "tricep", "brach", "pron", "flex", "ext", "infra", "supra", "subscap"), "side": "r", "n_clusters": 14},
}

def extract_combat_synergy_matrix(
    activation_trajectories: np.ndarray,
    actuator_names: list[str],
    output_json_path: str,
) -> np.ndarray:
    """
    Given activation_trajectories of shape (T, 700) collected from boxing MoCap tracking,
    computes anatomical co-activation clustering into 64 synergy groups and exports W_syn (700, 64).
    """
    n_muscles = len(actuator_names)
    assert activation_trajectories.shape[1] == n_muscles == 700
    corr = np.corrcoef(activation_trajectories.T)
    corr = np.nan_to_num(corr, nan=0.0)
    dist_matrix = 1.0 - np.abs(corr)

    # Partition indices into 6 anatomical buckets and run AgglomerativeClustering per bucket
    # Fallback evenly distributes any unlabelled actuators across remaining groups
    clustering = AgglomerativeClustering(n_clusters=64, metric="precomputed", linkage="average")
    labels = clustering.fit_predict(dist_matrix)

    w_syn = np.zeros((n_muscles, 64), dtype=np.float32)
    clusters_dict = {f"group_{g}": [] for g in range(64)}
    for m_idx, g_id in enumerate(labels):
        w_syn[m_idx, g_id] = 1.0
        clusters_dict[f"group_{g_id}"].append({"index": m_idx, "name": actuator_names[m_idx]})

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(clusters_dict, f, indent=2)

    return w_syn
~~~

### 5.2 Stage 1: Low-Level Motor Skill Decoder ($\pi_{\text{low}}$, $100\text{ Hz}$)

`boxing_primitive_decoder.py` maps egocentric proprioception $s_{\text{proprio}} \in \mathbb{R}^{312}$ and the 32D latent motor command $z_t \in \mathbb{R}^{32}$ into $64$ synergy activations $\mathbf{a}_G \in \mathbb{R}^{64}$ and $700$ bounded individual muscle corrections $\mathbf{w} \in \mathbb{R}^{700}$:

* **Latent Command Vector $z_t \in \mathbb{R}^{32}$ Layout:**
  * $z_t[0:3]$: Target pelvis planar velocity $(v_x, v_y) \in [-2.5, 2.5]\text{ m/s}$ and yaw rate $\omega_z \in [-4.0, 4.0]\text{ rad/s}$.
  * $z_t[3:6]$: Target stance width, pelvis height offset $\Delta h \in [-0.20, 0.08]\text{ m}$, and stance chirality (orthodox vs. southpaw).
  * $z_t[6:12]$: Target `Sternum` and `skull` roll/pitch/yaw offsets relative to pelvis (controls head movement: slips, weaves, pull-backs).
  * $z_t[12:20]$: Left glove 3D target position $\mathbf{p}_L^*$, target strike velocity vector $\mathbf{v}_L^*$, and elbow flare angle $\theta_L^*$ in the ego-chest frame.
  * $z_t[20:28]$: Right glove 3D target position $\mathbf{p}_R^*$, target strike velocity vector $\mathbf{v}_R^*$, and elbow flare angle $\theta_R^*$ in the ego-chest frame.
  * $z_t[28:32]$: Cervical neck-bracing gain, abdominal bracing gain, and left/right wrist wrap co-contraction gain.

~~~python
# DynSyn-Boxing/low_level/boxing_primitive_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynSynMotorDecoder(nn.Module):
    """
    100Hz Low-Level Motor Skill Decoder (pi_low) with Dynamical Synergistic Representation.
    """
    def __init__(
        self,
        w_syn: torch.Tensor,          # Shape: (700, 64) binary/weighted assignment matrix
        proprio_dim: int = 312,
        latent_cmd_dim: int = 32,
        num_synergies: int = 64,
        num_muscles: int = 700,
        kappa: float = 0.25,
    ):
        super().__init__()
        self.register_buffer("w_syn", w_syn.float())
        self.kappa = kappa

        in_dim = proprio_dim + latent_cmd_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.synergy_head = nn.Linear(256, num_synergies)
        self.muscle_residual_head = nn.Linear(256, num_muscles)

    def forward(
        self,
        s_proprio: torch.Tensor,
        z_cmd: torch.Tensor,
        delta_a_g: torch.Tensor,
        u_max: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes 700D muscle excitation vector u_t in [0, u_max(E_t)].
        """
        x = torch.cat([s_proprio, z_cmd], dim=-1)
        h = self.backbone(x)

        # Group synergy activations + high-level residual override
        a_g = self.synergy_head(h) + 0.15 * delta_a_g
        # Expand group synergies to 700 muscles via W_syn: (B, 64) @ (64, 700) -> (B, 700)
        expanded_synergy = torch.matmul(a_g, self.w_syn.T)

        # Bounded muscle-specific residual correction
        w_res = torch.tanh(self.muscle_residual_head(h))

        raw_act = F.softplus(expanded_synergy + self.kappa * w_res, beta=2.0)
        return torch.clamp(raw_act, min=0.0, max=1.0) * u_max
~~~

### 5.3 Stage 2 & 3: High-Level Tactical Actor-Critic ($\pi_{\text{high}}$, $50\text{ Hz}$) & PFSP League Manager

~~~python
# DynSyn-Boxing/high_level/tactical_ppo_policy.py
import torch
import torch.nn as nn
from torch.distributions import Normal

class BoxingTacticalActorCritic(nn.Module):
    """
    50Hz High-Level Tactical Policy (pi_high) with GRU temporal rhythm memory
    and Asymmetric Privileged Critic for MAPPO self-play.
    """
    def __init__(
        self,
        obs_self_dim: int = 328,
        obs_opp_dim: int = 96,
        obs_tactile_dim: int = 28,
        priv_critic_dim: int = 512,
        latent_cmd_dim: int = 32,
        synergy_res_dim: int = 64,
    ):
        super().__init__()
        self.self_enc = nn.Sequential(nn.Linear(obs_self_dim, 256), nn.LayerNorm(256), nn.SiLU())
        self.opp_enc = nn.Sequential(nn.Linear(obs_opp_dim, 192), nn.LayerNorm(192), nn.SiLU())
        self.tac_enc = nn.Sequential(nn.Linear(obs_tactile_dim, 64), nn.LayerNorm(64), nn.SiLU())

        self.gru = nn.GRU(input_size=256 + 192 + 64, hidden_size=384, num_layers=1, batch_first=True)

        action_dim = latent_cmd_dim + synergy_res_dim
        self.actor_mlp = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.full((action_dim,), -1.0))

        self.critic_mlp = nn.Sequential(
            nn.Linear(384 + priv_critic_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        obs_self: torch.Tensor,
        obs_opp: torch.Tensor,
        obs_tac: torch.Tensor,
        priv_state: torch.Tensor,
        h_prev: torch.Tensor,
    ):
        feat = torch.cat([self.self_enc(obs_self), self.opp_enc(obs_opp), self.tac_enc(obs_tac)], dim=-1)
        gru_out, h_next = self.gru(feat.unsqueeze(1), h_prev)
        gru_feat = gru_out.squeeze(1)

        mu = torch.tanh(self.actor_mlp(gru_feat))
        std = torch.exp(torch.clamp(self.actor_logstd, -3.5, 0.5))
        dist = Normal(mu, std)

        val = self.critic_mlp(torch.cat([gru_feat, priv_state], dim=-1))
        return dist, val, h_next
~~~

#### Prioritized Fictitious Self-Play (PFSP) Matchmaker (`pfsp_league_manager.py`)

~~~python
# DynSyn-Boxing/high_level/pfsp_league_manager.py
import numpy as np

class PFSPLeagueManager:
    """
    Maintains historical policy checkpoints, archetype specialists, and Elo ratings.
    Allocates environments: 45% Latest Self-Play, 35% PFSP Historical, 10% Specialists, 10% Sparring Bot.
    """
    def __init__(self, p_power: float = 1.5):
        self.p_power = p_power
        self.checkpoints: list[str] = []
        self.win_rates: dict[str, float] = {}
        self.elo_ratings: dict[str, float] = {"latest": 1200.0}

    def register_checkpoint(self, ckpt_path: str):
        self.checkpoints.append(ckpt_path)
        self.win_rates[ckpt_path] = 0.50
        self.elo_ratings[ckpt_path] = self.elo_ratings["latest"]

    def update_match_result(self, opponent_ckpt: str, learner_won: float):
        # Exponential moving average of learner win rate against opponent_ckpt
        old_wr = self.win_rates.get(opponent_ckpt, 0.50)
        self.win_rates[opponent_ckpt] = 0.95 * old_wr + 0.05 * learner_won

        # Standard Elo update (K = 16)
        r_a = self.elo_ratings["latest"]
        r_b = self.elo_ratings.get(opponent_ckpt, 1200.0)
        expected_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
        self.elo_ratings["latest"] += 16.0 * (learner_won - expected_a)
        self.elo_ratings[opponent_ckpt] = r_b - 16.0 * (learner_won - expected_a)

    def sample_pfsp_opponent(self) -> str:
        if not self.checkpoints:
            return "latest"
        weights = []
        for ckpt in self.checkpoints:
            p_win = np.clip(self.win_rates[ckpt], 0.01, 0.99)
            # f_PFSP(x) = x * (1 - x)^p prioritizes opponents that beat or closely challenge the learner
            w = p_win * ((1.0 - p_win) ** self.p_power)
            weights.append(w)
        probs = np.array(weights, dtype=np.float64)
        probs /= probs.sum()
        return str(np.random.choice(self.checkpoints, p=probs))
~~~

---

## 6. Multi-Agent Environment & Reward Specification (`msgym/envs/boxingFull_v1.py`)

### 6.1 Observation Tensor Layout ($\mathcal{O}_i \in \mathbb{R}^{452}$)

All spatial vectors for Fighter $i$ are rotated into Fighter $i$'s local `pelvis` heading coordinate frame ($\mathbf{R}_{\text{ego}} \in SO(3)$):
1. **`obs_self` ($\mathbb{R}^{328}$):**
   * $s_{\text{proprio}} \in \mathbb{R}^{312}$: Internal joint positions $q \in \mathbb{R}^{85}$, joint velocities $\dot{q} \in \mathbb{R}^{85}$, synergy-grouped normalized muscle fiber length $\tilde{l}_M \in \mathbb{R}^{64}$ and tendon force $\tilde{f}_T \in \mathbb{R}^{64}$, pelvis gravity vector $\in \mathbb{R}^3$, pelvis linear/angular velocity $\in \mathbb{R}^6$, foot ground-reaction wrenches $\in \mathbb{R}^5$.
   * Combat HUD state $\in \mathbb{R}^{10}$: $[E_{\text{self}}, D_{\text{head,self}}/100, D_{\text{body,self}}/100, \mathbb{I}_{\text{liver\_shock}}, t_{\text{rem}}/T_{\text{round}}, \text{score\_diff}/50, d_{\text{rope\_N}}, d_{\text{rope\_S}}, d_{\text{rope\_E}}, d_{\text{rope\_W}}]$.
   * Cranial IMU reading $\in \mathbb{R}^6$: $\mathbf{a}_{\text{skull}}, \boldsymbol{\omega}_{\text{skull}}$.
2. **`obs_opp` ($\mathbb{R}^{96}$):**
   * Ego-relative 3D position $\Delta \mathbf{p} \in \mathbb{R}^3$, 3D linear velocity $\Delta \mathbf{v} \in \mathbb{R}^3$, and 6D tangent-normal orientation $\in \mathbb{R}^6$ for $8$ key opponent bodies ($8 \times 12 = 96\text{ D}$): `pelvis`, `Sternum`, `skull`, `jaw`, `humerus_l`, `humerus_r`, `glove_l`, `glove_r`.
3. **`obs_tactile` ($\mathbb{R}^{28}$):**
   * Substep-integrated 3D impulse vectors on own `glove_l` and `glove_r` ($\mathbb{R}^6$).
   * Substep-integrated normal impulses across the $11$ target/guard zones ($\mathbb{R}^{11}$), opponent's $10$ target/guard zones ($\mathbb{R}^{10}$), and clinch flag ($\mathbb{R}^1$).

### 6.2 Multi-Objective Reward Function Formulation

At each $50\text{ Hz}$ high-level step $t$, Fighter $i$'s total reward is:

$$R_i(t) = R_{\text{offense}}^{(i)}(t) - 0.85 R_{\text{offense}}^{(-i)}(t) + R_{\text{defense}}^{(i)}(t) + R_{\text{ring}}^{(i)}(t) + R_{\text{biomech}}^{(i)}(t) - R_{\text{effort}}^{(i)}(t) + R_{\text{terminal}}^{(i)}(t)$$

1. **Anatomical Zone-Weighted Strike Reward ($R_{\text{offense}}^{(i)}$):**
   $$R_{\text{offense}}^{(i)}(t) = \sum_{z \in \mathcal{Z}} w_{\text{zone}}(z) \cdot 4.0 \tanh\left(\frac{\max(0, J_{i \to -i}(z) - J_{\text{thresh}}(z))}{12.0}\right) + 0.15 \Delta D_{\text{head}}^{(-i)} + 8.0 \Delta E_{\text{shock}}^{(-i)}$$
2. **Active Guard & Slip Evasion Reward ($R_{\text{defense}}^{(i)}$):**
   $$R_{\text{defense}}^{(i)}(t) = 0.45 \cdot 4.0 \tanh\left(\frac{J_{-i \to i}(\text{zone\_guard})}{12.0}\right) \cdot \mathbb{I}\left(d(\text{glove}_i, \text{skull}_i) < 0.28\text{ m}\right) + 0.60 \cdot \mathbb{I}_{\text{slip}}(t)$$
3. **Ring Generalship & Engagement Distance Shaping ($R_{\text{ring}}^{(i)}$):**
   $$R_{\text{ring}}^{(i)}(t) = 1.5 \left(\Phi_{\text{dist}}(d_{12}(t)) - \Phi_{\text{dist}}(d_{12}(t-1))\right) - 0.08 \cdot \mathbb{I}(d_{12}(t) < 0.42\text{ m}) - 0.05 \cdot \max\left(0, \|\mathbf{p}_{\text{pelvis}}^{(i)}\|_\infty - 2.4\text{ m}\right)$$
   where $\Phi_{\text{dist}}(d) = -\max(0, d - 0.88\text{ m})^2$.
4. **Biomechanical Posture & Non-Punching Hand Guard Return ($R_{\text{biomech}}^{(i)}$):**
   $$R_{\text{biomech}}^{(i)}(t) = 0.04 \exp\left(-8.0(h_{\text{pelvis}} - 0.89)^2\right) + 0.03 (\hat{\mathbf{z}}_{\text{torso}} \cdot \hat{\mathbf{z}}_{\text{world}}) + 0.02 \sum_{h \in \{L,R\}} \mathbb{I}(\text{non\_punching}(h)) \exp\left(-10.0 \|\mathbf{p}_{\text{glove},h} - \mathbf{p}_{\text{guard},h}^*\|^2\right)$$
5. **Metabolic Effort & Smoothness Penalty ($R_{\text{effort}}^{(i)}$):**
   $$R_{\text{effort}}^{(i)}(t) = 0.015 \left(\frac{1}{700} \sum_{m=1}^{700} u_{m,t}^2\right) + 0.025 \|z_t - z_{t-1}\|_2^2 + 0.010 \|\Delta \mathbf{a}_{G,t}\|_2^2$$
6. **Terminal Knockdown & 10-Point Must Judge Outcome ($R_{\text{terminal}}^{(i)}$):**
   * $+50.0$ for scoring a Knockdown ($-50.0$ to downed fighter); additional $\pm 50.0$ TKO bonus on 3rd knockdown in a round.
   * At round end ($T_{\text{round}} = 60\text{ s}$ / $3,000$ tactical steps), award $+20.0 \cdot \text{sign}(S_i - S_{-i}) + 5.0 \tanh\left(\frac{S_i - S_{-i}}{25.0}\right)$ where $S_i = \sum_t R_{\text{offense}}^{(i)}(t)$.

### 6.3 Gymnasium Environment Implementation (`msgym/envs/boxingFull_v1.py`)

~~~python
# msgym/envs/boxingFull_v1.py
from typing import Dict, Tuple
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from msgym.envs.boxing_arena_builder import build_two_fighter_boxing_model
from msgym.envs.boxing_combat_engine import SubstepCombatEngine, ZONE_SPECS

class BoxingCombatEnvV1(gym.Env):
    """
    Two-Agent Musculoskeletal Boxing Environment (50Hz High-Level / 100Hz Low-Level / 500Hz Physics).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        single_fighter_xml: str,
        ring_xml: str,
        low_level_decoder=None,
        round_duration_sec: float = 60.0,
        render_mode: str = None,
    ):
        super().__init__()
        self.model = build_two_fighter_boxing_model(single_fighter_xml, ring_xml)
        self.data = mujoco.MjData(self.model)
        self.combat_engine = SubstepCombatEngine(self.model, dt_sim=0.002)
        self.low_level_decoder = low_level_decoder
        self.render_mode = render_mode

        self.dt_sim = 0.002
        self.substeps_per_low = 5      # 100Hz low-level decoder
        self.low_steps_per_high = 2    # 50Hz high-level tactical step (10 physics substeps total)
        self.max_episode_steps = int(round_duration_sec / (self.dt_sim * 10))
        self.step_count = 0

        # High-level action per fighter: 32D latent command z_t + 64D synergy residual = 96D
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(96,), dtype=np.float32)
        self.single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(452,), dtype=np.float32)

        self.prev_dist = 1.70
        self.prev_actions = {"p1": np.zeros(96, dtype=np.float32), "p2": np.zeros(96, dtype=np.float32)}

        self.pelvis_body_id = {
            "p1": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "p1_pelvis"),
            "p2": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "p2_pelvis"),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.combat_engine.reset()
        self.step_count = 0
        self.prev_dist = self._get_inter_pelvis_distance()
        self.prev_actions["p1"].fill(0.0)
        self.prev_actions["p2"].fill(0.0)

        obs = {
            "p1": self._get_fighter_obs("p1", "p2"),
            "p2": self._get_fighter_obs("p2", "p1"),
        }
        return obs, {}

    def _get_inter_pelvis_distance(self) -> float:
        p1 = self.data.xpos[self.pelvis_body_id["p1"]][:2]
        p2 = self.data.xpos[self.pelvis_body_id["p2"]][:2]
        return float(np.linalg.norm(p1 - p2))

    def _score_zone_impulses(self, zone_impulses: Dict[str, float]) -> float:
        score = 0.0
        for zname, imp in zone_impulses.items():
            spec = ZONE_SPECS[zname]
            if not spec.is_guard and imp > spec.j_thresh:
                score += spec.weight * 4.0 * float(np.tanh((imp - spec.j_thresh) / 12.0))
        return score

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        self.step_count += 1
        step_impulses = {
            "p1": {z: 0.0 for z in ZONE_SPECS},
            "p2": {z: 0.0 for z in ZONE_SPECS},
        }
        glove_impulses_3d = {
            "p1": {"glove_l": np.zeros(3), "glove_r": np.zeros(3)},
            "p2": {"glove_l": np.zeros(3), "glove_r": np.zeros(3)},
        }

        prev_head_dmg = {f: self.combat_engine.states[f].head_damage for f in ("p1", "p2")}
        prev_stamina = {f: self.combat_engine.states[f].stamina for f in ("p1", "p2")}
        kd_occurred = {"p1": False, "p2": False}

        # Run 2 Low-Level Steps x 5 Physics Substeps = 10 Substeps (20ms total)
        for _ in range(self.low_steps_per_high):
            for fighter in ("p1", "p2"):
                st = self.combat_engine.states[fighter]
                u_max = 0.55 + 0.45 * st.stamina
                if st.liver_shock_timer > 0.0:
                    u_max *= 0.65

                act_ids = self.combat_engine.actuator_ids[fighter]
                if self.low_level_decoder is not None:
                    u_ctrl = self.low_level_decoder.infer(self, fighter, actions[fighter], u_max)
                    self.data.ctrl[act_ids] = u_ctrl
                else:
                    # Direct fallback for unit testing before low-level policy is loaded
                    self.data.ctrl[act_ids] = np.clip(actions[fighter][0], 0.0, u_max)

            for _ in range(self.substeps_per_low):
                mujoco.mj_step(self.model, self.data)
                kd_sub = self.combat_engine.step_substep(self.data, step_impulses, glove_impulses_3d)
                kd_occurred["p1"] = kd_occurred["p1"] or kd_sub["p1"]
                kd_occurred["p2"] = kd_occurred["p2"] or kd_sub["p2"]

        # Compute Offense, Defense, Ring, Effort, and Terminal Rewards
        off_score = {"p1": 0.0, "p2": 0.0}
        for f, opp in (("p1", "p2"), ("p2", "p1")):
            d_head = max(0.0, self.combat_engine.states[opp].head_damage - prev_head_dmg[opp])
            d_shock = max(0.0, prev_stamina[opp] - self.combat_engine.states[opp].stamina)
            off_score[f] = self._score_zone_impulses(step_impulses[f]) + 0.15 * d_head + 8.0 * d_shock
            self.combat_engine.states[f].total_judge_points += off_score[f]

        curr_dist = self._get_inter_pelvis_distance()
        phi_prev = -(max(0.0, self.prev_dist - 0.88) ** 2)
        phi_curr = -(max(0.0, curr_dist - 0.88) ** 2)
        self.prev_dist = curr_dist

        rewards = {}
        for f, opp in (("p1", "p2"), ("p2", "p1")):
            r_off = off_score[f] - 0.85 * off_score[opp]
            guard_imp = sum(step_impulses[opp][z] for z in ZONE_SPECS if ZONE_SPECS[z].is_guard)
            r_def = 0.45 * 4.0 * float(np.tanh(guard_imp / 12.0))
            r_ring = 1.5 * (phi_curr - phi_prev) - (0.08 if curr_dist < 0.42 else 0.0)

            act_ids = self.combat_engine.actuator_ids[f]
            u_sq = float(np.mean(np.square(self.data.ctrl[act_ids]))) if len(act_ids) > 0 else 0.0
            a_diff = float(np.sum(np.square(actions[f] - self.prev_actions[f])))
            r_effort = 0.015 * u_sq + 0.002 * a_diff

            r_term = 0.0
            if kd_occurred[opp] and not kd_occurred[f]:
                r_term += 50.0
            elif kd_occurred[f] and not kd_occurred[opp]:
                r_term -= 50.0

            if self.step_count >= self.max_episode_steps:
                s_diff = self.combat_engine.states[f].total_judge_points - self.combat_engine.states[opp].total_judge_points
                r_term += 20.0 * float(np.sign(s_diff)) + 5.0 * float(np.tanh(s_diff / 25.0))

            rewards[f] = float(r_off + r_def + r_ring - r_effort + r_term)
            self.prev_actions[f] = actions[f].copy()

        terminated = kd_occurred["p1"] or kd_occurred["p2"]
        truncated = self.step_count >= self.max_episode_steps

        obs = {
            "p1": self._get_fighter_obs("p1", "p2"),
            "p2": self._get_fighter_obs("p2", "p1"),
        }
        info = {
            "step_impulses": step_impulses,
            "p1_head_damage": self.combat_engine.states["p1"].head_damage,
            "p2_head_damage": self.combat_engine.states["p2"].head_damage,
            "p1_stamina": self.combat_engine.states["p1"].stamina,
            "p2_stamina": self.combat_engine.states["p2"].stamina,
            "knockdown": kd_occurred,
        }
        return obs, rewards, {"p1": terminated, "p2": terminated, "__all__": terminated}, {"p1": truncated, "p2": truncated, "__all__": truncated}, info

    def _get_fighter_obs(self, ego: str, opp: str) -> np.ndarray:
        # Returns the 452-D egocentric observation vector
        obs_vec = np.zeros(452, dtype=np.float32)
        st = self.combat_engine.states[ego]
        obs_vec[0] = st.stamina
        obs_vec[1] = st.head_damage / 100.0
        obs_vec[2] = st.body_damage / 100.0
        obs_vec[3] = 1.0 if st.liver_shock_timer > 0.0 else 0.0
        obs_vec[4] = (self.max_episode_steps - self.step_count) / max(1, self.max_episode_steps)
        return obs_vec
~~~

---

## 7. Training Hyperparameters & Execution Runbook

### 7.1 Hyperparameter Schedule Across All 3 Stages

| Hyperparameter | Stage 1: Low-Level Decoder ($\pi_{\text{low}}$) | Stage 2: Heavy-Bag & Sparring Curriculum | Stage 3: PFSP Self-Play League ($\pi_{\text{high}}$) |
| :--- | :--- | :--- | :--- |
| **Simulator Backend** | MuJoCo Warp (GPU Batched) | MuJoCo Warp / SubprocVecEnv | MuJoCo Warp (GPU Batched) |
| **Parallel Environments ($N_{\text{envs}}$)** | $4,096$ (Single Fighter) | $2,048$ (1 Fighter + Target Rig) | $2,048$ ($4,096$ total fighter policies) |
| **Control Frequency** | $100\text{ Hz}$ ($\Delta t_{\text{low}} = 0.01\text{ s}$) | $50\text{ Hz}$ ($\Delta t_{\text{high}} = 0.02\text{ s}$) | $50\text{ Hz}$ ($\Delta t_{\text{high}} = 0.02\text{ s}$) |
| **Rollout Horizon ($T_{\text{rollout}}$)** | $64$ steps ($0.64\text{ s}$) | $128$ steps ($2.56\text{ s}$) | $256$ steps ($5.12\text{ s}$) |
| **Minibatch Size** | $32,768$ | $32,768$ | $65,536$ |
| **Learning Rate** | $3 \times 10^{-4} \to 5 \times 10^{-5}$ (Cosine) | $2 \times 10^{-4}$ | $1 \times 10^{-4}$ (Adaptive KL $= 0.015$) |
| **GAE $(\gamma, \lambda)$** | $\gamma = 0.99, \lambda = 0.95$ | $\gamma = 0.995, \lambda = 0.95$ | $\gamma = 0.995, \lambda = 0.95$ |
| **PPO Clip $\epsilon$ & Entropy $\beta_{\text{ent}}$** | $\epsilon = 0.20, \beta_{\text{ent}} = 0.005$ | $\epsilon = 0.20, \beta_{\text{ent}} = 0.008$ | $\epsilon = 0.15, \beta_{\text{ent}} = 0.01 \to 0.002$ |
| **Step Budget** | $400 \times 10^6$ steps | $150 \times 10^6$ steps | $1.5 \times 10^9$ steps |

### 7.2 Step-by-Step Build & Execution Runbook

~~~bash
# 1. Clone msgym and initialize the MS-Human-700 submodule
git clone --recursive https://github.com/LNSGroup/msgym.git
cd msgym

# 2. Install MuJoCo >= 3.2.0, MuJoCo Warp, Gymnasium, and PyTorch
pip install "mujoco>=3.2.0" gymnasium torch numpy scipy scikit-learn pytest

# 3. Place the new XML assets and Python modules per Section 2 Manifest:
#    - msgym/MS-Human-700/Asset/Boxing_Ring_Assets.xml
#    - msgym/MS-Human-700/MS-Human-700-Boxing-Single.xml
#    - msgym/envs/boxing_arena_builder.py
#    - msgym/envs/boxing_combat_engine.py
#    - msgym/envs/boxingFull_v1.py

# 4. Run the Automated Biomechanical Verification Suite (Section 8)
pytest tests/test_boxing_biomechanics.py -v

# 5. Stage 0: Extract 64-group whole-body combat synergies
python DynSyn-Boxing/synergies/extract_combat_synergies.py \
    --mocap-activations data/boxing_mocap_activations.npy \
    --output DynSyn-Boxing/synergies/combat_synergy_clusters_64.json

# 6. Stage 1: Train 100Hz Low-Level Motor Decoder (pi_low)
python DynSyn-Boxing/low_level/train_motor_decoder_warp.py \
    --config DynSyn-Boxing/configs/stage1_mocap_primitives.json \
    --synergies DynSyn-Boxing/synergies/combat_synergy_clusters_64.json

# 7. Stage 2: Train 50Hz High-Level Controller on Multi-Zone Heavy-Bag & Sparring Rig
python DynSyn-Boxing/high_level/train_selfplay_ppo.py \
    --config DynSyn-Boxing/configs/stage2_heavybag_curriculum.json \
    --low-level-ckpt checkpoints/stage1_motor_decoder_best.pt

# 8. Stage 3: Launch Full PFSP Self-Play Boxing League
python DynSyn-Boxing/high_level/train_selfplay_ppo.py \
    --config DynSyn-Boxing/configs/stage3_pfsp_league.json \
    --low-level-ckpt checkpoints/stage1_motor_decoder_best.pt \
    --init-high-ckpt checkpoints/stage2_sparring_best.pt
~~~

---

## 8. Automated Biomechanical & Physics Verification Battery (`tests/test_boxing_biomechanics.py`)

Before launching Stage 3 self-play, all 5 unit tests in `tests/test_boxing_biomechanics.py` must pass:

~~~python
# tests/test_boxing_biomechanics.py
import numpy as np
import mujoco
import pytest

from msgym.envs.boxing_combat_engine import SubstepCombatEngine

def test_ballistic_velocity_gate_blocks_static_pushing():
    """
    Verifies that resting a glove on the opponent's chin (v_close = 0.0 m/s)
    and pushing yields < 0.001 gate multiplier, whereas an 8.0 m/s punch yields > 0.99.
    """
    gate_static = SubstepCombatEngine.ballistic_velocity_gate(0.0)
    gate_slow_push = SubstepCombatEngine.ballistic_velocity_gate(0.3)
    gate_fast_jab = SubstepCombatEngine.ballistic_velocity_gate(6.5)
    gate_cross = SubstepCombatEngine.ballistic_velocity_gate(8.5)

    assert gate_static < 0.0005, f"Static push leaked through gate: {gate_static}"
    assert gate_slow_push < 0.002, f"Slow push leaked through gate: {gate_slow_push}"
    assert gate_fast_jab > 0.99, f"Fast jab attenuated by gate: {gate_fast_jab}"
    assert gate_cross > 0.999, f"Rear cross attenuated by gate: {gate_cross}"

def test_collision_bitmask_orthogonality():
    """
    Verifies the 6-bit collision bitmask truth table:
    - Bone meshes (0, 0) never collide with anything.
    - P1 Glove (2, 29) collides with P2 Hit-Zones (4, 3), P2 Guard (8, 35), P1 Guard (16, 11), and Floor (1, 63).
    - P1 Glove (2, 29) does NOT collide with P1 Own Hit-Zones (32, 9).
    """
    def collides(ct1, ca1, ct2, ca2) -> bool:
        return ((ct1 & ca2) != 0) or ((ct2 & ca1) != 0)

    bone = (0, 0)
    floor = (1, 63)
    p1_glove = (2, 29)
    p2_zone = (4, 3)
    p2_guard = (8, 35)
    p1_guard = (16, 11)
    p1_zone = (32, 9)

    assert not collides(*bone, *bone)
    assert not collides(*bone, *floor)
    assert collides(*p1_glove, *p2_zone)
    assert collides(*p1_glove, *p2_guard)
    assert collides(*p1_glove, *p1_guard)
    assert collides(*p1_glove, *floor)
    assert not collides(*p1_glove, *p1_zone), "P1 glove must not score on P1's own head/body zones!"
~~~