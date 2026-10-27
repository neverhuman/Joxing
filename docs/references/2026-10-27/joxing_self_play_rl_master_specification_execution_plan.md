# Joxing Self-Play RL Master Specification & Next-Steps Blueprint

**Target Objective:** Train a single parameterized policy $\pi_\theta(a_t \mid o_t, h_t)$ and value critic $V_\phi(s_t, h_t)$ capable of controlling either the Blue corner or the Red corner in **Joxing**, converging via self-play to realistic, non-exploitative, two-player boxing tactics without collapsing into mutual passivity, rock-paper-scissors strategy cycling, or frame-order exploits.

---

## 1. Executive Summary: Why Naive Self-Play Fails in 2-Player Boxing

In a continuous-physics, multi-pool combat simulation like Joxing (featuring Stamina, Guard, Head, Body, and Chin health pools alongside swept-capsule collision detection), simply connecting two instances of a PPO agent ($\pi_\theta$ vs. $\pi_\theta$) to the environment fails due to five structural pathologies:

1. **Corner-Orientation Blindness (Coordinate Asymmetry):** If observations contain raw world-space coordinates or unrotated velocity vectors, a single policy $\pi_\theta$ must learn two completely separate state-action mappings for the Blue corner ($+X$ facing) and Red corner ($-X$ facing). Any slight imbalance in early rollouts biases the shared weights toward one corner.
2. **Sequential Tick-Order Bias:** If the engine updates Boxer A's kinematics, hitboxes, and stun interrupts before Boxer B's within a single simulation tick, identical actions taken on the same frame yield asymmetric outcomes. In self-play, the policy rapidly exploits this micro-advantage.
3. **The Pacifist Nash Equilibrium:** Early in training, policies have high action entropy and miss most punches. Because throwing punches drains stamina and exposes the fighter to counter-punch chin damage, the immediate expected return of attacking is negative. Without carefully structured dense-to-zero-sum reward annealing, both copies of $\pi_\theta$ learn to backpedal to the ropes and hold high guard for all rounds.
4. **On-Policy Importance Sampling Corruption:** When training against a historical pool of frozen checkpoints ($\pi_{\theta_{\text{old}}}$), feeding the opponent's transitions into the learner's PPO rollout buffer violates the importance sampling ratio $r_t(\theta) = \frac{\pi_\theta(a_t \mid o_t)}{\pi_{\theta_{\text{old}}}(a_t \mid o_t)}$, causing destructive gradient updates and critic value cancellation.
5. **Non-Transitive Rock-Paper-Scissors Cycling:** Boxing mechanics are inherently non-transitive (e.g., *Out-Boxing Jab* beats *Slow Heavy Hook*, *High-Guard Counter* beats *Jab Spam*, *Body-Hook Pressure* beats *High-Guard*). Naive $\pi_t$ vs. $\pi_t$ self-play forgets how to defend strategies from $5\times 10^6$ steps ago, cycling endlessly without improving true skill.

---

## 2. Canonical Ego-Centric Observation & Action Architecture

To allow a single neural network $\pi_\theta$ to play both Blue and Red corners seamlessly, the environment wrapper must enforce **Rotation-Symmetric Ego-Centric Invariance** (Importantly: *rotational* invariance by $\pi$ radians, **not** reflection/mirror inversion across a plane, which would illegally turn an Orthodox fighter into a Southpaw fighter in the Red corner).

### 2.1 The Ego-Centric Coordinate Frame ($\mathcal{F}_{\text{ego}}$)

For either fighter $i \in \{\text{Blue}, \text{Red}\}$ with opponent $j \neq i$:
* Let $\mathbf{p}_i = (x_i, y_i) \in \mathbb{R}^2$ be fighter $i$'s pelvis position in world space, and $\psi_i \in (-\pi, \pi]$ be fighter $i$'s facing yaw angle.
* Define the 2D rotation matrix into fighter $i$'s local facing frame:
  $$R(-\psi_i) = \begin{bmatrix} \cos\psi_i & \sin\psi_i \\ -\sin\psi_i & \cos\psi_i \end{bmatrix}$$
* Any world-space position vector $\mathbf{q}$ or velocity vector $\mathbf{v}$ is transformed into $\mathcal{F}_{\text{ego}}^{(i)}$ via:
  $$\mathbf{q}_{\text{ego}}^{(i)} = R(-\psi_i)(\mathbf{q} - \mathbf{p}_i), \quad \mathbf{v}_{\text{ego}}^{(i)} = R(-\psi_i)\mathbf{v}$$

Why rotation by $-\psi_i$ works:
* In $\mathcal{F}_{\text{ego}}^{(i)}$, $+x_{\text{ego}}$ is always **straight ahead** toward the direction fighter $i$ is facing.
* $+y_{\text{ego}}$ is always to fighter $i$'s **left side** (the lead-hand side for an Orthodox stance), and $-y_{\text{ego}}$ is always to fighter $i$'s **right side** (the rear power-hand side).
* Consequently, a Lead Jab or Left Hook has the exact same kinematic meaning, target offset, and action index whether the agent is spawned in the Blue corner ($\psi_0 = 0$) or the Red corner ($\psi_0 = \pi$).

### 2.2 Canonical Observation Vector Specification ($o_t \in \mathbb{R}^{64}$)

All observations must be normalized to $[-1.0, 1.0]$ or $[0.0, 1.0]$ using physical ring bounds ($R_{\text{ring}}$), maximum physiological pools, and maximum limb velocities ($v_{\max}$) before entering the neural network. Notice that **zero raw world coordinates** are included in the local combat features; ring awareness is represented purely via ego-centric ray/boundary distances.

| Index | Feature Group | Mathematical Definition in $\mathcal{F}_{\text{ego}}^{(i)}$ | Dim | Normalization |
| :--- | :--- | :--- | :---: | :--- |
| `0..1` | Relative Opponent Pos | $\Delta \mathbf{p}_{\text{ego}} = R(-\psi_i)(\mathbf{p}_j - \mathbf{p}_i)$ | $2$ | $/ (2 R_{\text{ring}})$ |
| `2` | Relative Distance | $d_{ij} = \|\mathbf{p}_j - \mathbf{p}_i\|_2$ | $1$ | $/ (2 R_{\text{ring}})$ |
| `3..4` | Relative Facing Angle | $(\cos(\psi_j - \psi_i), \sin(\psi_j - \psi_i))$ | $2$ | $[-1, 1]$ |
| `5..6` | Ego Linear Velocity | $\mathbf{v}_{\text{ego}}^{(i)} = R(-\psi_i)\mathbf{v}_i$ | $2$ | $/ v_{\max}$ |
| `7..8` | Opponent Rel Velocity | $\Delta \mathbf{v}_{\text{ego}} = R(-\psi_i)(\mathbf{v}_j - \mathbf{v}_i)$ | $2$ | $/ (2 v_{\max})$ |
| `9..13` | Ego Physiology Pools | $(\text{Stamina}_i, \text{Guard}_i, \text{Head}_i, \text{Body}_i, \text{Chin}_i)$ | $5$ | $/ \text{Pool}_{\max} \in [0, 1]$ |
| `14..18` | Opponent Vis. Pools | $(\text{Stamina}_j, \text{Guard}_j, \text{Head}_j, \text{Body}_j, \text{Chin}_j)$ | $5$ | $/ \text{Pool}_{\max} \in [0, 1]$ |
| `19..24` | Ego Left Glove State | $\mathbf{p}_{L,\text{ego}}^{(i)} \in \mathbb{R}^3, \mathbf{v}_{L,\text{ego}}^{(i)} \in \mathbb{R}^3$ | $6$ | $/ L_{\text{reach}}, / v_{\text{glove},\max}$ |
| `25..30` | Ego Right Glove State | $\mathbf{p}_{R,\text{ego}}^{(i)} \in \mathbb{R}^3, \mathbf{v}_{R,\text{ego}}^{(i)} \in \mathbb{R}^3$ | $6$ | $/ L_{\text{reach}}, / v_{\text{glove},\max}$ |
| `31..36` | Opp Left Glove State | $\mathbf{p}_{L,\text{ego}}^{(j)} \in \mathbb{R}^3, \mathbf{v}_{L,\text{ego}}^{(j)} \in \mathbb{R}^3$ | $6$ | $/ L_{\text{reach}}, / v_{\text{glove},\max}$ |
| `37..42` | Opp Right Glove State | $\mathbf{p}_{R,\text{ego}}^{(j)} \in \mathbb{R}^3, \mathbf{v}_{R,\text{ego}}^{(j)} \in \mathbb{R}^3$ | $6$ | $/ L_{\text{reach}}, / v_{\text{glove},\max}$ |
| `43..48` | Ego Action Phase & Lock | One-hot state (Idle, Windup, Active, Recovery, Stun, Guard) + remaining phase ticks | $6$ | $[0, 1]$ |
| `49..54` | Opp Action Telegraph | Opponent phase one-hot + current punch type category + normalized windup progress | $6$ | $[0, 1]$ |
| `55..58` | Ego Ring-Rope Raycasts | Distance to ropes along $(+x_{\text{ego}}, -x_{\text{ego}}, +y_{\text{ego}}, -y_{\text{ego}})$ | $4$ | $/ (2 R_{\text{ring}}) \in [0, 1]$ |
| `59..60` | Ring Center Vector | $\mathbf{c}_{\text{ego}}^{(i)} = R(-\psi_i)(-\mathbf{p}_i)$ | $2$ | $/ R_{\text{ring}}$ |
| `61..63` | Bout Clock & Score Diff | $(\text{RoundTimeRem}, \text{RoundIdx}/3, \tanh(\Delta \text{CompuBoxScore}/20))$ | $3$ | $[-1, 1]$ |

> **Design Note on Critic Privileged Information:** If partial observability is desired later (e.g., hiding exact opponent `Chin` or `Stamina` pools from the actor $\pi_\theta$), replace indices `14..18` in the actor observation with noisy physiological cues (breathing rate, guard droop), while keeping the exact ground-truth pools in the **Asymmetric Critic** input $s_t$. During initial self-play bring-up, keep `14..18` fully visible to verify convergence first.

### 2.3 Canonical Action Space & World Projection

Use a **Factorized Multi-Discrete Action Space** $\mathcal{A} = \mathcal{A}_{\text{move}} \times \mathcal{A}_{\text{combat}}$ so movement footwork and upper-body combat actions compose cleanly without combinatorial explosion:

1. **Movement Head ($\mathcal{A}_{\text{move}} \in \{0, \dots, 8\}$):**
   * $0$: Stand / Plant Feet (bonus to punch power and guard stability)
   * $1..8$: 8-way local directional step in $\mathcal{F}_{\text{ego}}^{(i)}$ (Forward $+x_{\text{ego}}$, Forward-Left, Step-Left $+y_{\text{ego}}$, Back-Left, Backpedal $-x_{\text{ego}}$, Back-Right, Step-Right $-y_{\text{ego}}$, Forward-Right)
   * **World-Space Conversion:** Given local step direction vector $\mathbf{d}_{\text{local}} \in \mathbb{R}^2$, the world-space impulse applied by the physics engine is:
     $$\mathbf{F}_{\text{move, world}}^{(i)} = R(+\psi_i)\mathbf{d}_{\text{local}}$$
2. **Combat & Defense Head ($\mathcal{A}_{\text{combat}} \in \{0, \dots, 11\}$):**
   * $0$: `NONE` (Relax hands / fastest stamina recovery)
   * $1$: `HIGH_GUARD` (Protects head capsule; decays `guard_pool` on impact)
   * $2$: `LOW_GUARD` (Protects body capsule; exposes head)
   * $3$: `PARRY_TIMED` (Short active window; high frame advantage if timed on contact, recovery lock if whiffed)
   * $4$: `SLIP_LEFT` (Head capsule lateral offset $+y_{\text{ego}}$)
   * $5$: `SLIP_RIGHT` (Head capsule lateral offset $-y_{\text{ego}}$)
   * $6$: `LEAD_JAB_HEAD` (Fast, low stamina cost, straight $+x_{\text{ego}}$)
   * $7$: `LEAD_JAB_BODY` (Fast, low stamina cost, angled down)
   * $8$: `REAR_CROSS_HEAD` (Medium windup, straight high-velocity power shot)
   * $9$: `LEAD_HOOK_HEAD` (Arcing lateral trajectory; bypasses narrow center guard)
   * $10$: `LEAD_HOOK_BODY` (Targets liver/ribs; drains opponent stamina pool)
   * $11$: `REAR_UPPERCUT_HEAD` (Short range; punishes forward lean/low guard)

---

## 3. Simultaneous Tick Resolution & Double-Buffered Physics

To pass `T12` (Simultaneous Trade Parity), the simulation engine must never execute `update_boxer(A)` followed immediately by `resolve_hits(A -> B)` before `B` has moved.

### 3.1 Required 4-Stage Deterministic Physics Tick

Every environment step (or physics sub-step $\Delta t$) must execute in four strictly separated phases:

```
+-------------------------------------------------------------------------+
| PHASE 1: Pre-Tick State Snapshot (Immutable Reference S_t)              |
|   Read current positions, velocities, timers, and pools of Blue & Red.  |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
| PHASE 2: Independent Kinematics & Swept Capsule Generation              |
|   Advance Blue limbs/pelvis from S_t -> S'_{t+1}(Blue)                  |
|   Advance Red limbs/pelvis from S_t  -> S'_{t+1}(Red)                   |
|   Construct swept glove segments [p_glove(t), p_glove(t + dt)] for both |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
| PHASE 3: Symmetric Contact Detection (Read-Only on S'_{t+1})            |
|   Evaluate Blue glove segments against Red head/body/guard capsules     |
|   Evaluate Red glove segments against Blue head/body/guard capsules     |
|   Compute ContactEvent(Blue->Red) and ContactEvent(Red->Blue)           |
|   (Neither event cancels the other on the same tick!)                   |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
| PHASE 4: Simultaneous Impulse, Pool Delta & Stun Commit                 |
|   Apply damage, stamina drain, chin decay, guard decay, and knockback   |
|   impulses to Blue and Red simultaneously.                              |
+-------------------------------------------------------------------------+
```

If both fighters throw a frame-identical Cross from symmetric positions, both `ContactEvent(Blue->Red)` and `ContactEvent(Red->Blue)` register, both fighters take identical damage and knockback, and if both breach the knockdown threshold on the same tick, a **Double Knockdown** state is triggered rather than awarding an arbitrary win to Boxer 0.

---

## 4. Invalid Action Masking & Commitment Lock Specification

In physics-based boxing, an agent cannot initiate a `REAR_UPPERCUT_HEAD` while:
1. Currently in the `Windup`, `Active`, or `Recovery` frames of a previous punch,
2. Locked in `hit_stun` or `guard_break_stun` frames ($t_{\text{stun}} > 0$), or
3. Stamina is below the minimum activation threshold ($\text{Stamina}_i < C_{\text{stamina}}(a)$).

### 4.1 Why Silent Engine Overrides Break PPO
If the policy network samples $a_{\text{combat}} = 8$ (`REAR_CROSS_HEAD`) during hit-stun, and the physics engine silently overrides it to $0$ (`NONE`), the rollout buffer records $(o_t, a_{\text{combat}}=8, \log \pi_\theta(8 \mid o_t), r_t)$. During PPO optimization, the policy attributes the negative reward of getting comboed during hit-stun to `REAR_CROSS_HEAD`, corrupting both the policy gradient and the entropy bonus.

### 4.2 Mathematical Formulation of Multi-Discrete Logit Masking
For each step $t$, the environment must return two boolean mask vectors alongside $o_t$:
$$m_t^{\text{move}} \in \{0, 1\}^{9}, \quad m_t^{\text{combat}} \in \{0, 1\}^{12}$$

Given raw unnormalized logits $z_t \in \mathbb{R}^{|\mathcal{A}_k|}$ from policy head $k \in \{\text{move}, \text{combat}\}$, compute the masked logits $\tilde{z}_t$:
$$\tilde{z}_{t, j} = \begin{cases} z_{t, j} & \text{if } m_{t, j} = 1 \\ -10^8 & \text{if } m_{t, j} = 0 \end{cases}$$

Then form the categorical distribution over $\tilde{z}_t$:
$$\pi_\theta(a_j \mid o_t) = \frac{\exp(\tilde{z}_{t, j})}{\sum_{l=1}^{|\mathcal{A}_k|} \exp(\tilde{z}_{t, l})}$$

**Crucial Invariant:** At least one action per head (specifically $a_{\text{move}} = 0$ and $a_{\text{combat}} = 0$) must **always** have $m_{t, 0} = 1$ so the denominator is strictly positive and $\log \pi_\theta(a \mid o_t)$ never produces `NaN`.

---

## 5. Two-Timescale Reward Architecture & Annealing Schedule

To bridge the gap between early exploration (learning to walk into range and throw punches) and late-stage strategic mastery (winning rounds on the CompuBox 10-Point Must ledger or via KO without getting countered), use a **Two-Timescale Annealed Reward Function**.

### 5.1 Total Step Reward Decomposition
At global training step $T_{\text{global}}$, the reward received by fighter $i$ (with opponent $j$) at tick $t$ is:
$$r_t^{(i)} = \alpha(T_{\text{global}}) \cdot r_{t, \text{dense}}^{(i)} + r_{t, \text{zero-sum}}^{(i)}$$

where the annealing coefficient $\alpha(T_{\text{global}})$ decays linearly from $1.0$ to a small residual floor $0.05$ over the first $T_{\text{anneal}} = 40\times 10^6$ environment steps:
$$\alpha(T_{\text{global}}) = \max\left(0.05, \; 1.0 - 0.95 \frac{T_{\text{global}}}{T_{\text{anneal}}}\right)$$

### 5.2 Component 1: Early-Stage Dense Shaping Reward ($r_{t, \text{dense}}^{(i)}$)
Designed solely to bootstrap locomotion, range finding, and punch mechanics during the first $10\times 10^6$ steps:

$$r_{t, \text{dense}}^{(i)} = w_{\text{dist}} \cdot \Phi_{\text{range}}(d_{ij}) + w_{\text{dmg}} \left(\Delta D_{\text{dealt}}^{(i \rightarrow j)} - 0.8 \Delta D_{\text{taken}}^{(j \rightarrow i)}\right) - w_{\text{whiff}} \cdot \mathbb{I}_{\text{whiff}}^{(i)}$$

* **Pocket Range Potential ($\Phi_{\text{range}}$):** Encourages closing distance from across the ring to striking range ($d_{\text{target}} = 0.85\text{ m}$), implemented as a potential-based shaping difference so agents cannot farm reward by oscillating back and forth:
  $$\Phi_{\text{range}}(s_t, s_{t+1}) = \gamma \phi(d_{ij, t+1}) - \phi(d_{ij, t}), \quad \text{where } \phi(d) = -\max(0, \; d - d_{\text{target}})$$
  Because $\Phi_{\text{range}}$ is a potential-based shaping function ($\gamma \phi(s') - \phi(s)$), it is mathematically guaranteed not to alter the optimal Nash policy once agents are in range.
* **Asymmetric Early Damage Coefficient ($1.0$ dealt vs. $-0.8$ taken):** Early on, if damage taken is weighted equally ($-1.0$) plus stamina tax, random policies prefer never throwing a punch. Weighting dealt damage slightly higher ($1.0$ vs. $0.8$) inside $r_{t, \text{dense}}$ breaks the initial pacifist barrier, and vanishes as $\alpha(T_{\text{global}}) \rightarrow 0.05$.

### 5.3 Component 2: Strictly Zero-Sum Competitive Reward ($r_{t, \text{zero-sum}}^{(i)}$)
For every tick $t$, this component satisfies the exact zero-sum invariant:
$$r_{t, \text{zero-sum}}^{(\text{Blue})} + r_{t, \text{zero-sum}}^{(\text{Red})} = 0$$

It consists of four zero-sum competitive channels:
1. **Zero-Sum Net Clean Damage Exchange (per tick):**
   $$r_{t, \text{exchange}}^{(i)} = c_{\text{head}}\left(\Delta H_j - \Delta H_i\right) + c_{\text{body}}\left(\Delta B_j - \Delta B_i\right) + c_{\text{guard}}\left(\Delta G_j^{\text{broken}} - \Delta G_i^{\text{broken}}\right)$$
2. **Zero-Sum Ring Generalship & Anti-Cornering Differential (per tick):**
   If both fighters are passive (no clean punches landed in the last $3.0$ seconds), the fighter closer to the center of the ring receives a micro zero-sum transfer from the fighter backing up against the ropes:
   $$r_{t, \text{ring}}^{(i)} = c_{\text{ring}} \cdot \mathbb{I}_{\text{stalling}} \cdot \left(\|\mathbf{p}_j\|_2 - \|\mathbf{p}_i\|_2\right)$$
   *Why this is vital:* It makes the "backpedal to the ropes and hold guard" pacifist equilibrium strictly losing ($<0$) for the retreating fighter and strictly winning ($>0$) for the fighter taking the center of the ring.
3. **End-of-Round CompuBox Judge Ledger (at Round Termination $t \in \{T_{R1}, T_{R2}, T_{R3}\}$):**
   Using the 10-Point Must System score $(S_{\text{Blue}}, S_{\text{Red}}) \in \{(10, 9), (10, 8), (10, 7), (10, 10)\}$:
   $$r_{\text{round}}^{(i)} = c_{\text{judge}} \cdot \left(S_i - S_j\right)$$
4. **Knockdown & KO Terminal Payouts:**
   * Knockdown scored: $+2.0$ for scorer, $-2.0$ for knocked-down fighter.
   * Terminal KO / TKO / Decision Win: $+10.0$ for winner, $-10.0$ for loser.

---

## 6. Recurrent PPO Pipeline & PFSP League Architecture

### 6.1 Dual-Agent Rollout Isolation (Learner vs. Opponent)

In vectorized self-play across $N_{\text{envs}}$ parallel rings, **never** mix Opponent transitions into the PPO optimization buffer when Opponent weights differ from Learner weights:

* **Case A: Historical / Anchor Matchup ($\pi_\theta$ vs. $\pi_{\text{frozen}}$):**
  * Randomly assign Learner ($\pi_\theta$) to Blue in $50\%$ of environments and Red in $50\%$ of environments at the start of every episode.
  * Step both agents in the ring.
  * Record **only** the Learner's transitions $(o_t^{\text{learner}}, h_t^{\text{learner}}, a_t^{\text{learner}}, m_t^{\text{learner}}, r_t^{\text{learner}}, d_t, V_\phi(s_t^{\text{learner}}))$ into the PPO rollout buffer.
* **Case B: Contemporary Mirror Matchup ($\pi_\theta$ vs. $\pi_\theta$):**
  * Because both corners use the exact same weights $\theta_{\text{current}}$, both streams are strictly on-policy ($r_t(\theta) = 1.0$ at the start of PPO epoch 1).
  * You may push both Blue and Red trajectories into the rollout buffer as two independent trajectory columns, doubling sample efficiency—**provided** `T11` (Symmetry) and `T16` (Zero-Sum) pass.

### 6.2 Recurrent Hidden State ($h_t, c_t$) Management

Boxing requires short-horizon memory ($2\text{--}4\text{ seconds}$) to track opponent combo patterns, guard fatigue, and retreat tendencies. Use a 1-layer GRU or LSTM ($d_{\text{hidden}} = 128$ or $256$) placed after the MLP feature extractor and before the Actor/Critic heads.

**Three Mandatory Recurrent Invariants:**
1. Maintain four separate state tensors across the vectorized environment batch $N$: `h_learner_actor`, `h_learner_critic`, `h_opp_actor`, `h_opp_critic` of shape `[1, N, d_hidden]`.
2. When environment $e \in \{0, \dots, N-1\}$ terminates (`done[e] == True`), zero out `h_*[:, e, :] = 0.0` before computing the next step's action on the auto-reset observation.
3. During PPO minibatch updates, slice rollouts into contiguous sequence chunks of length $L_{\text{seq}} = 16$ or $32$ ticks, storing the initial hidden state $h_{t_0}$ at the start of each chunk and masking out recurrent carry across episode boundaries using $(1.0 - \text{done}_{t-1})$.

### 6.3 Prioritized Fictitious Self-Play (PFSP) Matchmaking

To eliminate Rock-Paper-Scissors cycling, implement a 3-bucket opponent sampler for the non-learner corner in each parallel environment:

$$\text{Opponent Distribution} = \begin{cases} 50\% & \text{Current Self } (\pi_\theta) \text{ or Latest Snapshot } (\pi_{\theta_{\text{latest}}}) \\ 35\% & \text{Historical Checkpoint Pool } \{\pi_1, \pi_2, \dots, \pi_K\} \text{ via PFSP} \\ 15\% & \text{Fixed Heuristic Anchor Bots } \{\text{Swarmer}, \text{OutBoxer}, \text{TurtleCounter}\} \end{cases}$$

**PFSP Sampling Probability Formula:**
For each checkpoint $k \in \{1, \dots, K\}$ in the historical pool, track the Learner's exponentially weighted moving average (EWMA) win rate $w_k \in [0, 1]$ against checkpoint $k$. Sample checkpoint $k$ with probability:
$$P(\text{select } k) = \frac{f(w_k)}{\sum_{m=1}^{K} f(w_m)}, \quad \text{where } f(w) = (1 - w)^p \text{ with } p = 2.0$$

* If the Learner crushes an old checkpoint ($w_k = 0.98$), $f(w_k) = 0.0004$ (rarely sampled, but never completely zeroed out so catastrophic forgetting is immediately detected).
* If the Learner struggles against a specific historical style ($w_k = 0.35$), $f(w_k) = 0.4225$ (heavily oversampled until the Learner adapts).

**Checkpoint Promotion Gate:**
Only add a new snapshot $\pi_{\theta_{\text{new}}}$ to the historical pool when:
1. At least $250,000$ environment steps have elapsed since the last snapshot, **AND**
2. The Learner's aggregate win rate against the Fixed Heuristic Anchor Bots is $\ge 0.75$ and its average win rate across the active PFSP pool is $\ge 0.58$.

---

## 7. Concrete Verification Test Suite (`T11`--`T18`)

Before launching long training runs, implement these eight deterministic automated tests (`tests/test_self_play_readiness.py` and corresponding Rust engine unit tests). **Every single test below must pass 100%.**

### `T11` — Mirror-State Ego-Centric Equivariance Test
* **Objective:** Prove that a single neural network sees identical observation tensors and produces symmetric world actions when placed in symmetric Blue and Red states.
* **Implementation:**
```python
def test_t11_mirror_state_equivariance(env, policy_net):
    # State 1: Blue at (-1.2, +0.3) facing +X (yaw=0), Red at (+1.2, -0.3) facing -X (yaw=pi)
    state_normal = create_symmetric_test_state(
        blue_pos=(-1.2, 0.3), blue_yaw=0.0,
        red_pos=(1.2, -0.3),  red_yaw=math.pi,
        blue_pools=(0.8, 0.9, 1.0, 0.7, 1.0),
        red_pools=(0.6, 0.5, 0.8, 0.9, 0.85),
    )
    obs_blue_1, mask_blue_1 = env.get_ego_obs_and_mask(state_normal, corner="blue")

    # State 2: Swap Blue and Red physical states & physiological pools completely
    state_swapped = create_symmetric_test_state(
        blue_pos=(-1.2, 0.3), blue_yaw=0.0,
        red_pos=(1.2, -0.3),  red_yaw=math.pi,
        blue_pools=(0.6, 0.5, 0.8, 0.9, 0.85),
        red_pools=(0.8, 0.9, 1.0, 0.7, 1.0),
    )
    # Rotate entire world by pi radians so Red is now in Blue's exact relative situation
    state_rotated = rotate_world_state_by_pi(state_swapped)
    obs_red_2, mask_red_2 = env.get_ego_obs_and_mask(state_rotated, corner="red")

    # Assertions
    np.testing.assert_allclose(obs_blue_1, obs_red_2, atol=1e-6)
    np.testing.assert_array_equal(mask_blue_1, mask_red_2)

    # Verify policy logits match identically
    logits_blue = policy_net.forward_logits(obs_blue_1, mask_blue_1)
    logits_red = policy_net.forward_logits(obs_red_2, mask_red_2)
    np.testing.assert_allclose(logits_blue, logits_red, atol=1e-6)
```

### `T12` — Simultaneous Trade & Tick-Parity Test
* **Objective:** Prove zero first-mover advantage when both fighters execute identical actions on the same frame.
* **Implementation:**
  1. Spawn Blue at $(-0.75, 0.0), \psi = 0$ and Red at $(+0.75, 0.0), \psi = \pi$ with identical pools.
  2. Feed identical sequence of actions: $3$ ticks of `STEP_FORWARD` followed by `REAR_CROSS_HEAD` on tick $4$, then $15$ ticks of `NONE`.
  3. Assert at every tick $t \in [0, 20]$:
     * $\|\text{Pools}_{\text{Blue}}(t) - \text{Pools}_{\text{Red}}(t)\|_\infty < 10^{-6}$
     * $x_{\text{Blue}}(t) == -x_{\text{Red}}(t)$ and $y_{\text{Blue}}(t) == -y_{\text{Red}}(t)$
     * Both fighters register exact same `hit_stun` frames and CompuBox landed punch counts.

### `T13` — Invalid-Action Logit Masking & Stamina Starvation Test
* **Objective:** Ensure illegal actions have zero probability of being sampled and never produce `NaN` gradients.
* **Implementation:**
  1. Force fighter state to `stamina = 0.01` (below minimum punch cost) and `stun_frames = 5`.
  2. Query `obs, mask = env.get_ego_obs_and_mask()`.
  3. Assert `mask["combat"][6:] == 0` (all offensive punches masked out) and `mask["combat"][0] == 1` (`NONE` stays legal).
  4. Sample $10,000$ actions from `MaskedCategorical(logits=random_logits, mask=mask)` and assert zero offensive punches are sampled and `dist.entropy()` is finite.

### `T14` — Static Shell-Guard & Parry Spam Exploitability Test
* **Objective:** Verify that defensive mechanics cannot be abused as an invincible stall strategy.
* **Implementation:**
  1. Hardcode Red to hold `HIGH_GUARD` continuously (`a_combat = 1`).
  2. Script Blue to alternate `LEAD_JAB_HEAD`, `REAR_CROSS_HEAD`, and `LEAD_HOOK_BODY`.
  3. Assert that within $\le 12$ seconds of sustained pressure, Red's `guard_pool` reaches $0.0$ (triggering `guard_break_stun`) OR Red's `body_pool` and `stamina_pool` drop below $50\%$ from unblocked body hooks.

### `T15` — Anti-Pacifism & Ring-Center Gradient Test
* **Objective:** Prove that passive backpedaling yields a strictly negative return while taking the center of the ring yields a positive return.
* **Implementation:**
  1. Simulate a full 3-round bout where Blue steps toward $(0, 0)$ and holds position without punching, while Red backpedals to the ropes and holds `HIGH_GUARD` without punching.
  2. Assert that cumulative bout return $G_{\text{Blue}} > 0$ and $G_{\text{Red}} < 0$, and that the CompuBox judges award all 3 rounds ($10\text{--}9$) to Blue on Ring Generalship.

### `T16` — Pure Zero-Sum Ledger Conservation Audit
* **Objective:** Prove that when $\alpha(T_{\text{global}}) = 0.0$, the game is strictly zero-sum across all possible trajectories (preventing collusive point farming).
* **Implementation:**
  1. Set `reward_anneal_alpha = 0.0`.
  2. Run $500$ full episodes using random valid actions, knockdowns, flash KOs, and 3-round decisions.
  3. Assert at every single step $t$ of every episode:
     $$\left| r_t^{(\text{Blue})} + r_t^{(\text{Red})} \right| < 10^{-7}$$

### `T17` — Vectorized Recurrent State Isolation & Mid-Batch Reset Test
* **Objective:** Verify that an episode reset in Environment $k$ zeroes out only Environment $k$'s LSTM/GRU hidden state and never leaks across batch indices or between Learner and Opponent.
* **Implementation:**
  1. Instantiate `VecEnv(num_envs=8)` and a recurrent policy `RecurrentActorCritic`.
  2. Force Environment $3$ to trigger a KO on step $10$ while Environments $0, 1, 2, 4, 5, 6, 7$ continue normally.
  3. Verify that on step $11$, the recurrent hidden state passed into `RecurrentActorCritic` has `h[:, 3, :] == 0.0` while `h[:, i, :] != 0.0` for all $i \neq 3$.
  4. Verify that gradients backpropagated through a PPO sequence chunk with `done[t] = True` at step $10$ have zero gradient flow from step $11$ back into step $9$'s hidden state ($\frac{\partial L_{t=11}}{\partial h_{t=9}} = 0$).

### `T18` — Non-Transitive Anchor Bot Convergence Smoke Test
* **Objective:** Verify that the PPO + PFSP pipeline can learn to beat three distinct boxing archetypes simultaneously without forgetting.
* **Implementation:**
  Build three lightweight deterministic Anchor Bots (under $80$ lines of code each):
  * `AnchorBot_Swarmer`: Steps forward constantly; throws `LEAD_HOOK_BODY` and `REAR_UPPERCUT_HEAD` whenever $d_{ij} < 0.75\text{ m}$.
  * `AnchorBot_OutBoxer`: Maintains distance $d_{ij} \in [0.95\text{ m}, 1.15\text{ m}]$; throws `LEAD_JAB_HEAD` and steps laterally whenever opponent approaches.
  * `AnchorBot_TurtleCounter`: Holds `HIGH_GUARD`; waits for opponent to enter punch `Recovery` frames, then throws `REAR_CROSS_HEAD`.
  * **Gate Assertion:** Train $\pi_\theta$ against the uniform mixture of these three bots for $5\times 10^6$ steps; assert win rate $\ge 85\%$ against **all three** bots simultaneously.

---

## 8. Reference Implementation Blueprint (Python / PyTorch)

Below is the canonical structure for the Self-Play Wrapper, Masked Recurrent Actor-Critic, and PFSP Matchmaker to integrate directly with Joxing's training pipeline:

```python
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MaskedMultiDiscretePolicy(nn.Module):
    """Recurrent Actor-Critic with strict logit masking for Joxing self-play."""

    def __init__(self, obs_dim: int = 64, hidden_dim: int = 256, move_dim: int = 9, combat_dim: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)

        self.move_head = nn.Linear(hidden_dim, move_dim)
        self.combat_head = nn.Linear(hidden_dim, combat_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Ensure at least action 0 (idle/none) is valid to prevent all -inf
        safe_mask = mask.clone()
        invalid_rows = (safe_mask.sum(dim=-1) == 0)
        if invalid_rows.any():
            safe_mask[invalid_rows, 0] = True
        return logits.masked_fill(~safe_mask, -1e8)

    def forward_step(
        self,
        obs: torch.Tensor,          # [N, obs_dim]
        h_in: torch.Tensor,         # [1, N, hidden_dim]
        move_mask: torch.Tensor,    # [N, 9] bool
        combat_mask: torch.Tensor,  # [N, 12] bool
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(obs).unsqueeze(0)  # [1, N, hidden_dim]
        gru_out, h_out = self.gru(x, h_in)
        feat = gru_out.squeeze(0)           # [N, hidden_dim]

        move_logits = self._apply_mask(self.move_head(feat), move_mask)
        combat_logits = self._apply_mask(self.combat_head(feat), combat_mask)

        move_dist = Categorical(logits=move_logits)
        combat_dist = Categorical(logits=combat_logits)

        if deterministic:
            a_move = torch.argmax(move_logits, dim=-1)
            a_combat = torch.argmax(combat_logits, dim=-1)
        else:
            a_move = move_dist.sample()
            a_combat = combat_dist.sample()

        actions = torch.stack([a_move, a_combat], dim=-1)  # [N, 2]
        log_prob = move_dist.log_prob(a_move) + combat_dist.log_prob(a_combat)
        entropy = move_dist.entropy() + combat_dist.entropy()
        value = self.critic_head(feat).squeeze(-1)

        return actions, log_prob, entropy, value, h_out


@dataclass
class OpponentCheckpoint:
    checkpoint_id: str
    step_created: int
    state_dict: Dict[str, torch.Tensor]
    elo: float = 1200.0
    learner_win_rate_ewma: float = 0.50
    matches_played: int = 0


class PFSPLeagueManager:
    """Manages historical checkpoints, Elo ratings, and PFSP opponent sampling."""

    def __init__(self, pfsp_power: float = 2.0, ewma_alpha: float = 0.05):
        self.pfsp_power = pfsp_power
        self.ewma_alpha = ewma_alpha
        self.pool: List[OpponentCheckpoint] = []
        self.learner_elo: float = 1200.0
        self.anchor_win_rates: Dict[str, float] = {
            "Swarmer": 0.0,
            "OutBoxer": 0.0,
            "TurtleCounter": 0.0,
        }

    def sample_matchup_type(self) -> str:
        r = np.random.rand()
        if r < 0.50 or len(self.pool) == 0:
            return "CURRENT_SELF"
        elif r < 0.85:
            return "PFSP_HISTORICAL"
        else:
            return "ANCHOR_BOT"

    def sample_historical_checkpoint(self) -> OpponentCheckpoint:
        weights = np.array(
            [max(1e-3, (1.0 - ckpt.learner_win_rate_ewma) ** self.pfsp_power) for ckpt in self.pool],
            dtype=np.float64,
        )
        probs = weights / weights.sum()
        idx = np.random.choice(len(self.pool), p=probs)
        return self.pool[idx]

    def record_result(self, ckpt: OpponentCheckpoint, learner_score: float) -> None:
        """learner_score: 1.0 for win, 0.5 for draw, 0.0 for loss."""
        ckpt.matches_played += 1
        ckpt.learner_win_rate_ewma = (
            (1.0 - self.ewma_alpha) * ckpt.learner_win_rate_ewma + self.ewma_alpha * learner_score
        )
        # Standard Elo update (K = 16)
        expected_learner = 1.0 / (1.0 + 10.0 ** ((ckpt.elo - self.learner_elo) / 400.0))
        delta = 16.0 * (learner_score - expected_learner)
        self.learner_elo += delta
        ckpt.elo -= delta

    def should_promote_checkpoint(self, steps_since_last: int, min_steps: int = 250_000) -> bool:
        if steps_since_last < min_steps:
            return False
        min_anchor_wr = min(self.anchor_win_rates.values())
        avg_pool_wr = np.mean([c.learner_win_rate_ewma for c in self.pool]) if self.pool else 1.0
        return (min_anchor_wr >= 0.75) and (avg_pool_wr >= 0.58)
```

---

## 9. Step-by-Step Execution Roadmap & Hyperparameters

Execute the work in four strictly gated milestones. Do not proceed to the next stage until all exit criteria of the current stage are green.

### Stage 1: Engine Symmetry & Determinism Hardening (Days 1--2)
* [ ] Implement the 4-stage simultaneous physics tick (Phase 1 Snapshot $\rightarrow$ Phase 2 Kinematics $\rightarrow$ Phase 3 Symmetric Swept Contact $\rightarrow$ Phase 4 Simultaneous Commit).
* [ ] Build the canonical ego-centric wrapper (`get_ego_obs_and_mask` and `project_ego_action_to_world`).
* [ ] Implement and pass **`T11`** (Mirror-State Equivariance), **`T12`** (Simultaneous Trade Parity), **`T13`** (Action Masking), and **`T16`** (Zero-Sum Ledger Conservation).

### Stage 2: Anchor Bot Curriculum & Single-Agent Sanity Check (Days 3--4)
* [ ] Implement the three heuristic Anchor Bots (`Swarmer`, `OutBoxer`, `TurtleCounter`).
* [ ] Implement and pass **`T14`** (Guard Break Exploitability), **`T15`** (Anti-Pacifism Ring Gradient), and **`T17`** (Vectorized Recurrent Reset Isolation).
* [ ] Train $\pi_\theta$ from scratch against the 3 Anchor Bots (50% in Blue corner, 50% in Red corner).
* [ ] **Exit Gate (`T18`):** Achieve $\ge 85\%$ win rate against all 3 Anchor Bots within $5\times 10^6$ environment steps, with $< 2\%$ win-rate discrepancy between Blue and Red corner spawns.

### Stage 3: PFSP Self-Play Bring-Up & Reward Annealing (Days 5--8)
* [ ] Enable the `PFSPLeagueManager` with $50\%$ Current Self, $35\%$ PFSP Historical Pool, and $15\%$ Anchor Bots.
* [ ] Enable linear annealing of $\alpha(T_{\text{global}}): 1.0 \rightarrow 0.05$ over $40\times 10^6$ steps.
* [ ] Log separated TensorBoard/W&B metrics:
  * `eval/win_rate_vs_swarmer`, `eval/win_rate_vs_outboxer`, `eval/win_rate_vs_turtle`
  * `eval/blue_ vs_red_corner_win_parity` (must stay within $[0.48, 0.52]$ in mirror matches)
  * `combat/punches_thrown_per_round`, `combat/clean_land_percentage`, `combat/ko_finish_rate`
  * `combat/guard_break_frequency`, `combat/stamina_starvation_ticks`

### Stage 4: Adversarial Exploiter Audit (Days 9--10)
* [ ] Freeze the $50\times 10^6$-step champion checkpoint $\pi_{\text{champ}}$.
* [ ] Train a dedicated **Exploiter Agent** $\pi_{\text{exploit}}$ exclusively against $\pi_{\text{champ}}$ for $8\times 10^6$ steps with $\alpha = 0.05$.
* [ ] Inspect replay videos of $\pi_{\text{exploit}}$ wins:
  * If $\pi_{\text{exploit}}$ wins via legitimate boxing adjustments (e.g., baiting counters, working the body early to win Round 3), inject $\pi_{\text{exploit}}$ into the PFSP pool and resume league training.
  * If $\pi_{\text{exploit}}$ discovers a physics/stamina exploit (e.g., rapid micro-stepping or parry-cancel spam), tune the stamina/recovery frame cost in the physics engine, re-verify `T11`--`T18`, and resume.

### Recommended Baseline PPO Hyperparameters for Joxing

| Hyperparameter | Recommended Value | Rationale for 2-Player Boxing |
| :--- | :--- | :--- |
| Action Decision Rate | $15\text{ Hz}$ or $20\text{ Hz}$ ($3\text{--}4$ physics ticks/action) | Matches human reaction cadence ($\sim 50\text{--}66\text{ ms}$) and shortens credit assignment horizon. |
| Parallel Environments ($N_{\text{envs}}$) | $64$ or $128$ | High diversity of simultaneous opponent checkpoints per batch. |
| Rollout Horizon ($T_{\text{rollout}}$) | $256$ steps ($\sim 12.8\text{--}17\text{ s}$ per rollout) | Captures multiple full exchanges and knockdown recovery cycles. |
| Recurrent Sequence Chunk ($L_{\text{seq}}$) | $16$ steps | Sufficient to model $1\text{ s}$ punch combos and opponent rhythm without vanishing gradients. |
| Discount Factor ($\gamma$) | $0.995$ | Effective horizon of $200$ steps ($\sim 10\text{--}13\text{ s}$), balancing immediate trades with round-end CompuBox scoring. |
| GAE Lambda ($\lambda_{\text{GAE}}$) | $0.95$ | Standard variance-bias tradeoff for multi-step combat trades. |
| PPO Clip Range ($\epsilon$) | $0.20$ | Prevents policy collapse during sudden opponent pool shifts. |
| Entropy Coefficient ($c_{\text{ent}}$) | $0.015 \rightarrow 0.003$ (annealed) | Prevents premature convergence to a single punch type (`JAB_HEAD` only). |
| Learning Rate | $3 \times 10^{-4} \rightarrow 5 \times 10^{-5}$ | Linear decay stabilizes late-stage Nash equilibrium refinement. |
| Max Gradient Norm | $0.5$ | Essential for recurrent GRU/LSTM stability during high-damage KO spikes. |