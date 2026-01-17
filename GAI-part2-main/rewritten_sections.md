# Rewritten Report Sections (Based on Current Implementation)

## 3. Control Sets & Architectures

We trained two separate AI agents using PPO (Proximal Policy Optimization), a standard reinforcement learning algorithm. Both agents used identical neural network architectures but operated under fundamentally different control schemes.

**Network Architecture:** We employed a standard Multi-Layer Perceptron (MLP) policy with two hidden layers of 64 neurons each (64×64), the default configuration in Stable Baselines3. This architecture proved sufficient for the complexity of our state space without requiring deeper or more complex networks.

**Observation Space Design:** Rather than processing raw visual input, we designed a compact numeric observation vector that provides the agent with essential battlefield information. The observation space includes:

- **Player state** (4 values): Position (x, y) normalized to [-1, 1], velocity (vx, vy) normalized to [-1, 1]
- **Player orientation** (2 values): Sine and cosine of the facing angle
- **Nearest enemy information** (3 values): Direction vector (dx, dy) and normalized distance
- **Nearest spawner information** (3 values): Direction vector (dx, dy) and normalized distance  
- **Status indicators** (2 values): Health [0-1] and current phase [0-1]

The observation dimension varies by control mode:
- **Directional Mode:** 14-dimensional observation vector
- **Rotation Mode:** 15-dimensional observation vector with an additional feature—the relative angle to the nearest enemy. This extra input compensates for the decoupled facing/movement relationship in rotation mode, helping the agent learn to aim while drifting.

### Control Mode Comparison

**1. Rotation Control (Asteroids-Style Physics)**
- **Action Space:** 5 discrete actions (No-op, Thrust Forward, Rotate Left, Rotate Right, Shoot)
- **Dynamics:** Physics-based momentum system where the ship maintains velocity and can face a different direction than its movement. The agent must apply thrust to accelerate and carefully time rotations to aim while drifting.
- **Challenge:** Requires coordinated control of independent movement and orientation axes. The agent must learn to predict drift, plan rotations in advance, and shoot while managing momentum—creating a complex credit assignment problem.

**2. Directional Control (Tank-Style Movement)**
- **Action Space:** 6 discrete actions (No-op, Up, Down, Left, Right, Shoot)
- **Dynamics:** Instant directional movement where the ship immediately moves and faces in the commanded direction. No momentum or drift—stopping and turning are instantaneous.
- **Advantages:** Decouples navigation from shooting mechanics. The agent can focus on strategic decision-making (where to go, when to shoot) without wrestling with low-level motion control. This dramatically reduces the exploration space and accelerates learning.

---

## 4. Hyperparameter Configuration

We used PPO with carefully tuned hyperparameters optimized for each control mode. The configuration balances exploration, learning stability, and convergence speed for the multi-phase spawner destruction task.

### Shared Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Algorithm** | PPO | Stable on-policy learning for episodic RL tasks |
| **Policy Network** | MLP (64×64) | Default SB3 architecture, sufficient for 14-15 dim state |
| **n_steps** | 2048 | Adequate trajectory length for multi-step credit assignment |
| **batch_size** | 64 | Balance between gradient stability and computational efficiency |
| **n_epochs** | 10 | Multiple optimization passes per rollout for sample efficiency |
| **gamma (γ)** | 0.99 | High discount factor for long-horizon spawner destruction rewards |
| **GAE lambda (λ)** | 0.95 | Advantage estimation with variance reduction |
| **clip_range** | 0.2 | Standard PPO clipping for stable policy updates |
| **max_steps** | 2000 | Episode timeout (sufficient for 3-phase completion) |
| **Total Timesteps** | 600,000 | Extended training for multi-phase strategic mastery |

### Mode-Specific Tuning

| Parameter | Directional Mode | Rotation Mode | Rationale |
|-----------|------------------|---------------|-----------|
| **learning_rate** | 3e-4 | 5e-4 | Rotation requires higher learning rate to overcome the added complexity of decoupled orientation control |
| **ent_coef** | 0.04 | 0.05 | Rotation needs stronger exploration incentive to discover effective thrust-rotate-shoot coordination strategies |
| **Observation Dims** | 14 | 15 | Rotation includes relative angle to enemy as explicit aiming guidance |
| **Action Space** | Discrete(6) | Discrete(5) | Different control paradigms (direct movement vs physics-based) |

The higher learning rate and entropy coefficient for rotation mode compensate for its larger effective state-action space and the need to explore coordinated maneuvers.

---

## 5. Training Results & Performance Comparison

Training was conducted on CPU over 600,000 timesteps with deterministic evaluation every 10,000 steps. Performance metrics were logged to TensorBoard and tracked via checkpoint evaluations (5 episodes per evaluation).

### Training Performance (from TensorBoard Logs)

| Metric | Directional Mode | Rotation Mode |
|--------|------------------|---------------|
| **Final Mean Reward (600k steps)** | 4,762.77 - 4,769.72 | 3,772.75 - 3,789.94 |
| **Training Time** | 7.523 minutes | 7.595 minutes |

### Best Model Evaluation (5-episode deterministic test)

| Metric | Directional Mode | Rotation Mode |
|--------|------------------|---------------|
| **Mean Reward** | 4,705.83 ± 14.30 | 4,304.31 ± 252.50 |
| **Best Episode** | 4,726.13 | 4,645.22 |
| **Worst Episode** | 4,691.27 | 4,086.42 |
| **Mean Episode Length** | 1,794.0 ± 170.7 steps | 1,517.4 ± 342.9 steps |
| **Phase Progression** | 3.0 ± 0.0 (100% Phase 3) | 3.0 ± 0.0 (100% Phase 3) |
| **Performance Variance** | Very Low (±14.30) | High (±252.50) |

### Key Findings

**1. Performance Gap:** The Directional agent achieved approximately **26% higher mean reward** during training (4,766 vs 3,781 ep_rew_mean at 600k timesteps) and **9.3% higher in final evaluation** (4,706 vs 4,304) compared to the Rotation agent. Despite both agents consistently reaching Phase 3, the Directional agent completes objectives more efficiently within each phase.

**2. Consistency:** The Directional agent exhibited remarkably stable performance with a standard deviation of only ±14.30, compared to Rotation's ±252.50. This 18× difference in variance suggests:
   - Directional control produces more reliable, consistent strategies
   - Rotation control leads to high-variance behavior where success depends heavily on early-game positioning and enemy spawn patterns

**3. Episode Duration:** Directional agents survived longer on average (1,794 vs 1,517 steps), suggesting better damage avoidance or more efficient enemy clearing. The lower variance in Directional episode length (±170.7 vs ±342.9) further supports superior strategic consistency.

### Behavioral Analysis

**Directional Mode Strategy:**
- **Direct Targeting:** Agents rapidly learned to prioritize spawners, developing a "beeline" approach—moving straight toward the nearest spawner while continuously firing.
- **Efficient Clearing:** The instant stop-and-turn capability allowed agents to quickly eliminate spawners without needing complex positioning maneuvers.
- **Strategic Focus:** Because movement mechanics were trivial, the agent devoted learning capacity to higher-level decision-making: target prioritization, enemy avoidance timing, and phase progression optimization.

**Rotation Mode Strategy:**
- **Movement Struggles:** Early training (0-200k steps) showed erratic, inefficient movement as agents learned to coordinate thrust and rotation. Many episodes ended with agents spinning aimlessly or drifting into walls.
- **"Turret" Behavior:** Successful rotation agents often adopted a stop-rotate-shoot pattern, essentially treating the ship as a stationary turret rather than leveraging momentum for evasive maneuvers.
- **Aim Difficulties:** The decoupled orientation made sustained firing on moving targets difficult. Agents frequently overshot rotations or drifted off-angle while shooting.
- **Inconsistent Performance:** High variance reflects the fragility of rotation strategies—small early-game mistakes (poor initial positioning, missed shots) cascaded into failed episodes.

### Interpretation

**Why Directional Mode Outperformed:**
1. **Reduced Action Space Complexity:** Directional control collapses the two-axis control problem (movement + orientation) into a single axis (direction), drastically simplifying exploration.
2. **Immediate Feedback:** Instant response to actions provides clearer credit assignment. In Rotation mode, the delayed effect of thrust and rotation obscures which actions led to success.
3. **Focus on Strategy Over Mechanics:** Directional agents could immediately focus on *what to do* rather than spending thousands of timesteps learning *how to move*.

**Rotation Mode's Theoretical Advantage Unrealized:**
Despite the potential for advanced maneuvers (strafing, evasive drifting), rotation agents never discovered these strategies. The added control complexity consumed learning capacity, preventing the emergence of sophisticated tactics. In a task where the primary objective is straightforward (destroy spawners), the simpler control scheme proved decisively superior.

**Conclusion:** For reinforcement learning in action-based games, **control simplicity is often more valuable than control flexibility**. The Directional mode's 9% reward advantage and 18× lower variance demonstrate that reducing low-level control burden allows agents to focus on strategic objectives, resulting in faster learning, more reliable performance, and higher overall achievement.
