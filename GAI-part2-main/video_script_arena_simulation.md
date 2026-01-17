# Arena Simulation Video Script (3 minutes)
## Presenter: [Your Name] - Part II: Deep RL Arena Control

---

## **[0:00-0:20] Introduction & Overview (20 seconds)**

**[Show: Title slide or code structure]**

"Hello! I'll be presenting Part 2 of our project—the Arena Simulation. We built a real-time combat environment where a deep reinforcement learning agent must survive waves of enemies across three escalating phases.

The key innovation: we trained TWO separate agents using PPO—one with rotation-based physics control, and one with direct tank-style movement—to compare which control scheme learns better strategies."

---

## **[0:20-1:10] Section 3: Control Sets Demo (50 seconds)**

**[Show: Gameplay - Rotation Mode running]**

"First, let me show you Rotation Mode in action. This uses Asteroids-style physics—the agent has 5 actions: thrust forward, rotate left, rotate right, shoot, or do nothing. Notice how the ship drifts with momentum while the agent tries to aim and fire.

**[Point out on screen]** The observation space includes 15 features: player position, velocity, orientation, nearest enemy direction, spawner location, health, and current phase. The extra feature here is the relative angle to enemies, helping the agent learn to aim while drifting."

**[Show: Gameplay - Directional Mode running]**

"Now here's Directional Mode—much simpler. The agent has 6 actions: move up, down, left, right, shoot, or nothing. Movement is instant, and the ship automatically faces where it moves. 

**[Point out behavior]** Watch how the agent moves decisively toward spawners and fires continuously. The 14-dimensional observation is similar, but without the extra angle feature since orientation follows movement automatically."

---

## **[1:10-1:50] Section 4: Training Configuration (40 seconds)**

**[Show: Table of hyperparameters or TensorBoard graphs]**

"Both agents used PPO with 64×64 neural networks—nothing fancy, just the standard Stable Baselines3 architecture. We trained each for 600,000 timesteps, which took about 7.5 minutes on CPU.

**[Highlight key parameters]** Here's what's important: we used identical settings for most parameters—2048 steps per rollout, batch size 64, gamma 0.99 for long-term planning.

But notice the difference: Rotation Mode needed a HIGHER learning rate—5e-4 instead of 3e-4—and more exploration entropy, 0.05 versus 0.04. This compensates for the added complexity of coordinating thrust and rotation."

---

## **[1:50-2:35] Section 5: Results & Performance (45 seconds)**

**[Show: Evaluation results table and gameplay comparison]**

"The results were striking. The Directional agent achieved a mean reward of 4,706, compared to Rotation's 4,304—that's 9% higher performance. But more importantly, look at the variance: Directional had a standard deviation of only ±14 points, while Rotation varied by ±252—that's 18 times MORE inconsistent.

**[Show: Phase progression in gameplay]** Both agents successfully reach Phase 3 every time—you can see enemies multiplying and speeding up. But the Directional agent survives longer—1,794 steps on average versus 1,517 for Rotation."

**[Show: Side-by-side if possible]** 

"Why? Directional control learned a 'beeline strategy'—move straight to spawners, fire continuously, done. Rotation agents struggled with coordination, often spinning inefficiently or adopting a stationary 'turret' behavior instead of leveraging momentum for evasive maneuvers."

---

## **[2:35-3:00] Conclusion & Key Takeaway (25 seconds)**

**[Show: Final gameplay or summary slide]**

"The key takeaway: simpler control beats flexible control for reinforcement learning. By reducing the low-level movement complexity, the Directional agent could focus on STRATEGY—target prioritization, enemy avoidance, phase management.

The Rotation agent spent most of its learning budget just figuring out how to move, never discovering the advanced tactics its control scheme theoretically allowed. This demonstrates that in RL, **control simplicity accelerates learning and improves reliability**.

Thank you!"

---

## **Visual Checklist for Video Recording:**

✅ **Rotation Mode clip showing:**
- Arena running with enemies spawning and moving
- Agent drifting with momentum while rotating to aim
- Projectiles firing and hitting enemies
- Phase progression (enemies multiplying/speeding up)
- Health bar visible

✅ **Directional Mode clip showing:**
- Arena running with smooth, decisive movement
- Agent moving directly toward spawners while firing
- Collisions functioning (enemies taking damage)
- Phase progression occurring
- More efficient spawner destruction

✅ **Evidence Requirements:**
- Models from `models/directional/` and `models/rotation/` folders
- Logs from `logs/directional/` and `logs/rotation/` showing training metrics
- Episode lengths matching reported averages (1,794 vs 1,517 steps)
- Both agents reaching Phase 3 consistently

---

## **Timing Breakdown:**
- Introduction: 20s
- Control Sets Demo: 50s
- Training Config: 40s
- Results & Performance: 45s
- Conclusion: 25s
- **Total: 3 minutes**

---

## **Presenter Tips:**

1. **Run both game modes before recording** to ensure models load correctly
2. **Capture 20-30 second clips** of each mode in action—you can edit them into the narration
3. **Point at specific screen elements** when mentioning observation features or behaviors
4. **Speak clearly but conversationally**—don't rush through the numbers
5. **Emphasize the comparison**: "Watch THIS mode vs THAT mode"
6. **End with confidence**: The conclusion about control simplicity is your key contribution

Good luck with your presentation! 🎥
