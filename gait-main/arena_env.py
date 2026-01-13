import math
import random
import time
from typing import Tuple, List

try:
	import gymnasium as gym
	from gymnasium import spaces
except ImportError:
	try:
		import gym
		from gym import spaces
	except Exception:
		# allow running the manual-play loop even if gym isn't installed in this interpreter
		gym = None

		class _DummySpaces:
			class Discrete:
				def __init__(self, n):
					self.n = n

				def sample(self):
					import random as _random
					return _random.randrange(self.n)

			class Box:
				def __init__(self, low, high, shape, dtype=None):
					self.shape = shape

					def sample(self):
						# return zeros for a deterministic placeholder
						if isinstance(self.shape, int):
							return [0.0] * self.shape
						return [0.0] * int(self.shape[0])

		spaces = _DummySpaces()

		class _DummyEnv:
			pass

		gym = type("gym", (), {"Env": _DummyEnv})
import numpy as np
import pygame

from entities import Player, Enemy, Spawner, Bullet


class ArenaEnv(gym.Env):
	"""A simple Pygame arena Gym environment.

	control_mode: 'rotation' or 'directional'
	"""

	metadata = {"render.modes": ["human"]}

	# Reward constants for easy tuning - HEAVILY FOCUSED ON SPAWNER OBJECTIVE
	REWARD_ENEMY_KILLED = 1.0
	REWARD_SPAWNER_DESTROYED = 500.0  # MASSIVE reward for primary objective
	REWARD_SPAWNER_HIT = 50.0  # Strong reward for progress on spawner
	REWARD_PHASE_COMPLETE = 1000.0  # Huge bonus for completing all spawners
	REWARD_TAKE_DAMAGE = -1.0
	REWARD_DEATH = -50.0
	REWARD_LIVING_PENALTY = 0.0  # REMOVED - no time pressure
	REWARD_SHAPING_COEF = 8.0  # MUCH stronger pull toward spawners (was 3.0)
	REWARD_CORNER_PENALTY = -0.5  # NEW: Punish staying in corners

	def __init__(self, width=800, height=600, control_mode="directional", max_steps=2000):
		super().__init__()
		assert control_mode in ("directional", "rotation")
		self.width = width
		self.height = height
		self.control_mode = control_mode
		self.max_steps = max_steps

		# normalization constants for observations
		self.max_player_speed = 400.0  # used to normalize vx,vy
		# maximum meaningful distance (diagonal of arena)
		self._max_dist = math.hypot(self.width, self.height)
		self.max_phase = 10.0

		# action space
		if self.control_mode == "rotation":
			# no-op, thrust, rotate left, rotate right, shoot
			self.action_space = spaces.Discrete(5)
		else:
			# no-op, up, down, left, right, shoot
			self.action_space = spaces.Discrete(6)

		# observation vector size: normalized positions, velocities, angle (sin,cos), nearest enemy rel (dx,dy,dist), nearest spawner rel, health, phase
		# [0-1]: Player position (px, py) normalized to [-1, 1]
		# [2-3]: Player velocity (vx, vy) normalized to [-1, 1]
		# [4-5]: Player orientation (sin(angle), cos(angle))
		# [6-8]: Nearest enemy (dx, dy, distance) normalized
		# [9-11]: Nearest spawner (dx, dy, distance) normalized
		# [12]: Player health normalized to [0, 1]
		# [13]: Current phase normalized to [0, 1]
		self.obs_size = 14
		self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_size,), dtype=np.float32)

		# game objects
		self.player = None
		self.enemies: List[Enemy] = []
		self.spawners: List[Spawner] = []
		self.bullets: List[Bullet] = []

		self.phase = 1
		self.steps = 0

		# rendering
		pygame.init()
		self.screen = None
		self.clock = pygame.time.Clock()

		# reward bookkeeping
		self._last_enemy_count = 0
		# bookkeeping for distance shaping: track previous nearest-target distance (normalized)
		self._last_target_dist = 1.0

		# spawner configuration: keep count consistent across phases
		self.initial_spawner_count = 3
		# base spawner positions (will be used to recreate spawners each phase)
		self.spawner_positions = [(50, 50), (self.width - 50, 50), (50, self.height - 50)]

	def reset(self, seed=None, options=None):
		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)
		
		self.player = Player(self.width / 2.0, self.height / 2.0)
		self.enemies = []
		self.bullets = []
		# reset phase first so spawner stats use phase=1
		self.phase = 1
		# create spawners using stored base positions to keep count consistent
		self.spawners = []
		for i, pos in enumerate(self.spawner_positions[: self.initial_spawner_count]):
			sp = Spawner(pos[0], pos[1])
			# spawn interval starts VERY SLOW to give agent time to learn
			sp.spawn_interval = 5.0  # Very slow spawning - one enemy every 5 seconds
			sp.cooldown = sp.spawn_interval * 0.5
			# spawner health: ONLY 1 HIT to destroy! Makes task very achievable
			sp.health = 1.0  # Only 1 HP - one shot kills
			sp.max_health = sp.health
			self.spawners.append(sp)

		self.steps = 0
		self._last_enemy_count = 0
		# initialize distance shaping baseline
		self._last_target_dist = 1.0
		obs = self._get_obs()
		return obs, {}

	def _get_obs(self):
		# Player position normalized to [-1, 1], accounting for radius so range is stable
		px = ((self.player.x - getattr(self.player, 'radius', 0)) / max(1.0, (self.width - 2 * getattr(self.player, 'radius', 0))))
		py = ((self.player.y - getattr(self.player, 'radius', 0)) / max(1.0, (self.height - 2 * getattr(self.player, 'radius', 0))))
		px = float(np.clip(px * 2.0 - 1.0, -1.0, 1.0))
		py = float(np.clip(py * 2.0 - 1.0, -1.0, 1.0))

		# velocities normalized to [-1, 1] by a chosen max player speed
		vx = float(np.clip(self.player.vx / self.max_player_speed, -1.0, 1.0))
		vy = float(np.clip(self.player.vy / self.max_player_speed, -1.0, 1.0))

		# orientation as sin/cos (already in [-1,1])
		a_s = math.sin(self.player.angle)
		a_c = math.cos(self.player.angle)

		# nearest enemy: encode direction (dx,dy) normalized by max distance and distance normalized to [0,1]
		if self.enemies:
			ne = min(self.enemies, key=lambda e: (e.x - self.player.x) ** 2 + (e.y - self.player.y) ** 2)
			dx_raw = ne.x - self.player.x
			dy_raw = ne.y - self.player.y
			d = math.hypot(dx_raw, dy_raw)
			dx_e = float(np.clip(dx_raw / max(1.0, self._max_dist), -1.0, 1.0))
			dy_e = float(np.clip(dy_raw / max(1.0, self._max_dist), -1.0, 1.0))
			dist_e = float(np.clip(d / max(1.0, self._max_dist), 0.0, 1.0))
		else:
			# encode 'no enemy' as zero direction and max distance (i.e., far away)
			dx_e = 0.0
			dy_e = 0.0
			dist_e = 1.0

		# nearest spawner (only active ones)
		active_spawners = [s for s in self.spawners if s.is_active()]
		if active_spawners:
			ns = min(active_spawners, key=lambda s: (s.x - self.player.x) ** 2 + (s.y - self.player.y) ** 2)
			dx_raw = ns.x - self.player.x
			dy_raw = ns.y - self.player.y
			d = math.hypot(dx_raw, dy_raw)
			dx_s = float(np.clip(dx_raw / max(1.0, self._max_dist), -1.0, 1.0))
			dy_s = float(np.clip(dy_raw / max(1.0, self._max_dist), -1.0, 1.0))
			dist_s = float(np.clip(d / max(1.0, self._max_dist), 0.0, 1.0))
		else:
			dx_s = 0.0
			dy_s = 0.0
			dist_s = 1.0

		# health normalized to [0,1]
		max_hp = max(1.0, getattr(self.player, 'max_health', 10.0))
		health = float(np.clip(self.player.health / max_hp, 0.0, 1.0))

		# phase normalized to [0,1]
		phase = float(np.clip(self.phase, 0.0, self.max_phase) / self.max_phase)

		obs = np.array([px, py, vx, vy, a_s, a_c, dx_e, dy_e, dist_e, dx_s, dy_s, dist_s, health, phase], dtype=np.float32)
		return obs

	def step(self, action):
		done = False
		reward = 0.0
		info = {}
		dt = 1.0 / 30.0
		self.steps += 1

		# --- UPDATED DISTANCE SHAPING ---
		def _nearest_norm_dist():
			# PRIORITY: If spawners exist, ONLY calculate distance to nearest ACTIVE spawner.
			# This forces the agent's shaping reward to purely focus on "closing the gap" to the objective.
			active_spawners = [s for s in self.spawners if s.is_active()]
			if active_spawners:
				ns = min(active_spawners, key=lambda s: (s.x - self.player.x) ** 2 + (s.y - self.player.y) ** 2)
				d = math.hypot(ns.x - self.player.x, ns.y - self.player.y) / max(1.0, self._max_dist)
				return float(np.clip(d, 0.0, 1.0))
			
			# FALLBACK: Only if no spawners (e.g. momentarily between phases), look at enemies
			elif self.enemies:
				ne = min(self.enemies, key=lambda e: (e.x - self.player.x) ** 2 + (e.y - self.player.y) ** 2)
				d = math.hypot(ne.x - self.player.x, ne.y - self.player.y) / max(1.0, self._max_dist)
				return float(np.clip(d, 0.0, 1.0))
			
			return 1.0  # No targets

		prev_target_dist = _nearest_norm_dist()

		# apply action via Player.apply_action (adds inertia instead of snapping velocities)
		shoot_request = None
		if hasattr(self.player, 'apply_action'):
			shoot_request = self.player.apply_action(action, self.control_mode, dt)
		else:
			# fallback to previous behavior (shouldn't happen)
			if self.control_mode == "rotation":
				if action == 1:
					accel = 200.0
					self.player.vx += math.cos(self.player.angle) * accel * dt
					self.player.vy += math.sin(self.player.angle) * accel * dt
				elif action == 2:
					self.player.angle -= 3.0 * dt
				elif action == 3:
					self.player.angle += 3.0 * dt
				elif action == 4:
					shoot_request = True
			else:
				speed = 160.0
				if action == 1:
					self.player.vy = -speed
					self.player.vx = 0
				elif action == 2:
					self.player.vy = speed
					self.player.vx = 0
				elif action == 3:
					self.player.vx = -speed
					self.player.vy = 0
				elif action == 4:
					self.player.vx = speed
					self.player.vy = 0
				elif action == 5:
					shoot_request = True

		# handle shooting request with player's cooldown
		if shoot_request == "shoot" or shoot_request is True:
			if hasattr(self.player, 'can_shoot') and self.player.can_shoot():
				# consume shoot cooldown immediately and spawn bullet
				if hasattr(self.player, 'consume_shoot'):
					self.player.consume_shoot()
				self._player_shoot()

		# update player
		# update player (applies drag and clamps)
		self.player.update(dt, self.width, self.height)

		# update spawners and collect spawned enemies
		for sp in list(self.spawners):
			if not sp.is_active():
				continue
			new_enemies = sp.update(dt, self.phase)
			if new_enemies:
				self.enemies.extend(new_enemies)

		# update enemies and check collisions with player
		for e in list(self.enemies):
			e.update(dt, self.player.x, self.player.y)
			# remove enemies that travel far off-screen to keep observations clean
			margin = 64
			if e.x < -margin or e.x > self.width + margin or e.y < -margin or e.y > self.height + margin:
				try:
					self.enemies.remove(e)
				except ValueError:
					pass
				continue
			if e.collides_with(self.player):
				# enemy damages player and is removed
				self.player.take_damage(1.0)
				try:
					self.enemies.remove(e)
				except ValueError:
					pass
				reward += self.REWARD_TAKE_DAMAGE

		# update bullets and handle collisions
		for b in list(self.bullets):
			alive = b.update(dt)
			if not alive:
				try:
					self.bullets.remove(b)
				except ValueError:
					pass
				continue
			# remove bullets that leave the arena by a margin
			b_margin = 64
			if b.x < -b_margin or b.x > self.width + b_margin or b.y < -b_margin or b.y > self.height + b_margin:
				try:
					self.bullets.remove(b)
				except ValueError:
					pass
				continue
			# handle player bullets hitting enemies or spawners
			if b.owner == "player":
				hit_something = False
				for e in list(self.enemies):
					if b.collides_with(e):
						e.take_damage(1.0)
						try:
							self.bullets.remove(b)
						except ValueError:
							pass
						if e.is_dead():
							try:
								self.enemies.remove(e)
							except ValueError:
								pass
							reward += self.REWARD_ENEMY_KILLED
						hit_something = True
						break
				if hit_something:
					continue
				# check spawners (only active ones)
				for sp in list(self.spawners):
					if sp.is_active() and b.collides_with(sp):
						prev_active = sp.is_active()
						sp.take_damage(1.0)
						try:
							self.bullets.remove(b)
						except ValueError:
							pass
						if not sp.is_active():
							reward += self.REWARD_SPAWNER_DESTROYED
							# Clear all enemies when spawner destroyed - gives agent breathing room
							self.enemies.clear()
						else:
							reward += self.REWARD_SPAWNER_HIT  # Reward damaging spawner
						break


		# progress phase when all spawners inactive
		if all(not s.active for s in self.spawners):
			self.phase += 1
			reward += self.REWARD_PHASE_COMPLETE
			# revive spawners with slightly increased difficulty
			self.spawners = []
			for i, pos in enumerate(self.spawner_positions[: self.initial_spawner_count]):
				sp = Spawner(pos[0], pos[1])
				# gradually decrease spawn intervals
				sp.spawn_interval = max(2.0, 4.0 - 0.5 * (self.phase - 1))
				sp.cooldown = sp.spawn_interval * 0.5
				# gradually increase spawner health
				sp.health = 2.0 + 0.5 * (self.phase - 1)
				sp.max_health = sp.health
				self.spawners.append(sp)

		# REMOVE living penalty to eliminate time pressure
		# reward += self.REWARD_LIVING_PENALTY

		# --- Penalize corner camping ---
		corner_margin = 80  # Distance from edge that counts as "corner"
		in_corner_x = (self.player.x < corner_margin) or (self.player.x > self.width - corner_margin)
		in_corner_y = (self.player.y < corner_margin) or (self.player.y > self.height - corner_margin)
		if in_corner_x and in_corner_y:
			reward += self.REWARD_CORNER_PENALTY  # Punish corner camping

		# --- distance shaping reward (small). Encourage moving closer to the nearest target (spawner priority)
		# Compute new nearest-target distance and reward proportionally to the reduction
		new_target_dist = _nearest_norm_dist()
		shaping_reward = self.REWARD_SHAPING_COEF * (prev_target_dist - new_target_dist)
		# Allow negative shaping (punish moving away from spawner) to enforce focus
		reward += float(shaping_reward)
		# store for next step
		self._last_target_dist = new_target_dist

		# death
		if self.player.health <= 0:
			done = True
			reward += self.REWARD_DEATH

		if self.steps >= self.max_steps:
			done = True

		obs = self._get_obs()
		truncated = False  # For gymnasium compatibility
		return obs, reward, done, truncated, info

	def _player_shoot(self):
		speed = 250.0  # bullet speed
		ang = self.player.angle
		vx = math.cos(ang) * speed
		vy = math.sin(ang) * speed
		b = Bullet(self.player.x + math.cos(ang) * (self.player.radius + 4), self.player.y + math.sin(ang) * (self.player.radius + 4), vx, vy)
		self.bullets.append(b)

	def render(self, mode="human"):
		if self.screen is None:
			self.screen = pygame.display.set_mode((self.width, self.height))
		self.screen.fill((30, 30, 40))

		# draw spawners
		for sp in self.spawners:
			sp.draw(self.screen)

		# draw enemies
		for e in self.enemies:
			e.draw(self.screen)

		# draw bullets
		for b in self.bullets:
			b.draw(self.screen)

		# draw player (triangle)
		self.player.draw(self.screen)

		# HUD
		font = pygame.font.SysFont(None, 20)
		hud = font.render(f"Health: {self.player.health:.1f}  Phase: {self.phase}  Enemies: {len(self.enemies)}", True, (230, 230, 230))
		self.screen.blit(hud, (6, 6))

		pygame.display.flip()
		self.clock.tick(60)

	def close(self):
		try:
			pygame.quit()
		except Exception:
			pass


if __name__ == "__main__":
	# Demo loop without keyboard input: actions are numeric, environment still renders.
	env = ArenaEnv(control_mode="rotation")
	obs, _ = env.reset()
	running = True
	print("Demo: running automated actions (no keyboard). Close window to exit.")
	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

		# choose an action programmatically (here: random actions for demo)
		if env.control_mode == "rotation":
			# actions: 0 noop,1 thrust,2 left,3 right,4 shoot
			action = random.choice([0, 1, 2, 3, 4])
		else:
			# actions: 0 noop,1 up,2 down,3 left,4 right,5 shoot
			action = random.choice([0, 1, 2, 3, 4, 5])

		result = env.step(action)
		obs, r, done = result[0], result[1], result[2]
		env.render()
		if done:
			print("Episode finished — resetting environment")
			obs, _ = env.reset()

	env.close()

