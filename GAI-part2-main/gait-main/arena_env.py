import math
import random
import time
from typing import Tuple, List, Optional

try:
	import gymnasium as gym
	from gymnasium import spaces
except ImportError as e:
	# Fail fast: gymnasium is required for correct Gym API behavior.
	raise ImportError("gymnasium is required to use ArenaEnv") from e
import numpy as np

import pygame

from entities import Player, Enemy, Spawner, Bullet, PALETTE
import pygame.gfxdraw
from dataclasses import dataclass


@dataclass
class Particle:
	x: float
	y: float
	vx: float
	vy: float
	life: float
	size: int
	color: Tuple[int, int, int, int]

	def update(self, dt: float) -> bool:
		self.x += self.vx * dt
		self.y += self.vy * dt
		self.life -= dt
		return self.life > 0

	def draw(self, surf: pygame.Surface):
		try:
			# alpha scales with remaining life
			a = max(0, min(255, int(255 * (self.life / max(1e-6, 0.6)))))
			col = (self.color[0], self.color[1], self.color[2], a)
			s = pygame.Surface((self.size * 2 + 4, self.size * 2 + 4), pygame.SRCALPHA)
			pygame.gfxdraw.filled_circle(s, s.get_width() // 2, s.get_height() // 2, self.size, col)
			pygame.gfxdraw.aacircle(s, s.get_width() // 2, s.get_height() // 2, self.size, col)
			surf.blit(s, (int(round(self.x)) - s.get_width() // 2, int(round(self.y)) - s.get_height() // 2))
		except Exception:
			pass

# Difficulty tuning constants
# How many spawners to allow at most (phase determines actual count: min(phase, MAX_SPAWNERS))
MAX_SPAWNERS = 3
# Initial spawn interval applied on reset (seconds between spawns)
INITIAL_SPAWN_INTERVAL = 5.0
# Fraction of spawn_interval used to initialize cooldown
SPAWNER_COOLDOWN_FACTOR = 0.5
# Initial spawner health on reset
INITIAL_SPAWNER_HEALTH = 1.0
# On phase progression: spawn interval = max(SPAWNER_MIN_SPAWN_INTERVAL, SPAWNER_BASE_SPAWN_INTERVAL - SPAWNER_SPAWN_DECAY*(phase-1))
SPAWNER_MIN_SPAWN_INTERVAL = 2.0
SPAWNER_BASE_SPAWN_INTERVAL = 4.0
SPAWNER_SPAWN_DECAY = 0.5
# Spawner health growth per phase
SPAWNER_BASE_HEALTH = 2.0
SPAWNER_HEALTH_PER_PHASE = 0.5

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
	REWARD_WALL_PENALTY = -0.3  # Penalty for ramming into walls

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
		# [14]: Relative angle to nearest enemy normalized to [-1, 1] (rotation mode only)
		# Directional mode: 14 obs (no relative angle - facing is tied to movement)
		# Rotation mode: 15 obs (includes relative angle - agent controls facing independently)
		self.obs_size = 15 if self.control_mode == "rotation" else 14
		# Define per-element bounds for the observation vector:
		# [px, py]        -> positions normalized to [-1, 1]
		# [vx, vy]        -> velocities normalized to [-1, 1]
		# [a_s, a_c]      -> sin/cos orientation in [-1, 1]
		# [dx_e, dy_e]    -> enemy direction components in [-1, 1]
		# [dist_e]        -> enemy distance in [0, 1]
		# [dx_s, dy_s]    -> spawner direction components in [-1, 1]
		# [dist_s]        -> spawner distance in [0, 1]
		# [health]        -> [0, 1]
		# [phase]         -> [0, 1]
		if self.control_mode == "rotation":
			# Rotation mode: 15 dimensions (includes relative angle to enemy)
			low = np.array([
				-1.0, -1.0,  # px, py
				-1.0, -1.0,  # vx, vy
				-1.0, -1.0,  # a_s, a_c
				-1.0, -1.0, 0.0,  # dx_e, dy_e, dist_e
				-1.0, -1.0, 0.0,  # dx_s, dy_s, dist_s
				0.0, 0.0,  # health, phase
				-1.0,  # rel_angle_to_enemy
			], dtype=np.float32)
			high = np.array([
				1.0, 1.0,  # px, py
				1.0, 1.0,  # vx, vy
				1.0, 1.0,  # a_s, a_c
				1.0, 1.0, 1.0,  # dx_e, dy_e, dist_e
				1.0, 1.0, 1.0,  # dx_s, dy_s, dist_s
				1.0, 1.0,  # health, phase
				1.0,  # rel_angle_to_enemy
			], dtype=np.float32)
		else:
			# Directional mode: 14 dimensions (no relative angle)
			low = np.array([
				-1.0, -1.0,  # px, py
				-1.0, -1.0,  # vx, vy
				-1.0, -1.0,  # a_s, a_c
				-1.0, -1.0, 0.0,  # dx_e, dy_e, dist_e
				-1.0, -1.0, 0.0,  # dx_s, dy_s, dist_s
				0.0, 0.0,  # health, phase
			], dtype=np.float32)
			high = np.array([
				1.0, 1.0,  # px, py
				1.0, 1.0,  # vx, vy
				1.0, 1.0,  # a_s, a_c
				1.0, 1.0, 1.0,  # dx_e, dy_e, dist_e
				1.0, 1.0, 1.0,  # dx_s, dy_s, dist_s
				1.0, 1.0,  # health, phase
			], dtype=np.float32)
		self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

		# game objects
		self.player: Optional[Player] = None
		self.enemies: List[Enemy] = []
		self.spawners: List[Spawner] = []
		self.bullets: List[Bullet] = []

		self.phase = 1
		self.steps = 0

		# rendering
		pygame.init()
		self.screen = None
		self.clock = pygame.time.Clock()

		# background (procedural galaxy) layers will be generated once
		self._bg_time = 0.0
		self.bg_layers = []  # list of dicts: {surf, speed, offset}
		self._create_bg_layers()

		# visual effects
		self.particles: List[Particle] = []
		self.camera_shake = 0.0
		self._shake_decay = 0.85

	def _create_bg_layers(self):
		"""Create procedural parallax star layers (galaxy background).
		Each layer is a surface with sparse stars; we store speed and an offset.
		"""
		# simple parameters for layers: (star_count, star_size, color, speed)
		params = [
			(120, 1, (200, 200, 255)),
			(60, 2, (170, 190, 255)),
			(30, 3, (255, 220, 200)),
		]
		self.bg_layers = []
		for i, (count, size, color) in enumerate(params):
			surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
			# random stars
			for _ in range(count):
				x = random.randrange(0, self.width)
				y = random.randrange(0, self.height)
				# small filled circle for star (anti-aliased)
				try:
					pygame.gfxdraw.filled_circle(surf, x, y, size, color)
					pygame.gfxdraw.aacircle(surf, x, y, size, color)
				except Exception:
					# fallback to basic draw
					pygame.draw.circle(surf, color, (x, y), size)
			# subtle nebula: draw a few soft larger circles
			for _ in range(6):
				nx = random.randrange(0, self.width)
				ny = random.randrange(0, self.height)
				r = random.randrange(40, 120)
				col = (color[0], color[1], color[2], 12)
				try:
					s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
					pygame.gfxdraw.filled_circle(s, r, r, r, col)
					surf.blit(s, (nx - r, ny - r), special_flags=0)
				except Exception:
					pass
			# layer speed (parallax) -- farther layers move slower
			speed = 6.0 / (1.0 + i * 1.4)
			self.bg_layers.append({
				"surf": surf,
				"speed": speed,
				"offset_x": 0.0,
				"offset_y": 0.0,
			})

		# reward bookkeeping
		self._last_enemy_count = 0
		# bookkeeping for distance shaping: track previous nearest-target distance (normalized)
		self._last_target_dist = 1.0

		# spawner configuration: number of spawners scales with phase (1,2,3)
		# keep max_spawners equal to available positions
		self.max_spawners = 3
		# base spawner positions (will be used to recreate spawners each phase)
		# move spawners slightly inward to avoid corner-wall hugging issues
		margin = 80
		self.spawner_positions = [
			(margin, margin),
			(self.width - margin, margin),
			(self.width // 2, self.height - margin),
		]

	def reset(self, seed=None, options=None):
		# propagate seed to base Env (required by Gymnasium) and to numpy/random
		super().reset(seed=seed)
		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)
		
		self.player = Player(self.width / 2.0, self.height / 2.0)
		self.enemies = []
		self.bullets = []
		# reset phase first so spawner stats use phase=1
		self.phase = 1
		# create spawners: phase 1 -> 1, phase 2 -> 2, phase 3+ -> 3 (clamped to available positions)
		self.spawners = []
		count = min(self.phase, len(self.spawner_positions), self.max_spawners)
		for i, pos in enumerate(self.spawner_positions[:count]):
			sp = Spawner(pos[0], pos[1])
			# spawn interval starts VERY SLOW to give agent time to learn
			sp.spawn_interval = INITIAL_SPAWN_INTERVAL
			sp.cooldown = sp.spawn_interval * SPAWNER_COOLDOWN_FACTOR
			# spawner health: ONLY 1 HIT to destroy! Makes task very achievable
			sp.health = INITIAL_SPAWNER_HEALTH
			sp.max_health = sp.health
			self.spawners.append(sp)

			self.steps = 0
		self._last_enemy_count = 0
		# initialize distance shaping baseline
		self._last_target_dist = 1.0
		obs = self._get_obs()
		return obs, {}

	def _get_obs(self):
		assert self.player is not None
		p = self.player
		# Player position normalized to [-1, 1], accounting for radius so range is stable
		px = ((p.x - getattr(p, 'radius', 0)) / max(1.0, (self.width - 2 * getattr(p, 'radius', 0))))
		py = ((p.y - getattr(p, 'radius', 0)) / max(1.0, (self.height - 2 * getattr(p, 'radius', 0))))
		px = float(np.clip(px * 2.0 - 1.0, -1.0, 1.0))
		py = float(np.clip(py * 2.0 - 1.0, -1.0, 1.0))

		# velocities normalized to [-1, 1] by a chosen max player speed
		vx = float(np.clip(p.vx / self.max_player_speed, -1.0, 1.0))
		vy = float(np.clip(p.vy / self.max_player_speed, -1.0, 1.0))

		# orientation as sin/cos (already in [-1,1])
		a_s = math.sin(p.angle)
		a_c = math.cos(p.angle)

		# nearest enemy: encode direction (dx,dy) normalized by max distance and distance normalized to [0,1]
		if self.enemies:
			ne = min(self.enemies, key=lambda e: (e.x - p.x) ** 2 + (e.y - p.y) ** 2)
			dx_raw = ne.x - p.x
			dy_raw = ne.y - p.y
			d = math.hypot(dx_raw, dy_raw)
			dx_e = float(np.clip(dx_raw / max(1.0, self._max_dist), -1.0, 1.0))
			dy_e = float(np.clip(dy_raw / max(1.0, self._max_dist), -1.0, 1.0))
			dist_e = float(np.clip(d / max(1.0, self._max_dist), 0.0, 1.0))
			# relative angle to enemy (angle from player facing to enemy direction)
			rel_angle = math.atan2(dy_raw, dx_raw) - p.angle
			rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))  # Normalize to [-π, π]
			rel_angle_norm = float(np.clip(rel_angle / math.pi, -1.0, 1.0))  # [-1, 1]
		else:
			# encode 'no enemy' as zero direction and max distance (i.e., far away)
			dx_e = 0.0
			dy_e = 0.0
			dist_e = 1.0
			rel_angle_norm = 0.0

		# nearest spawner (only active ones)
		active_spawners = [s for s in self.spawners if s.is_active()]
		if active_spawners:
			ns = min(active_spawners, key=lambda s: (s.x - p.x) ** 2 + (s.y - p.y) ** 2)
			dx_raw = ns.x - p.x
			dy_raw = ns.y - p.y
			d = math.hypot(dx_raw, dy_raw)
			dx_s = float(np.clip(dx_raw / max(1.0, self._max_dist), -1.0, 1.0))
			dy_s = float(np.clip(dy_raw / max(1.0, self._max_dist), -1.0, 1.0))
			dist_s = float(np.clip(d / max(1.0, self._max_dist), 0.0, 1.0))
		else:
			dx_s = 0.0
			dy_s = 0.0
			dist_s = 1.0

		# health normalized to [0,1]
		max_hp = max(1.0, getattr(p, 'max_health', 10.0))
		health = float(np.clip(p.health / max_hp, 0.0, 1.0))

		# phase normalized to [0,1]
		phase = float(np.clip(self.phase, 0.0, self.max_phase) / self.max_phase)

		if self.control_mode == "rotation":
			# Rotation mode: include relative angle (15 dimensions)
			obs = np.array([px, py, vx, vy, a_s, a_c, dx_e, dy_e, dist_e, dx_s, dy_s, dist_s, health, phase, rel_angle_norm], dtype=np.float32)
		else:
			# Directional mode: exclude relative angle (14 dimensions)
			obs = np.array([px, py, vx, vy, a_s, a_c, dx_e, dy_e, dist_e, dx_s, dy_s, dist_s, health, phase], dtype=np.float32)
		return obs

	def step(self, action):
		assert self.player is not None
		p = self.player
		terminated = False
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
				ns = min(active_spawners, key=lambda s: (s.x - p.x) ** 2 + (s.y - p.y) ** 2)
				d = math.hypot(ns.x - p.x, ns.y - p.y) / max(1.0, self._max_dist)
				return float(np.clip(d, 0.0, 1.0))
			
			# FALLBACK: Only if no spawners (e.g. momentarily between phases), look at enemies
			elif self.enemies:
				ne = min(self.enemies, key=lambda e: (e.x - p.x) ** 2 + (e.y - p.y) ** 2)
				d = math.hypot(ne.x - p.x, ne.y - p.y) / max(1.0, self._max_dist)
				return float(np.clip(d, 0.0, 1.0))
			
			return 1.0  # No targets

		prev_target_dist = _nearest_norm_dist()

		# apply action via Player.apply_action (adds inertia instead of snapping velocities)
		shoot_request = None
		if hasattr(p, 'apply_action'):
			shoot_request = p.apply_action(action, self.control_mode, dt)
		else:
			# fallback to previous behavior (shouldn't happen)
			if self.control_mode == "rotation":
				if action == 1:
					accel = 200.0
					p.vx += math.cos(p.angle) * accel * dt
					p.vy += math.sin(p.angle) * accel * dt
				elif action == 2:
					p.angle -= 3.0 * dt
				elif action == 3:
					p.angle += 3.0 * dt
				elif action == 4:
					shoot_request = True
			else:
				speed = 160.0
				if action == 1:
					p.vy = -speed
					p.vx = 0
				elif action == 2:
					p.vy = speed
					p.vx = 0
				elif action == 3:
					p.vx = -speed
					p.vy = 0
				elif action == 4:
					p.vx = speed
					p.vy = 0
				elif action == 5:
					shoot_request = True

		# handle shooting request with player's cooldown
		if shoot_request == "shoot" or shoot_request is True:
			if hasattr(p, 'can_shoot') and p.can_shoot():
				# consume shoot cooldown immediately and spawn bullet
				if hasattr(p, 'consume_shoot'):
					p.consume_shoot()
				self._player_shoot()
				# Add aim-alignment reward ONLY in rotation mode
				# In directional mode, facing is tied to movement direction, so aim reward creates
				# conflicting gradients between "navigate to spawner" and "face enemies"
				if self.control_mode == "rotation":
					# If we're very close to an active spawner, skip aim-based penalty/reward
					active_spawners = [s for s in self.spawners if s.is_active()]
					dist_to_spawner = None
					if active_spawners:
						ns = min(active_spawners, key=lambda s: (s.x - p.x) ** 2 + (s.y - p.y) ** 2)
						dist_to_spawner = math.hypot(ns.x - p.x, ns.y - p.y) / max(1.0, self._max_dist)
					# when close to spawner, don't apply aim alignment reward/penalty (encourage spawner-focused shooting)
					if active_spawners and dist_to_spawner is not None and dist_to_spawner < 0.2:
						# no aim penalty near spawner
						pass
					else:
						aim_reward = self._aim_alignment_reward()
						reward += aim_reward
						# Penalize shooting when misaligned (reduces spam behavior)
						if aim_reward < 0.05:
							reward -= 0.1
				# Emergency shooting bonus: small positive incentive when spawners exist and enemies are present
				if self.spawners and any(s.is_active() for s in self.spawners) and self.enemies:
					reward += 0.05

		# update player
		# update player (applies drag and clamps)
		p.update(dt, self.width, self.height)

		# update spawners and collect spawned enemies
		for sp in list(self.spawners):
			if not sp.is_active():
				continue
			new_enemies = sp.update(dt, self.phase)
			if new_enemies:
				self.enemies.extend(new_enemies)

		# update enemies and check collisions with player
		for e in list(self.enemies):
			e.update(dt, p.x, p.y)
			# remove enemies that travel far off-screen to keep observations clean
			margin = 64
			if e.x < -margin or e.x > self.width + margin or e.y < -margin or e.y > self.height + margin:
				try:
					self.enemies.remove(e)
				except ValueError:
					pass
				continue
			if e.collides_with(p):
				# enemy damages player and is removed
				p.take_damage(1.0)
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
							# spawn some particles on enemy death and apply camera shake
							self._spawn_particles(e.x, e.y, (255, 160, 40, 255), count=12, size=2, life=0.6)
							self.camera_shake = max(self.camera_shake, 6.0)
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
							# spawn a larger explosion and clear enemies
							self._spawn_particles(sp.x, sp.y, (200, 100, 255, 255), count=30, size=3, life=1.0)
							self.camera_shake = max(self.camera_shake, 10.0)
							self.enemies.clear()
						else:
							reward += self.REWARD_SPAWNER_HIT  # Reward damaging spawner
							# Extra small bonus when player is close to the spawner (encourage finishing/pressing the objective)
							try:
								dist_to_spawner = math.hypot(sp.x - p.x, sp.y - p.y) / max(1.0, self._max_dist)
								if dist_to_spawner < 0.2:
									reward += 0.5
							except Exception:
								# safety: if anything goes wrong, keep the base spawner hit reward
								pass
							# small hit vfx
							self._spawn_particles(sp.x, sp.y, (200, 100, 255, 200), count=8, size=2, life=0.5)
						break


		# progress phase when all spawners inactive
		if all(not s.active for s in self.spawners):
			self.phase += 1
			reward += self.REWARD_PHASE_COMPLETE
			# revive spawners with slightly increased difficulty
			self.spawners = []
			count = min(self.phase, len(self.spawner_positions), self.max_spawners)
			for i, pos in enumerate(self.spawner_positions[:count]):
				sp = Spawner(pos[0], pos[1])
				# gradually decrease spawn intervals using tunables
				sp.spawn_interval = max(SPAWNER_MIN_SPAWN_INTERVAL, SPAWNER_BASE_SPAWN_INTERVAL - SPAWNER_SPAWN_DECAY * (self.phase - 1))
				sp.cooldown = sp.spawn_interval * SPAWNER_COOLDOWN_FACTOR
				# gradually increase spawner health using tunables
				sp.health = SPAWNER_BASE_HEALTH + SPAWNER_HEALTH_PER_PHASE * (self.phase - 1)
				sp.max_health = sp.health
				self.spawners.append(sp)

		# REMOVE living penalty to eliminate time pressure
		# reward += self.REWARD_LIVING_PENALTY

		# --- Penalize corner camping ---
		corner_margin = 80  # Distance from edge that counts as "corner"
		in_corner_x = (p.x < corner_margin) or (p.x > self.width - corner_margin)
		in_corner_y = (p.y < corner_margin) or (p.y > self.height - corner_margin)
		if in_corner_x and in_corner_y:
			reward += self.REWARD_CORNER_PENALTY  # Punish corner camping

		# Small penalty for pushing/running into arena walls (helps rotation agents avoid ramming)
		# If player is at/near the boundary and velocity points into the wall, penalize slightly.
		wall_thresh = 1.0
		# left wall
		if p.x <= p.radius + wall_thresh and p.vx < -1e-3:
			reward += self.REWARD_WALL_PENALTY
		# right wall
		if p.x >= self.width - p.radius - wall_thresh and p.vx > 1e-3:
			reward += self.REWARD_WALL_PENALTY
		# top wall
		if p.y <= p.radius + wall_thresh and p.vy < -1e-3:
			reward += self.REWARD_WALL_PENALTY
		# bottom wall
		if p.y >= self.height - p.radius - wall_thresh and p.vy > 1e-3:
			reward += self.REWARD_WALL_PENALTY

		# --- distance shaping reward (small). Encourage moving closer to the nearest target (spawner priority)
		# Compute new nearest-target distance and reward proportionally to the reduction
		new_target_dist = _nearest_norm_dist()
		shaping_reward = self.REWARD_SHAPING_COEF * (prev_target_dist - new_target_dist)
		# Prevent oscillation near the target: clamp shaping to non-negative when already very close
		if new_target_dist < 0.1:
			shaping_reward = max(0.0, shaping_reward)
		# Allow negative shaping (punish moving away from spawner) to enforce focus when not extremely close
		reward += float(shaping_reward)
		# store for next step
		self._last_target_dist = new_target_dist

		# Small stall penalty to discourage waiting/loitering (breaks infinite waiting)
		reward -= 0.002

		# Enemy proximity danger penalty: standing still near enemies should be bad
		if self.enemies:
			nearest_enemy_dist = min(math.hypot(e.x - p.x, e.y - p.y) for e in self.enemies) / max(1.0, self._max_dist)
			if nearest_enemy_dist < 0.15:
				reward -= 0.2

		# --- Directional arrival control (anti-overshoot) ---
		if self.control_mode == "directional":
			active_spawners = [s for s in self.spawners if s.is_active()]
			if active_spawners:
				ns = min(active_spawners, key=lambda s: (s.x - p.x) ** 2 + (s.y - p.y) ** 2)
				dist = math.hypot(ns.x - p.x, ns.y - p.y) / max(1.0, self._max_dist)

				# Only care when close to target
				if dist < 0.15:
					speed = math.hypot(p.vx, p.vy) / self.max_player_speed
					# Reward slowing down near target (arrival shaping)
					reward += 0.4 * (1.0 - speed)
					# Penalize high velocity near the target to encourage stopping
					if speed > 0.6:
						reward -= 0.2

		# death
		if p.health <= 0:
			terminated = True
			reward += self.REWARD_DEATH

		if self.steps >= self.max_steps:
			terminated = True

		# update particles and camera shake
		alive_particles = []
		for p in self.particles:
			if p.update(dt):
				alive_particles.append(p)
		self.particles = alive_particles
		# decay camera shake
		self.camera_shake *= self._shake_decay
		# advance background time for parallax animation
		self._bg_time += dt
		for layer in self.bg_layers:
			# move offsets horizontally (wrap-around handled in render)
			layer['offset_x'] += layer['speed'] * dt * 8.0


		obs = self._get_obs()
		truncated = False  # For gymnasium compatibility
		return obs, reward, terminated, truncated, info
	def _aim_alignment_reward(self):
		"""Calculate reward for aiming toward nearest enemy when shooting.
		Returns positive reward proportional to how well-aligned the shot is.
		This provides immediate feedback for the shooting action."""
		p = self.player
		if not self.enemies:
			return 0.0
		
		# Find nearest enemy
		ne = min(self.enemies, key=lambda e: (e.x - p.x)**2 + (e.y - p.y)**2)
		dx = ne.x - p.x
		dy = ne.y - p.y
		dist = math.hypot(dx, dy)
		if dist < 1e-5:
			return 0.0
		
		# Player facing vector
		fx = math.cos(p.angle)
		fy = math.sin(p.angle)
		
		# Normalized direction to enemy
		tx = dx / dist
		ty = dy / dist
		
		# Cosine similarity ∈ [-1,1]
		cos_sim = fx * tx + fy * ty
		
		# Reward only if roughly aiming at enemy (positive alignment)
		# Scale: 0.3 max reward when perfectly aligned
		return max(0.0, cos_sim) * 0.3
	def _player_shoot(self):
		assert self.player is not None
		speed = 350.0  # bullet speed (increased from 250.0 for better hit probability)
		ang = self.player.angle
		vx = math.cos(ang) * speed
		vy = math.sin(ang) * speed
		b = Bullet(self.player.x + math.cos(ang) * (self.player.radius + 4), self.player.y + math.sin(ang) * (self.player.radius + 4), vx, vy)
		self.bullets.append(b)

		# small muzzle flash particles
		mx = self.player.x + math.cos(ang) * (self.player.radius + 6)
		my = self.player.y + math.sin(ang) * (self.player.radius + 6)
		self._spawn_particles(mx, my, (255, 220, 120, 255), count=6, size=2, life=0.12, spread=30)
		self.camera_shake = max(self.camera_shake, 1.2)


	def _spawn_particles(self, x, y, color, count=8, size=2, life=0.6, spread=40):
		"""Spawn simple particles around (x,y)."""
		for i in range(count):
			a = random.uniform(0, 2 * math.pi)
			r = random.random() ** 0.5 * spread
			vx = math.cos(a) * r
			vy = math.sin(a) * r
			p = Particle(x, y, vx, vy, life, size, color)
			self.particles.append(p)

	def render(self):
		assert self.player is not None
		if self.screen is None:
			self.screen = pygame.display.set_mode((self.width, self.height))
		# draw world to an offscreen surface so we can apply camera shake
		world = pygame.Surface((self.width, self.height), pygame.SRCALPHA).convert_alpha()
		# galaxy background: draw parallax star layers
		for layer in self.bg_layers:
			s = layer['surf']
			offx = int(layer['offset_x']) % self.width
			# blit twice for wrap-around
			world.blit(s, (-offx, 0))
			if offx > 0:
				world.blit(s, (self.width - offx, 0))

		# draw spawners
		for sp in self.spawners:
			sp.draw(world)

		# draw enemies
		for e in self.enemies:
			e.draw(world)

		# draw bullets
		for b in self.bullets:
			b.draw(world)

		# draw player (triangle)
		self.player.draw(world)

		# draw particles on top
		for p in list(self.particles):
			p.draw(world)

		# compute camera shake offset
		off_x = int(round(random.uniform(-1.0, 1.0) * self.camera_shake))
		off_y = int(round(random.uniform(-1.0, 1.0) * self.camera_shake))
		# blit world with offset
		self.screen.fill((0, 0, 0))
		self.screen.blit(world, (off_x, off_y))

		# HUD
		font = pygame.font.SysFont(None, 20)
		text_color = PALETTE.get("hud_text", (230, 230, 230))
		# draw semi-transparent background for HUD for readability
		hud_text = f"Health: {self.player.health:.1f}  Phase: {self.phase}  Enemies: {len(self.enemies)}"
		hud = font.render(hud_text, True, text_color)
		pad_x = 8
		pad_y = 6
		hud_w = hud.get_width() + pad_x * 2
		hud_h = hud.get_height() + pad_y * 2
		hud_surf = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA).convert_alpha()
		hud_bg = PALETTE.get("hud_bg", (0, 0, 0, 150))
		hud_surf.fill(hud_bg)
		hud_surf.blit(hud, (pad_x, pad_y))
		self.screen.blit(hud_surf, (6, 6))

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
		obs, r, terminated = result[0], result[1], result[2]
		env.render()
		if terminated:
			print("Episode finished — resetting environment")
			obs, _ = env.reset()

	env.close()

