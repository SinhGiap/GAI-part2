import math
import random
import pygame


class Player:
    """Player entity with Pygame rendering and physics."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = 0.0  # facing direction in radians
        self.radius = 12
        self.health = 15.0  # Increased from 10 - more forgiving
        self.max_health = 15.0
        self.color = (100, 200, 255)
        
        # shooting cooldown
        self.shoot_cooldown = 0.0
        self.shoot_interval = 0.15  # Faster shooting - was 0.25

    def apply_action(self, action, control_mode, dt):
        """Apply action with inertia-based movement."""
        shoot_request = None
        
        if control_mode == "rotation":
            # actions: 0=noop, 1=thrust, 2=rotate_left, 3=rotate_right, 4=shoot
            if action == 1:  # thrust
                accel = 300.0  # Increased from 200 - faster movement
                self.vx += math.cos(self.angle) * accel * dt
                self.vy += math.sin(self.angle) * accel * dt
            elif action == 2:  # rotate left
                self.angle -= 5.0 * dt  # Faster rotation (was 3.0)
            elif action == 3:  # rotate right
                self.angle += 5.0 * dt  # Faster rotation (was 3.0)
            elif action == 4:  # shoot
                shoot_request = "shoot"
        else:  # directional
            # actions: 0=noop, 1=up, 2=down, 3=left, 4=right, 5=shoot
            accel = 300.0
            if action == 1:  # up
                self.vy -= accel * dt
            elif action == 2:  # down
                self.vy += accel * dt
            elif action == 3:  # left
                self.vx -= accel * dt
            elif action == 4:  # right
                self.vx += accel * dt
            elif action == 5:  # shoot
                shoot_request = "shoot"
            
            # update angle to face movement direction for directional mode
            if self.vx != 0 or self.vy != 0:
                self.angle = math.atan2(self.vy, self.vx)
        
        return shoot_request

    def can_shoot(self):
        """Check if player can shoot (cooldown expired)."""
        return self.shoot_cooldown <= 0.0

    def consume_shoot(self):
        """Reset shoot cooldown."""
        self.shoot_cooldown = self.shoot_interval

    def update(self, dt, width, height):
        """Update player physics with drag and boundary clamping."""
        # apply drag
        drag = 0.95
        self.vx *= drag
        self.vy *= drag
        
        # clamp velocity
        max_speed = 400.0
        speed = math.hypot(self.vx, self.vy)
        if speed > max_speed:
            self.vx = (self.vx / speed) * max_speed
            self.vy = (self.vy / speed) * max_speed
        
        # update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # clamp to arena bounds
        self.x = max(self.radius, min(width - self.radius, self.x))
        self.y = max(self.radius, min(height - self.radius, self.y))
        
        # update shoot cooldown
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= dt

    def take_damage(self, damage):
        """Apply damage to player."""
        self.health -= damage

    def is_dead(self):
        """Check if player is dead."""
        return self.health <= 0

    def draw(self, screen):
        """Draw player as a triangle pointing in the facing direction."""
        # calculate triangle vertices
        tip_x = self.x + math.cos(self.angle) * self.radius
        tip_y = self.y + math.sin(self.angle) * self.radius
        
        # left and right base points
        base_angle_left = self.angle + 2.5
        base_angle_right = self.angle - 2.5
        base_dist = self.radius * 0.6
        
        left_x = self.x + math.cos(base_angle_left) * base_dist
        left_y = self.y + math.sin(base_angle_left) * base_dist
        right_x = self.x + math.cos(base_angle_right) * base_dist
        right_y = self.y + math.sin(base_angle_right) * base_dist
        
        vertices = [(tip_x, tip_y), (left_x, left_y), (right_x, right_y)]
        pygame.draw.polygon(screen, self.color, vertices)
        
        # draw health bar
        bar_width = 30
        bar_height = 4
        health_ratio = max(0.0, min(1.0, self.health / self.max_health))
        bar_x = self.x - bar_width / 2
        bar_y = self.y - self.radius - 10
        
        # background (red)
        pygame.draw.rect(screen, (200, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        # foreground (green)
        pygame.draw.rect(screen, (50, 200, 50), (bar_x, bar_y, bar_width * health_ratio, bar_height))


class Enemy:
    """Enemy entity that moves toward player."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.radius = 8
        self.health = 1.0
        self.max_health = 1.0
        self.speed = 60.0  # Slower enemies - easier to avoid
        self.color = (220, 80, 80)

    def update(self, dt, player_x, player_y):
        """Move toward player."""
        dx = player_x - self.x
        dy = player_y - self.y
        dist = math.hypot(dx, dy)
        
        if dist > 0:
            self.vx = (dx / dist) * self.speed
            self.vy = (dy / dist) * self.speed
        
        self.x += self.vx * dt
        self.y += self.vy * dt

    def take_damage(self, damage):
        """Apply damage to enemy."""
        self.health -= damage

    def is_dead(self):
        """Check if enemy is dead."""
        return self.health <= 0

    def collides_with(self, other):
        """Check collision with another entity."""
        dx = self.x - other.x
        dy = self.y - other.y
        dist = math.hypot(dx, dy)
        return dist < (self.radius + other.radius)

    def draw(self, screen):
        """Draw enemy as a red circle."""
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


class Spawner:
    """Spawner that periodically creates enemies."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 16
        self.health = 3.0
        self.max_health = 3.0
        self.active = True
        self.spawn_interval = 2.5  # seconds between spawns
        self.cooldown = 0.0
        self.color = (180, 100, 180)

    def update(self, dt, phase):
        """Update spawner and spawn enemies."""
        if not self.active:
            return []
        
        self.cooldown -= dt
        spawned = []
        
        if self.cooldown <= 0:
            # spawn enemy at a random angle around the spawner
            angle = random.uniform(0, 2 * math.pi)
            offset = self.radius + 20
            ex = self.x + math.cos(angle) * offset
            ey = self.y + math.sin(angle) * offset
            enemy = Enemy(ex, ey)
            # enemies get slightly faster with phase
            enemy.speed = 80.0 + 10.0 * (phase - 1)
            spawned.append(enemy)
            self.cooldown = self.spawn_interval
        
        return spawned

    def take_damage(self, damage):
        """Apply damage to spawner."""
        self.health -= damage
        if self.health <= 0:
            self.active = False

    def is_active(self):
        """Check if spawner is still active."""
        return self.active

    def collides_with(self, other):
        """Check collision with another entity."""
        dx = self.x - other.x
        dy = self.y - other.y
        dist = math.hypot(dx, dy)
        return dist < (self.radius + getattr(other, 'radius', 0))

    def draw(self, screen):
        """Draw spawner as a purple square with health bar."""
        if not self.active:
            return
        
        # draw square
        half_size = self.radius
        rect = pygame.Rect(self.x - half_size, self.y - half_size, half_size * 2, half_size * 2)
        pygame.draw.rect(screen, self.color, rect)
        pygame.draw.rect(screen, (255, 255, 255), rect, 2)  # border
        
        # health bar
        bar_width = 30
        bar_height = 4
        health_ratio = max(0.0, min(1.0, self.health / self.max_health))
        bar_x = self.x - bar_width / 2
        bar_y = self.y - self.radius - 10
        
        pygame.draw.rect(screen, (100, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(screen, (100, 200, 100), (bar_x, bar_y, bar_width * health_ratio, bar_height))


class Bullet:
    """Bullet projectile."""

    def __init__(self, x, y, vx, vy, owner="player"):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = 3
        self.owner = owner  # "player" or "enemy"
        self.lifetime = 3.0  # seconds before auto-removal
        self.color = (255, 255, 100) if owner == "player" else (255, 100, 100)

    def update(self, dt):
        """Update bullet position and lifetime."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.lifetime -= dt
        return self.lifetime > 0

    def collides_with(self, other):
        """Check collision with another entity."""
        dx = self.x - other.x
        dy = self.y - other.y
        dist = math.hypot(dx, dy)
        return dist < (self.radius + getattr(other, 'radius', 0))

    def draw(self, screen):
        """Draw bullet as a small circle."""
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
