import math
import random
import pygame
import pygame.gfxdraw

# Difficulty tuning constants (enemy/spawner)
ENEMY_BASE_SPEED = 40.0
ENEMY_SPAWN_BASE_SPEED = 50.0
ENEMY_SPEED_PER_PHASE = 5.0
SPAWNER_DEFAULT_SPAWN_INTERVAL = 2.5

# Player tuning
PLAYER_ROTATION_ACCEL = 200.0 # thrust accel when in rotation control
PLAYER_ROTATION_TURN_RATE = 8.0  # radians per second
PLAYER_DIRECTION_ACCEL = 300.0

# Visual palette (4-6 colors)
PALETTE = {
    "bg": (30, 30, 40),
    "primary": (100, 200, 255),
    "enemy": (220, 80, 80),
    "spawner": (180, 100, 180),
    "hud_bg": (0, 0, 0, 150),
    "hud_text": (230, 230, 230),
}


def make_triangle_surface(radius: int, color):
    # create a surface that fits the triangle comfortably
    size = int(radius * 4)
    # create an alpha-capable surface without calling convert_alpha() so
    # sprite creation works before a display mode is set (headless-safe)
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx = cy = size // 2
    # forward vector points to the right on the sprite (angle=0)
    tip = (cx + radius, cy)
    base_back = radius * 0.6
    base_side = radius * 0.6
    base_cx = cx - int(base_back)
    left = (base_cx, cy - int(base_side))
    right = (base_cx, cy + int(base_side))
    # anti-aliased filled triangle via gfxdraw
    pygame.gfxdraw.filled_trigon(surf, tip[0], tip[1], left[0], left[1], right[0], right[1], color)
    pygame.gfxdraw.aatrigon(surf, tip[0], tip[1], left[0], left[1], right[0], right[1], color)
    return surf


def make_circle_surface(radius: int, color):
    size = int(radius * 2 + 4)
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx = cy = size // 2
    pygame.gfxdraw.filled_circle(surf, cx, cy, radius, color)
    pygame.gfxdraw.aacircle(surf, cx, cy, radius, color)
    return surf


def make_square_surface(radius: int, color):
    size = int(radius * 2 + 4)
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    rect = pygame.Rect(2, 2, size - 4, size - 4)
    pygame.draw.rect(surf, color, rect)
    pygame.draw.rect(surf, (255, 255, 255), rect, 2)

    # draw evil face on the spawner
    try:
        eye_w = max(2, radius // 3)
        eye_h = max(1, radius // 4)
        cx = size // 2
        cy = size // 2
        # left eye
        lx1 = cx - radius // 2
        ly1 = cy - radius // 3
        lx2 = lx1 + eye_w
        ly2 = ly1 - eye_h
        lx3 = lx1 + eye_w
        ly3 = ly1 + eye_h
        pygame.gfxdraw.filled_trigon(surf, int(lx1), int(ly1), int(lx2), int(ly2), int(lx3), int(ly3), (20, 20, 20))
        pygame.gfxdraw.aatrigon(surf, int(lx1), int(ly1), int(lx2), int(ly2), int(lx3), int(ly3), (20, 20, 20))

        # right eye
        rx1 = cx + radius // 2
        ry1 = cy - radius // 3
        rx2 = rx1 - eye_w
        ry2 = ry1 - eye_h
        rx3 = rx1 - eye_w
        ry3 = ry1 + eye_h
        pygame.gfxdraw.filled_trigon(surf, int(rx1), int(ry1), int(rx2), int(ry2), int(rx3), int(ry3), (20, 20, 20))
        pygame.gfxdraw.aatrigon(surf, int(rx1), int(ry1), int(rx2), int(ry2), int(rx3), int(ry3), (20, 20, 20))

        # jagged grin
        mouth_y = cy + radius // 6
        mouth_w = radius
        teeth = 5
        for i in range(teeth):
            tx = cx - mouth_w // 2 + int(i * (mouth_w / teeth))
            # teeth
            t1 = (tx, mouth_y)
            t2 = (tx + int(mouth_w / teeth / 2), mouth_y + radius // 5)
            t3 = (tx + int(mouth_w / teeth), mouth_y)
            pygame.gfxdraw.filled_trigon(surf, int(t1[0]), int(t1[1]), int(t2[0]), int(t2[1]), int(t3[0]), int(t3[1]), (240, 240, 240))
        # mouth
        pygame.gfxdraw.aapolygon(surf, [(cx - mouth_w // 2, mouth_y), (cx + mouth_w // 2, mouth_y)], (40, 0, 0))
    except Exception:
        pass
    return surf


class Player:
    """Player entity with Pygame rendering and physics."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = 0.0  # facing direction in radians
        self.radius = 12
        self.health = 15.0  
        self.max_health = 15.0
        self.color = (100, 200, 255)
        
        # shooting cooldown
        self.shoot_cooldown = 0.0
        self.shoot_interval = 0.15  # Faster shooting - was 0.25

        # Pre-rendered sprite for crisp rotation and stable drawing
        try:
            self.sprite = make_triangle_surface(self.radius, PALETTE.get("primary", self.color))
        except Exception:
            # fallback if gfxdraw not available
            self.sprite = None

    def apply_action(self, action, control_mode, dt):
        """Apply action with inertia-based movement."""
        shoot_request = None
        
        if control_mode == "rotation":
            # actions: 0=noop, 1=thrust, 2=rotate_left, 3=rotate_right, 4=shoot
            if action == 1:  # thrust
                # Use a lower thrust so the agent can more easily turn before hitting walls
                accel = PLAYER_ROTATION_ACCEL
                self.vx += math.cos(self.angle) * accel * dt
                self.vy += math.sin(self.angle) * accel * dt
            elif action == 2:  # rotate left
                # Turn faster so the agent can change heading readily
                self.angle -= PLAYER_ROTATION_TURN_RATE * dt
            elif action == 3:  # rotate right
                self.angle += PLAYER_ROTATION_TURN_RATE * dt
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
        # draw drop shadow under player
        try:
            shadow_r = max(2, int(self.radius * 0.6))
            shadow_surf = pygame.Surface((shadow_r * 2 + 4, shadow_r * 2 + 4), pygame.SRCALPHA)
            cx = shadow_surf.get_width() // 2
            cy = shadow_surf.get_height() // 2
            pygame.gfxdraw.filled_circle(shadow_surf, cx, cy, shadow_r, (0, 0, 0, 90))
            screen.blit(shadow_surf, (int(round(self.x)) - cx + 3, int(round(self.y)) - cy + 4))
        except Exception:
            pass

        # If we have a pre-rendered sprite, rotate and blit it with integer center
        sprite = getattr(self, 'sprite', None)
        if sprite is not None:
            # convert angle (radians) to degrees; flip sign to match screen Y
            deg = -math.degrees(self.angle)
            rot = pygame.transform.rotate(sprite, deg)
            rect = rot.get_rect(center=(int(round(self.x)), int(round(self.y))))
            screen.blit(rot, rect)
            return

        # Fallback: draw polygon directly (shouldn't happen normally)
        fx = math.cos(self.angle)
        fy = math.sin(self.angle)
        px = -fy
        py = fx
        tip_x = self.x + fx * self.radius
        tip_y = self.y + fy * self.radius
        base_back = self.radius * 0.6
        base_side = self.radius * 0.6
        base_cx = self.x - fx * base_back
        base_cy = self.y - fy * base_back
        left_x = base_cx + px * base_side
        left_y = base_cy + py * base_side
        right_x = base_cx - px * base_side
        right_y = base_cy - py * base_side
        vertices = [
            (int(round(tip_x)), int(round(tip_y))),
            (int(round(left_x)), int(round(left_y))),
            (int(round(right_x)), int(round(right_y))),
        ]
        pygame.gfxdraw.filled_trigon(screen, vertices[0][0], vertices[0][1], vertices[1][0], vertices[1][1], vertices[2][0], vertices[2][1], self.color)
        pygame.gfxdraw.aatrigon(screen, vertices[0][0], vertices[0][1], vertices[1][0], vertices[1][1], vertices[2][0], vertices[2][1], self.color)
        
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
        self.speed = ENEMY_BASE_SPEED  # Slower enemies - easier to avoid (reduced)
        self.color = (220, 80, 80)
        # prerender enemy sprite
        try:
            self.sprite = make_circle_surface(self.radius, PALETTE.get("enemy", self.color))
        except Exception:
            self.sprite = None

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
        """Draw enemy using pre-rendered sprite for crispness."""
        sprite = getattr(self, 'sprite', None)
        if sprite is not None:
            rect = sprite.get_rect(center=(int(round(self.x)), int(round(self.y))))
            screen.blit(sprite, rect)
            return
        pygame.gfxdraw.filled_circle(screen, int(round(self.x)), int(round(self.y)), self.radius, self.color)
        pygame.gfxdraw.aacircle(screen, int(round(self.x)), int(round(self.y)), self.radius, self.color)


class Spawner:
    """Spawner that periodically creates enemies."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 16
        self.health = 3.0
        self.max_health = 3.0
        self.active = True
        self.spawn_interval = SPAWNER_DEFAULT_SPAWN_INTERVAL  # seconds between spawns
        self.cooldown = 0.0
        self.color = (180, 100, 180)
        try:
            self.sprite = make_square_surface(self.radius, PALETTE.get("spawner", self.color))
        except Exception:
            self.sprite = None
        # animation state for pulsing / spawn-readiness
        self.pulse_timer = 0.0

    def update(self, dt, phase):
        """Update spawner and spawn enemies."""
        if not self.active:
            return []
        # update pulse animation
        self.pulse_timer += dt
        self.cooldown -= dt
        spawned = []
        
        if self.cooldown <= 0:
            # spawn enemy at a random angle around the spawner
            # Ensure enemies originate from inside the spawner area (not outside)
            angle = random.uniform(0, 2 * math.pi)
            # Prefer spawning inside the spawner radius so enemies appear to come
            # out of the spawner. Offset range is [0, max_offset]. If spawner is
            # too small relative to enemy, spawn at center (offset=0).
            enemy_radius = 8
            max_offset = max(0.0, self.radius - enemy_radius - 1.0)
            # random distance from center (uniform) — use sqrt sampling for even area
            if max_offset > 0:
                r = math.sqrt(random.random()) * max_offset
            else:
                r = 0.0
            ex = self.x + math.cos(angle) * r
            ey = self.y + math.sin(angle) * r
            enemy = Enemy(ex, ey)
            # enemies get slightly faster with phase, but start slower
            # base spawn speed smaller to keep early phases easy
            enemy.speed = ENEMY_SPAWN_BASE_SPEED + ENEMY_SPEED_PER_PHASE * (phase - 1)
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
        # draw drop shadow
        shadow_radius = max(2, int(self.radius * 0.6))
        shadow_surf = pygame.Surface((shadow_radius * 2 + 4, shadow_radius * 2 + 4), pygame.SRCALPHA)
        sx = shadow_surf.get_width() // 2
        sy = shadow_surf.get_height() // 2
        pygame.gfxdraw.filled_circle(shadow_surf, sx, sy, shadow_radius, (0, 0, 0, 80))
        screen.blit(shadow_surf, (int(round(self.x)) - sx + 4, int(round(self.y)) - sy + 6))

        # pulsing spawn ring to signal readiness (drawn under the sprite so face stays visible)
        try:
            prog = 0.0
            if self.spawn_interval > 0:
                # progress 0..1 where 1 means ready
                prog = 1.0 - max(0.0, min(1.0, self.cooldown / self.spawn_interval))
            # pulse using timer for a breathing effect
            pulse = 0.5 + 0.5 * math.sin(self.pulse_timer * 6.0)
            ring_r = int(self.radius + 8 + prog * 18 + pulse * 4)
            # stronger alpha and colored glow for visibility
            glow_alpha = int(220 * (0.15 + prog * 0.85))
            ring_color = (255, 200, 100, glow_alpha)
            ring_surf = pygame.Surface((ring_r * 2 + 6, ring_r * 2 + 6), pygame.SRCALPHA)
            # soft glow underneath (low alpha filled circle)
            try:
                pygame.gfxdraw.filled_circle(ring_surf, ring_r + 3, ring_r + 3, ring_r - 2, (ring_color[0], ring_color[1], ring_color[2], max(8, glow_alpha // 8)))
            except Exception:
                pass
            # bright outline (draw multiple concentric aa circles for thickness)
            for offset in range(0, 3):
                a = max(10, int(glow_alpha * (0.6 - 0.18 * offset)))
                try:
                    pygame.gfxdraw.aacircle(ring_surf, ring_r + 3, ring_r + 3, ring_r - offset, (ring_color[0], ring_color[1], ring_color[2], a))
                except Exception:
                    pass
            screen.blit(ring_surf, (int(round(self.x)) - ring_r - 3, int(round(self.y)) - ring_r - 3))
        except Exception:
            pass

        # draw pre-rendered sprite centered (face will be on top of ring)
        sprite = getattr(self, 'sprite', None)
        if sprite is not None:
            rect = sprite.get_rect(center=(int(round(self.x)), int(round(self.y))))
            screen.blit(sprite, rect)
        else:
            half_size = self.radius
            rect = pygame.Rect(int(self.x - half_size), int(self.y - half_size), half_size * 2, half_size * 2)
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
        # previous position for streak drawing
        self.prev_x = x
        self.prev_y = y

    def update(self, dt):
        """Update bullet position and lifetime."""
        # store previous position for visual streak
        self.prev_x = self.x
        self.prev_y = self.y
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
        """Draw bullet as a small circle (antialiased)."""
        # draw streak from prev to current
        x0 = int(round(self.prev_x))
        y0 = int(round(self.prev_y))
        x1 = int(round(self.x))
        y1 = int(round(self.y))
        try:
            # thicker aa line by drawing multiple aalines with slight offsets
            pygame.gfxdraw.aatrigon(screen, x0, y0, x1, y1, x1, y1, self.color)
        except Exception:
            pass
        # fallback: small circle at current
        pygame.gfxdraw.filled_circle(screen, x1, y1, self.radius, self.color)
        pygame.gfxdraw.aacircle(screen, x1, y1, self.radius, self.color)
