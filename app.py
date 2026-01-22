import pygame
import random
import numpy as np
import math
import json
import os
from concurrent.futures import ThreadPoolExecutor

# --- CONFIG ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 650
CREATURE_COUNT = 30
CREATURE_LIMIT = 100
FOOD_COUNT = 300
REPRODUCE_ENERGY = 170
START_ENERGY = 140
SCENT_GRID_SIZE = 20
SCENT_DECAY = 0.95
SCENT_STRENGTH = 17.0
VISION_SEGMENTS = 36
VISION_RANGE = 200
GENERATION_STEPS = 1000  # cycles per generation
SCENT_SPREAD_RADIUS = 6
SCENT_FALLOFF = 1.5
SHOW_SCENT = True

MODEL_SAVE_PATH = "newest"

# --- INIT ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
log_data = []

# --- SCENT MAP ---
scent_map = np.zeros((SCREEN_WIDTH // SCENT_GRID_SIZE, SCREEN_HEIGHT // SCENT_GRID_SIZE))
poop_map = np.zeros_like(scent_map)

def update_scent_map():
    global scent_map, poop_map
    scent_map *= SCENT_DECAY
    poop_map *= SCENT_DECAY

def deposit_scent(x, y):
    gx = int(x) // SCENT_GRID_SIZE
    gy = int(y) // SCENT_GRID_SIZE
    for dx in range(-SCENT_SPREAD_RADIUS, SCENT_SPREAD_RADIUS + 1):
        for dy in range(-SCENT_SPREAD_RADIUS, SCENT_SPREAD_RADIUS + 1):
            nx, ny = gx + dx, gy + dy
            if 0 <= nx < scent_map.shape[0] and 0 <= ny < scent_map.shape[1]:
                dist = math.sqrt(dx*dx + dy*dy)
                if dist <= SCENT_SPREAD_RADIUS:
                    strength = SCENT_STRENGTH * math.exp(-dist * SCENT_FALLOFF)
                    scent_map[nx, ny] += strength

def deposit_poop(x, y):
    gx = int(x) // SCENT_GRID_SIZE
    gy = int(y) // SCENT_GRID_SIZE
    if 0 <= gx < poop_map.shape[0] and 0 <= gy < poop_map.shape[1]:
        poop_map[gx, gy] += SCENT_STRENGTH

def get_local_scent(x, y):
    gx = int(x) // SCENT_GRID_SIZE
    gy = int(y) // SCENT_GRID_SIZE
    if 0 <= gx < scent_map.shape[0] and 0 <= gy < scent_map.shape[1]:
        return scent_map[gx, gy] / 100.0
    return 0.0

def get_local_poop(x, y):
    gx = int(x) // SCENT_GRID_SIZE
    gy = int(y) // SCENT_GRID_SIZE
    if 0 <= gx < poop_map.shape[0] and 0 <= gy < poop_map.shape[1]:
        return poop_map[gx, gy] / 100.0
    return 0.0

def draw_scent_map():
    surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    for gx in range(scent_map.shape[0]):
        for gy in range(scent_map.shape[1]):
            intensity = scent_map[gx, gy]
            if intensity > 0.1:
                alpha = min(100, int(intensity * 15))
                color = (0, 100, 255, alpha)
                rect = pygame.Rect(gx * SCENT_GRID_SIZE, gy * SCENT_GRID_SIZE, SCENT_GRID_SIZE, SCENT_GRID_SIZE)
                pygame.draw.rect(surface, color, rect)
    screen.blit(surface, (0, 0))

# --- BRAIN ---
class Brain:
    def __init__(self):
        input_size = 13 + VISION_SEGMENTS
        self.w1 = np.random.randn(16, input_size)
        input_size += 4  # unused, but kept for compatibility
        self.w2 = np.random.randn(6, 16)

    def forward(self, inputs):
        h = np.tanh(np.dot(self.w1, inputs))
        out = np.tanh(np.dot(self.w2, h))
        return out

    def mutate(self):
        child = Brain()
        child.w1 = self.w1 + np.random.randn(*self.w1.shape) * 0.2
        child.w2 = self.w2 + np.random.randn(*self.w2.shape) * 0.2
        return child
        
    def to_dict(self):
        """Convert brain weights to JSON-serializable format"""
        return {
            'w1': self.w1.tolist(),
            'w2': self.w2.tolist()
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create brain from dictionary"""
        brain = cls()
        brain.w1 = np.array(data['w1'])
        brain.w2 = np.array(data['w2'])
        return brain

# --- CREATURE ---
class Creature:
    def __init__(self, x, y, brain=None):
        self.x = x
        self.y = y
        self.angle = random.uniform(0, 2 * math.pi)
        self.energy = START_ENERGY
        self.brain = brain if brain else Brain()
        self.dx = 0.0
        self.dy = 0.0
        self.color = (0, 200, 0)
        self.jump_cooldown = 0

    def vision_input(self, food_list):
        result = []
        for i in range(VISION_SEGMENTS):
            rel_angle = (-math.pi / 2) + (i / VISION_SEGMENTS) * math.pi
            dir_angle = self.angle + rel_angle
            tx = self.x + math.cos(dir_angle) * VISION_RANGE
            ty = self.y + math.sin(dir_angle) * VISION_RANGE
            seen = any(math.hypot(fx - tx, fy - ty) < 10 for fx, fy in food_list)
            result.append(1.0 if seen else 0.0)
        return np.array(result)

    def count_nearby_creatures(self, creatures, radius=50):
        count = 0
        for c in creatures:
            if c is not self:
                dist = math.hypot(c.x - self.x, c.y - self.y)
                if dist < radius:
                    count += 1
        return count

    def update(self, food_list, all_creatures):
        if not food_list:
            return 0  # No food eaten

        closest = min(food_list, key=lambda f: math.hypot(f[0] - self.x, f[1] - self.y))
        dx_food, dy_food = closest[0] - self.x, closest[1] - self.y
        dist = math.hypot(dx_food, dy_food)
        angle_to = math.atan2(dy_food, dx_food) - self.angle

        scent = get_local_scent(self.x, self.y)
        poop = get_local_poop(self.x, self.y)

        hunger = 1.0 - (self.energy / REPRODUCE_ENERGY)
        nearby = self.count_nearby_creatures(all_creatures)
        noise = random.uniform(-1, 1)

        inputs = np.concatenate((
            np.array([
                math.cos(angle_to), math.sin(angle_to),
                self.energy / REPRODUCE_ENERGY,
                scent,
                poop,
                math.tanh(dist / 100.0),
                math.sin(self.angle), math.cos(self.angle),
                self.dx, self.dy,
                hunger,
                nearby,
                noise
            ]),
            self.vision_input(food_list)
        ))

        speed, turn, scent_out, poop_out, jump_out, color_out = self.brain.forward(inputs)

        self.angle += turn * 0.5
        move_speed = max(0, speed) * 3
        self.dx = math.cos(self.angle) * move_speed
        self.dy = math.sin(self.angle) * move_speed
        self.x += self.dx
        self.y += self.dy

        # Keep inside screen bounds
        self.x = max(0, min(SCREEN_WIDTH, self.x))
        self.y = max(0, min(SCREEN_HEIGHT, self.y))

        self.energy -= 0.1 * move_speed
        self.energy -= 0.1  # energy drain

        if scent_out > 0.3:
            deposit_scent(self.x, self.y)
        if poop_out > 0.3:
            deposit_poop(self.x, self.y)

        if jump_out > 0.7 and self.jump_cooldown == 0:
            if self.energy > 5:
                self.energy -= 5
                self.x += math.cos(self.angle) * 15
                self.y += math.sin(self.angle) * 15
                self.jump_cooldown = 20
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1

        green_val = int(100 + (color_out + 1) / 2 * 155)
        green_val = max(0, min(255, green_val))
        self.color = (0, green_val, 0)

        food_eaten = 0
        # Eat food if close enough
        for f in food_list:
            if math.hypot(f[0] - self.x, f[1] - self.y) < 10:
                self.energy += 30
                food_list.remove(f)
                food_eaten += 1
                break

        return food_eaten

    def draw(self):
        global picked_creature
        color = self.color
        if self is picked_creature:
            color = (255, 0, 0)  # red highlight if held
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 5)

    @staticmethod
    def mix_brains(brain1, brain2):
        child = Brain()
        # Mix weights by randomly picking gene-wise from parents
        w1_shape = brain1.w1.shape
        mask = np.random.rand(*w1_shape) < 0.5
        child.w1 = np.where(mask, brain1.w1, brain2.w1)

        w2_shape = brain1.w2.shape
        mask2 = np.random.rand(*w2_shape) < 0.5
        child.w2 = np.where(mask2, brain1.w2, brain2.w2)

        # Add mutation noise after crossover
        child.w1 += np.random.randn(*w1_shape) * 0.1
        child.w2 += np.random.randn(*w2_shape) * 0.1

        return child

    def reproduce(self, other_parent=None):
        self.energy /= 2
        if other_parent is None:
            # Just mutate self brain like before
            new_brain = self.brain.mutate()
        else:
            # Mix brains 50/50
            new_brain = Creature.mix_brains(self.brain, other_parent.brain)

        return Creature(self.x, self.y, new_brain)


# --- FOOD ---
def spawn_food(n):
    return [(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)) for _ in range(n)]

# --- MAIN LOOP ---
picked_creature = None  # global holder for picked creature
executor = ThreadPoolExecutor(max_workers=8)

def run_generation(creatures, food):
    global scent_map, poop_map, picked_creature

    start_population = len(creatures)
    food_eaten = 0

    for step in range(GENERATION_STEPS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, creatures, food

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if event.button == 1:  # Left click
                    if picked_creature is None:
                        # Pick creature if close
                        for c in creatures:
                            if math.hypot(c.x - mx, c.y - my) < 10:
                                picked_creature = c
                                break
                    else:
                        # Drop the creature
                        picked_creature.x, picked_creature.y = mx, my
                        picked_creature = None

                elif event.button == 3:  # Right click - add food
                    food.append((mx, my))

        screen.fill((20, 20, 30))
        update_scent_map()
        if SHOW_SCENT:
            draw_scent_map()

        # If holding creature, follow mouse
        if picked_creature is not None:
            mx, my = pygame.mouse.get_pos()
            picked_creature.x, picked_creature.y = mx, my

        # Update creatures in parallel (except picked)
        futures = []
        for c in creatures:
            if c is not picked_creature:
                futures.append(executor.submit(c.update, food, creatures))

        for f in futures:
            food_eaten += f.result()

        # Draw creatures & food
        for c in creatures:
            c.draw()

        for f in food:
            pygame.draw.circle(screen, (200, 150, 0), f, 3)

        pygame.display.flip()
        clock.tick(1000)

    survivors = [c for c in creatures if c.energy > 0]
    survived_count = len(survivors)
    success_rate = survived_count / start_population * 100 if start_population > 0 else 0

    log_data.append({
        'generation': generation,
        'start_population': start_population,
        'food_eaten': food_eaten,
        'survived': survived_count,
        'success_percent': success_rate
    })

    survivors_sorted = sorted(survivors, key=lambda x: x.energy, reverse=True)
    best_6 = survivors_sorted[:6]

    qualified = [c for c in survivors_sorted[6:] if c.energy > REPRODUCE_ENERGY]
    selected_parents = best_6 + qualified

    new_creatures = []
    current_count = len(best_6)

    for c in selected_parents:
        roll = random.random()
        if roll < 0.05:
            babies = 1
        elif roll < 0.80:
            babies = 2
        else:
            babies = 3

        for _ in range(babies):
            if current_count < CREATURE_LIMIT:
                # pick a random other parent different from c
                other_parent = random.choice([p for p in selected_parents if p != c]) if len(selected_parents) > 1 else None
                new_creature = c.reproduce(other_parent)
                new_creatures.append(new_creature)
                current_count += 1

    creatures = best_6 + new_creatures

    scent_map = np.zeros_like(scent_map)
    poop_map = np.zeros_like(poop_map)

    food = spawn_food(FOOD_COUNT)

    with open('simulation_log.json', 'w') as f:
        json.dump(log_data, f, indent=2)

    return True, creatures, food

if os.path.exists(f"{MODEL_SAVE_PATH}.json"):
    with open(f"{MODEL_SAVE_PATH}.json", 'r') as f:
        data = json.load(f)
    brain = Brain.from_dict(data)
    creatures = [Creature(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT), brain=brain) for _ in range(CREATURE_COUNT)]
else:
    creatures = [Creature(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)) for _ in range(CREATURE_COUNT)]

food = spawn_food(FOOD_COUNT)

if MODEL_SAVE_PATH and not os.path.exists(f"{MODEL_SAVE_PATH}.json"):
    # Create empty file if doesn't exist
    with open(f"{MODEL_SAVE_PATH}.json", 'w') as f:
        json.dump({}, f)

running = True
generation = 0
while running:
    generation += 1
    pygame.display.set_caption(f"Generation: {generation} | Creatures: {len(creatures)} | Food: {len(food)}")
    running, creatures, food = run_generation(creatures, food)
    
    # SAVE best brain after each generation:
    if MODEL_SAVE_PATH:
        best_creature = max(creatures, key=lambda c: c.energy, default=None)
        if best_creature:
            with open(f"{MODEL_SAVE_PATH}.json", 'w') as f:
                json.dump(best_creature.brain.to_dict(), f)
            print(f"Model saved to {MODEL_SAVE_PATH}.json (generation {generation})")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
pygame.quit()