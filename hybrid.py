import pygame
import random
import numpy as np
import math
import json
import os
import torch
import torch.nn as nn

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
LAYER_NEURONS = 64  # hidden layer size
MODEL_SAVE_PATH = "newest"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = device.type == "cuda"
print("Using device:", device)

# --- INIT ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
log_data = []

# --- SCENT MAP ---
scent_map = np.zeros((SCREEN_WIDTH // SCENT_GRID_SIZE, SCREEN_HEIGHT // SCENT_GRID_SIZE), dtype=np.float32)
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
                dist = dx*dx + dy*dy
                if dist <= SCENT_SPREAD_RADIUS * SCENT_SPREAD_RADIUS:
                    dist_sqrt = math.sqrt(dist)
                    strength = SCENT_STRENGTH * math.exp(-dist_sqrt * SCENT_FALLOFF)
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
    cell_w = SCENT_GRID_SIZE
    cell_h = SCENT_GRID_SIZE
    for gx in range(scent_map.shape[0]):
        for gy in range(scent_map.shape[1]):
            intensity = scent_map[gx, gy]
            if intensity > 0.1:
                alpha = min(100, int(intensity * 15))
                if alpha <= 0:
                    continue
                color = (0, 100, 255, alpha)
                rect = pygame.Rect(gx * cell_w, gy * cell_h, cell_w, cell_h)
                pygame.draw.rect(surface, color, rect)
    screen.blit(surface, (0, 0))

# --- BRAIN (Hybrid: NumPy on CPU, PyTorch on GPU) ---
class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        # we removed "nearby" from inputs → 12 + vision
        input_size = 12 + VISION_SEGMENTS
        hidden = LAYER_NEURONS

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 6),
            nn.Tanh()
        )

        if USE_GPU:
            self.net.to(device)

        self.refresh_numpy_weights()

    def refresh_numpy_weights(self):
        with torch.no_grad():
            w1 = self.net[0].weight.detach().cpu().numpy()
            b1 = self.net[0].bias.detach().cpu().numpy()
            w2 = self.net[2].weight.detach().cpu().numpy()
            b2 = self.net[2].bias.detach().cpu().numpy()
        self.np_w1 = w1
        self.np_b1 = b1
        self.np_w2 = w2
        self.np_b2 = b2

    def forward_cpu(self, x_np):
        h = np.tanh(self.np_w1 @ x_np + self.np_b1)
        out = np.tanh(self.np_w2 @ h + self.np_b2)
        return out

    def forward_gpu(self, x_np):
        # build tensor directly on device
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        out = self.net(x)
        return out.detach().cpu().numpy()

    def forward(self, x_np):
        if USE_GPU:
            return self.forward_gpu(x_np)
        else:
            return self.forward_cpu(x_np)

    def mutate(self):
        child = Brain()
        with torch.no_grad():
            for p_child, p_self in zip(child.net.parameters(), self.net.parameters()):
                p_child.copy_(p_self + 0.2 * torch.randn_like(p_self))
        child.refresh_numpy_weights()
        return child

    def to_dict(self):
        return {k: v.detach().cpu().numpy().tolist() for k, v in self.net.state_dict().items()}

    @classmethod
    def from_dict(cls, data):
        brain = cls()
        # backward compatibility with old "w1"/"w2" format
        if "w1" in data and "w2" in data:
            with torch.no_grad():
                brain.net[0].weight.copy_(torch.tensor(data["w1"], dtype=torch.float32))
                brain.net[0].bias.zero_()
                brain.net[2].weight.copy_(torch.tensor(data["w2"], dtype=torch.float32))
                brain.net[2].bias.zero_()
        else:
            sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}
            brain.net.load_state_dict(sd)
        brain.refresh_numpy_weights()
        return brain

# --- CREATURE ---
class Creature:
    __slots__ = ("x", "y", "angle", "energy", "brain", "dx", "dy",
                 "color", "jump_cooldown")

    def __init__(self, x, y, brain=None):
        self.x = float(x)
        self.y = float(y)
        self.angle = random.uniform(0, 2 * math.pi)
        self.energy = float(START_ENERGY)
        self.brain = brain if brain else Brain()
        self.dx = 0.0
        self.dy = 0.0
        self.color = (0, 200, 0)
        self.jump_cooldown = 0

    def vision_input(self, food_list, k_nearest=10):
        if not food_list:
            return np.zeros(VISION_SEGMENTS, dtype=np.float32)

        # find up to k nearest food items
        # this is much cheaper than scanning all food for every ray
        dists = []
        sx, sy = self.x, self.y
        for fx, fy in food_list:
            d = (fx - sx) * (fx - sx) + (fy - sy) * (fy - sy)
            dists.append((d, fx, fy))
        dists.sort(key=lambda t: t[0])
        nearest = dists[:k_nearest]

        result = np.zeros(VISION_SEGMENTS, dtype=np.float32)
        for i in range(VISION_SEGMENTS):
            rel_angle = (-math.pi / 2.0) + (i / VISION_SEGMENTS) * math.pi
            dir_angle = self.angle + rel_angle
            dx = math.cos(dir_angle)
            dy = math.sin(dir_angle)

            for d2, fx, fy in nearest:
                vx = fx - sx
                vy = fy - sy
                proj = vx * dx + vy * dy
                if 0 < proj < VISION_RANGE:
                    perp = abs(vx * dy - vy * dx)
                    if perp < 15.0:
                        result[i] = 1.0
                        break
        return result

    def update(self, food_list):
        if not food_list:
            return 0  # No food eaten

        # nearest food (for direction + distance)
        sx, sy = self.x, self.y
        closest = min(food_list, key=lambda f: (f[0] - sx) ** 2 + (f[1] - sy) ** 2)
        dx_food = closest[0] - sx
        dy_food = closest[1] - sy
        dist = math.hypot(dx_food, dy_food)
        angle_to = math.atan2(dy_food, dx_food) - self.angle

        scent = get_local_scent(sx, sy)
        poop = get_local_poop(sx, sy)

        hunger = 1.0 - (self.energy / REPRODUCE_ENERGY)
        noise = random.uniform(-1.0, 1.0)

        base_inputs = np.array([
            math.cos(angle_to),          # 0
            math.sin(angle_to),          # 1
            self.energy / REPRODUCE_ENERGY,  # 2
            scent,                       # 3
            poop,                        # 4
            math.tanh(dist / 100.0),     # 5
            math.sin(self.angle),        # 6
            math.cos(self.angle),        # 7
            self.dx,                     # 8
            self.dy,                     # 9
            hunger,                      # 10
            noise                        # 11
        ], dtype=np.float32)

        vision = self.vision_input(food_list)
        inputs = np.concatenate((base_inputs, vision), dtype=np.float32)

        out = self.brain.forward(inputs)
        speed, turn, scent_out, poop_out, jump_out, color_out = out

        self.angle += float(turn) * 0.5
        move_speed = max(0.0, float(speed)) * 3.0
        self.dx = math.cos(self.angle) * move_speed
        self.dy = math.sin(self.angle) * move_speed
        self.x += self.dx
        self.y += self.dy

        # Keep inside screen bounds
        if self.x < 0:
            self.x = 0.0
        elif self.x > SCREEN_WIDTH:
            self.x = float(SCREEN_WIDTH)
        if self.y < 0:
            self.y = 0.0
        elif self.y > SCREEN_HEIGHT:
            self.y = float(SCREEN_HEIGHT)

        self.energy -= 0.1 * move_speed
        self.energy -= 0.1  # base drain

        if scent_out > 0.3:
            deposit_scent(self.x, self.y)
        if poop_out > 0.3:
            deposit_poop(self.x, self.y)

        if jump_out > 0.7 and self.jump_cooldown == 0:
            if self.energy > 5.0:
                self.energy -= 5.0
                self.x += math.cos(self.angle) * 15.0
                self.y += math.sin(self.angle) * 15.0
                self.jump_cooldown = 20
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1

        green_val = int(100 + (color_out + 1.0) * 0.5 * 155)
        if green_val < 0:
            green_val = 0
        elif green_val > 255:
            green_val = 255
        self.color = (0, green_val, 0)

        food_eaten = 0
        # Eat food if close enough
        # iterate over copy to allow removal
        for f in list(food_list):
            if (f[0] - self.x) ** 2 + (f[1] - self.y) ** 2 < 10 * 10:
                self.energy += 30.0
                food_list.remove(f)
                food_eaten += 1
                break

        return food_eaten

    def draw(self, picked_creature):
        color = self.color if self is not picked_creature else (255, 0, 0)
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 5)

    @staticmethod
    def mix_brains(brain1, brain2):
        child = Brain()
        with torch.no_grad():
            sd1 = brain1.net.state_dict()
            sd2 = brain2.net.state_dict()
            new_sd = {}
            for k in sd1.keys():
                w1 = sd1[k]
                w2 = sd2[k]
                mask = torch.rand_like(w1) < 0.5
                mixed = torch.where(mask, w1, w2)
                mixed = mixed + 0.1 * torch.randn_like(mixed)
                new_sd[k] = mixed
            child.net.load_state_dict(new_sd)
        child.refresh_numpy_weights()
        return child

    def reproduce(self, other_parent=None):
        self.energy *= 0.5
        if other_parent is None:
            new_brain = self.brain.mutate()
        else:
            new_brain = Creature.mix_brains(self.brain, other_parent.brain)
        return Creature(self.x, self.y, new_brain)

# --- FOOD ---
def spawn_food(n):
    return [(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)) for _ in range(n)]

# --- MAIN LOOP ---
picked_creature = None

def run_generation(creatures, food, generation):
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
                        for c in creatures:
                            if (c.x - mx) ** 2 + (c.y - my) ** 2 < 10 * 10:
                                picked_creature = c
                                break
                    else:
                        picked_creature.x, picked_creature.y = float(mx), float(my)
                        picked_creature = None

                elif event.button == 3:  # Right click - add food
                    food.append((mx, my))

        screen.fill((20, 20, 30))
        update_scent_map()
        if SHOW_SCENT:
            draw_scent_map()

        if picked_creature is not None:
            mx, my = pygame.mouse.get_pos()
            picked_creature.x, picked_creature.y = float(mx), float(my)

        # Update creatures
        for c in creatures:
            if c is not picked_creature:
                food_eaten += c.update(food)

        # Draw creatures & food
        for c in creatures:
            c.draw(picked_creature)

        for f in food:
            pygame.draw.circle(screen, (200, 150, 0), f, 3)

        pygame.display.flip()
        # you can lower this if you want raw max speed
        clock.tick(1000)

    survivors = [c for c in creatures if c.energy > 0.0]
    survived_count = len(survivors)
    success_rate = (survived_count / start_population * 100.0) if start_population > 0 else 0.0

    log_data.append({
        "generation": generation,
        "start_population": start_population,
        "food_eaten": food_eaten,
        "survived": survived_count,
        "success_percent": success_rate
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
                if len(selected_parents) > 1:
                    other_parent = random.choice([p for p in selected_parents if p is not c])
                else:
                    other_parent = None
                new_creature = c.reproduce(other_parent)
                new_creatures.append(new_creature)
                current_count += 1

    creatures = best_6 + new_creatures

    scent_map = np.zeros_like(scent_map)
    poop_map = np.zeros_like(poop_map)

    food = spawn_food(FOOD_COUNT)

    with open("simulation_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    return True, creatures, food

# --- LOAD / INIT POPULATION ---
if os.path.exists(f"{MODEL_SAVE_PATH}.json"):
    with open(f"{MODEL_SAVE_PATH}.json", "r") as f:
        data = json.load(f)
    if data:
        brain = Brain.from_dict(data)
        creatures = [Creature(random.randint(0, SCREEN_WIDTH),
                              random.randint(0, SCREEN_HEIGHT),
                              brain=brain)
                     for _ in range(CREATURE_COUNT)]
    else:
        creatures = [Creature(random.randint(0, SCREEN_WIDTH),
                              random.randint(0, SCREEN_HEIGHT))
                     for _ in range(CREATURE_COUNT)]
else:
    creatures = [Creature(random.randint(0, SCREEN_WIDTH),
                          random.randint(0, SCREEN_HEIGHT))
                 for _ in range(CREATURE_COUNT)]

food = spawn_food(FOOD_COUNT)

if MODEL_SAVE_PATH and not os.path.exists(f"{MODEL_SAVE_PATH}.json"):
    with open(f"{MODEL_SAVE_PATH}.json", "w") as f:
        json.dump({}, f)

running = True
generation = 0
while running:
    generation += 1
    pygame.display.set_caption(
        f"Generation: {generation} | Creatures: {len(creatures)} | Food: {len(food)}"
    )
    running, creatures, food = run_generation(creatures, food, generation)

    if MODEL_SAVE_PATH and creatures:
        best_creature = max(creatures, key=lambda c: c.energy)
        with open(f"{MODEL_SAVE_PATH}.json", "w") as f:
            json.dump(best_creature.brain.to_dict(), f)
        print(f"Model saved to {MODEL_SAVE_PATH}.json (generation {generation})")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
