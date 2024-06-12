import pygame
import numpy as np
import pygame.gfxdraw
from numba import njit
from scipy.spatial import distance

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
RADIUS = 400  # Initial radius of the container
BALL_RADIUS = 8  # Radius of the ball bearings
NUM_BALLS = 5000  # Number of ball bearings
GEN_BALLS = 1500
GRAVITY = 0.001  # Gravity constant
REST_COEFF = 0.8  # Restitution coefficient
FRICTION = 0.998  # Friction coefficient
MIN_DISTANCE = 2 * BALL_RADIUS  # Minimum distance between ball centers
REPULSIVE_FORCE = 0.85  # Strength of the repulsive force
MAX_ATTEMPTS = 100
MAX_TICK_RATE = 120  # Max tick rate for updating
VIBRATION_AMPLITUDE = 3  # Amplitude of the boundary vibration
VIBRATION_FREQUENCY = 3  # Frequency of the boundary vibration

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (150, 150, 255)
BLACK = (0, 0, 0)
active = 0

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ball Bearing Simulation with NumPy and Numba")

def is_valid_position(new_position, existing_positions, ball_radius, radius):
    for pos in existing_positions:
        if np.linalg.norm(new_position - pos) < ball_radius:
            return False
    return np.linalg.norm(new_position - np.array([WIDTH//2, HEIGHT//2])) < radius - ball_radius

def initialize_positions(num_balls, width, radius, ball_radius):
    global active
    positions = []

    for _ in range(num_balls):
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            new_position = (np.random.rand(2)) * (2 * (radius - ball_radius)) + (width // 2 - radius + ball_radius)
            if is_valid_position(new_position, positions, ball_radius, radius):
                positions.append(new_position)
                break
            attempts += 1
        if attempts == MAX_ATTEMPTS:
            active = len(positions)
            print(f"found {active} good positions")
            return np.vstack((np.array(positions), np.zeros((NUM_BALLS - active, 2))))
    active = len(positions)
    return np.vstack((np.array(positions), np.zeros((NUM_BALLS - active, 2))))

positions = initialize_positions(GEN_BALLS, WIDTH, RADIUS, BALL_RADIUS)
velocities = np.zeros((NUM_BALLS, 2))
colours = np.zeros(NUM_BALLS)

# Grid size for spatial partitioning
GRID_SIZE = 2 * BALL_RADIUS
NUM_CELLS_X = WIDTH // GRID_SIZE
NUM_CELLS_Y = HEIGHT // GRID_SIZE

# Maximum number of balls that can be in one cell (arbitrary large number)
MAX_BALLS_PER_CELL = 10

@njit
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

@njit
def norm(a):
    return np.sqrt(a[0] * a[0] + a[1] * a[1])

def draw_balls(screen, positions, colours):
    for idx, pos in enumerate(positions[:active]):
        pygame.gfxdraw.filled_circle(screen, int(pos[0]), int(pos[1]), BALL_RADIUS, (255-colours[idx],255-colours[idx],255))
        pygame.gfxdraw.aacircle(screen, int(pos[0]), int(pos[1]), BALL_RADIUS, BLACK)

@njit
def resolve_collision(pos1, vel1, pos2, vel2):
    normal = pos2 - pos1
    dist = norm(normal)
    if dist == 0:
        return vel1, vel2

    normal = normal / dist

    rel_vel = vel2 - vel1
    vel_along_normal = dot(rel_vel, normal)

    if vel_along_normal > 0:
        return vel1, vel2

    impulse = -(1 + REST_COEFF) * vel_along_normal
    impulse /= 1 / BALL_RADIUS + 1 / BALL_RADIUS

    vel1 -= impulse * normal / BALL_RADIUS
    vel2 += impulse * normal / BALL_RADIUS

    return vel1, vel2

@njit
def apply_repulsive_force(pos1, pos2, vel1, vel2, repulsive_force):
    normal = pos2 - pos1
    dist = norm(normal)
    if dist == 0 or dist >= MIN_DISTANCE:
        return vel1, vel2

    normal = normal / dist
    force_magnitude = repulsive_force * (MIN_DISTANCE - dist) / dist
    force = force_magnitude * normal

    vel1 -= force
    vel2 += force

    return vel1, vel2

# Initialize grid with optimized resetting
@njit(parallel=True)
def reset_grid(grid_counts, num_cells_x, num_cells_y):
    grid_counts[:, :] = 0

@njit(parallel=True)
def update_positions(positions, velocities, grid, grid_counts, grid_size, num_cells_x, num_cells_y, active, time, radius):
    velocities[:, 1] += GRAVITY
    velocities *= FRICTION
    positions[:active] += velocities[:active]
    
    reset_grid(grid_counts, num_cells_x, num_cells_y)

    for i in range(active):
        cell_x = int(positions[i, 0] / grid_size)
        cell_y = int(positions[i, 1] / grid_size)
        if cell_x >= 0 and cell_x < num_cells_x and cell_y >= 0 and cell_y < num_cells_y:
            count = grid_counts[cell_x, cell_y]
            if count < MAX_BALLS_PER_CELL:
                grid[cell_x, cell_y, count] = i
                grid_counts[cell_x, cell_y] += 1

    for i in range(active):
        dist = np.sqrt((positions[i, 0] - WIDTH // 2) ** 2 + (positions[i, 1] - HEIGHT // 2) ** 2)
        if dist + BALL_RADIUS > radius:
            normal = np.array([positions[i, 0] - WIDTH // 2, positions[i, 1] - HEIGHT // 2])
            normal /= dist
            vel_norm = dot(velocities[i], normal)
            velocities[i] -= 2 * vel_norm * normal * REST_COEFF

            overlap = dist + BALL_RADIUS - radius
            positions[i, 0] -= overlap * normal[0]
            positions[i, 1] -= overlap * normal[1]

    for i in range(num_cells_x):
        for j in range(num_cells_y):
            count = grid_counts[i, j]
            for k in range(count):
                for l in range(k + 1, count):
                    ball1 = grid[i, j, k]
                    ball2 = grid[i, j, l]
                    dx = positions[ball2, 0] - positions[ball1, 0]
                    dy = positions[ball2, 1] - positions[ball1, 1]
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    if distance < MIN_DISTANCE:
                        velocities[ball1], velocities[ball2] = resolve_collision(
                            positions[ball1], velocities[ball1], positions[ball2], velocities[ball2]
                        )
                        velocities[ball1], velocities[ball2] = apply_repulsive_force(
                            positions[ball1], positions[ball2], velocities[ball1], velocities[ball2], REPULSIVE_FORCE
                        )

    for i in range(num_cells_x):
        for j in range(num_cells_y):
            neighbors = [
                (i-1, j-1), (i, j-1), (i+1, j-1),
                (i-1, j),            (i+1, j),
                (i-1, j+1), (i, j+1), (i+1, j+1)
            ]
            count = grid_counts[i, j]
            for k in range(count):
                ball1 = grid[i, j, k]
                for (nx, ny) in neighbors:
                    if 0 <= nx < num_cells_x and 0 <= ny < num_cells_y:
                        ncount = grid_counts[nx, ny]
                        for l in range(ncount):
                            ball2 = grid[nx, ny, l]
                            dx = positions[ball2, 0] - positions[ball1, 0]
                            dy = positions[ball2, 1] - positions[ball1, 1]
                            distance = np.sqrt(dx ** 2 + dy ** 2)
                            if distance < MIN_DISTANCE:
                                velocities[ball1], velocities[ball2] = resolve_collision(
                                    positions[ball1], velocities[ball1], positions[ball2], velocities[ball2]
                                )
                                velocities[ball1], velocities[ball2] = apply_repulsive_force(
                                    positions[ball1], positions[ball2], velocities[ball1], velocities[ball2], REPULSIVE_FORCE
                                )

def remove_ball(pos, positions):
    global active
    for i in range(active):
        if np.linalg.norm(positions[i] - pos) < BALL_RADIUS:
            positions[i] = positions[active - 1]  # Replace with the last active ball
            active -= 1
            break

# Initialize grid
grid = np.zeros((NUM_CELLS_X, NUM_CELLS_Y, MAX_BALLS_PER_CELL), dtype=np.int32)
grid_counts = np.zeros((NUM_CELLS_X, NUM_CELLS_Y), dtype=np.int32)

# Main loop
running = True
clock = pygame.time.Clock()

render_interval = 1000 // 60  # Milliseconds per frame
last_time = pygame.time.get_ticks()
i = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = np.array(event.pos)
            if event.button == 1:  # Left mouse button
                if active < NUM_BALLS:
                    if is_valid_position(pos, positions[:active], BALL_RADIUS, RADIUS):
                        positions[active] = pos
                        active += 1
            elif event.button == 3:  # Right mouse button
                remove_ball(pos, positions)

    mouse_buttons = pygame.mouse.get_pressed()
    if mouse_buttons[0]:  # Left mouse button is being held down
        pos = pygame.mouse.get_pos()
        if active < NUM_BALLS and is_valid_position(np.array(pos), positions[:active], BALL_RADIUS, RADIUS):
            positions[active] = pos
            active += 1

    # Calculate vibrating radius
    current_time = pygame.time.get_ticks()
    vibrating_radius = RADIUS # + VIBRATION_AMPLITUDE * np.sin(VIBRATION_FREQUENCY * current_time * 0.001)

    # Update and draw balls
    update_positions(positions, velocities, grid, grid_counts, GRID_SIZE, NUM_CELLS_X, NUM_CELLS_Y, active, current_time, vibrating_radius)

    if current_time - last_time >= render_interval:
        last_time = current_time

        screen.fill(WHITE)

        # Draw container
        pygame.gfxdraw.aacircle(screen, WIDTH // 2, HEIGHT // 2, int(vibrating_radius), RED)

        if i % 10 == 0:
            # Colour balls by nearest neighbour distances
            # Select smallest six distances, then average
            XX = distance.cdist(positions[:active+1], positions[:active+1], 'euclidean')
            XX.sort()
            colours = XX[:,1:7].mean(axis=1) - 2*BALL_RADIUS
            colours = np.clip(colours / BALL_RADIUS * 2550, 0 , 255)
        i += 1

        # Draw balls
        draw_balls(screen, positions, colours)

        pygame.display.flip()

    # Control the max tick rate for updating (if needed)
    # clock.tick(MAX_TICK_RATE)

pygame.quit()
