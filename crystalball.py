import time
t0 = time.time()
import pygame
import numpy as np
import pygame.gfxdraw
from numba import njit

JIT = True
DRAW_GRID = False  # slow
USE_GRAVITY = True
BALLS_REPELL = True
show_angles = 0

NJIT_PARALLEL = True
NJIT_CACHE = True
PRINT_FPS = True

ENFORCE_MAX_CALCULATION_FPS = False
MAX_TICK_RATE = 50  # Max tick rate for updating


# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
RADIUS = 400  # Initial radius of the container
BALL_RADIUS = 8  # Radius of the ball bearings
NUM_BALLS = 5000  # Number of ball bearings
GEN_BALLS = 1500
GRAVITY = USE_GRAVITY * 0.001  # Gravity constant
REST_COEFF = 0.8  # Restitution coefficient
FRICTION = 0.998  # Friction coefficient
MIN_DISTANCE = 2 * BALL_RADIUS  # Minimum distance between ball centers
REPULSIVE_FORCE = BALLS_REPELL*0.85*1.0  # Strength of the repulsive force
MAX_ATTEMPTS = 100
VIBRATION_ANGLE = 0
VIBRATION_AMPLITUDE = 0  # Amplitude of the boundary vibration
VIBRATION_FREQUENCY = 2  # Frequency of the boundary vibration
CAPTION = "Ball Bearing Simulation with NumPy and Numba"
CAPTION_VIEWS = ("Hybrid view","Domains view","Defects and boundaries view")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (150, 150, 255)
BLACK = (0, 0, 0)
active = 0

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(CAPTION)

@njit(cache=NJIT_CACHE)
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

@njit(cache=NJIT_CACHE)
def norm(a):
    return np.sqrt(a[0] * a[0] + a[1] * a[1])


@njit(cache=NJIT_CACHE)
def is_valid_position(new_position, existing_positions, active, ball_radius, radius):
    for pos in existing_positions[:active]:
        if norm(new_position - pos) < ball_radius:
            return False
    return norm(new_position - np.array([WIDTH//2, HEIGHT//2])) < radius - ball_radius

@njit(cache=NJIT_CACHE)
def initialize_positions(num_balls, width, radius, ball_radius):
    positions_np = np.zeros((NUM_BALLS,2), dtype = np.float32)
    active = 0
    for _ in range(num_balls):
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            new_position = (np.random.rand(2)) * (2 * (radius - ball_radius)) + (width // 2 - radius + ball_radius)
            if is_valid_position(new_position, positions_np, active, ball_radius, radius):
                positions_np[active,:] = new_position
                active += 1
                break
            attempts += 1
        if attempts == MAX_ATTEMPTS:
            break
    return positions_np, active

t1 = time.time()
print ("pre-init", t1-t0)
positions, active = initialize_positions(GEN_BALLS, WIDTH, RADIUS, BALL_RADIUS)
t2 = time.time()
print ("post-init", t2-t1)
velocities = np.zeros((NUM_BALLS, 2))
colours = np.zeros((NUM_BALLS, 2))
# Grid size for spatial partitioning
GRID_SIZE = 2 * BALL_RADIUS
NUM_CELLS_X = WIDTH // GRID_SIZE
NUM_CELLS_Y = HEIGHT // GRID_SIZE

# Maximum number of balls that can be in one cell (arbitrary large number)
MAX_BALLS_PER_CELL = 10

def draw_grid():
    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            grid_colour = np.clip((255*grid_counts[i,j]//2, 0, (255*grid_counts[i,j]//2),128),0,255)
            pygame.gfxdraw.filled_polygon(screen, ((GRID_SIZE*i,GRID_SIZE*j),(GRID_SIZE*(i+1),GRID_SIZE*j),(GRID_SIZE*(i+1),GRID_SIZE*(j+1)),(GRID_SIZE*i,GRID_SIZE*(j+1))),grid_colour)

def draw_balls(screen, positions, colours):
    for idx, pos in enumerate(positions[:active]):
        if show_angles == 0:
            pygame.gfxdraw.filled_circle(screen, int(pos[0]), int(pos[1]), BALL_RADIUS, (colours[idx,0],colours[idx,1],colours[idx,0]))
        elif show_angles == 1:
            pygame.gfxdraw.filled_circle(screen, int(pos[0]), int(pos[1]), BALL_RADIUS, (255-colours[idx,0],200-(colours[idx,0]*200)//255,255))
        elif show_angles == 2:
            pygame.gfxdraw.filled_circle(screen, int(pos[0]), int(pos[1]), BALL_RADIUS, (255-colours[idx,1],200-(colours[idx,1]*200)//255,255))
        
        pygame.gfxdraw.aacircle(screen, int(pos[0]), int(pos[1]), BALL_RADIUS, BLACK)

@njit(cache=NJIT_CACHE)
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

@njit(cache=NJIT_CACHE)
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
@njit(parallel=NJIT_PARALLEL, cache=NJIT_CACHE)
def reset_grid(grid_counts):
    grid_counts[:, :] = 0

@njit(parallel=NJIT_PARALLEL, cache=NJIT_CACHE)
def update_positions(positions, velocities, grid, grid_counts, grid_size, num_cells_x, num_cells_y, active, time, centre_pos, radius, colours, show_angles):

    # Calculate velocity changes due to gravity and friction
    velocities[:, 1] += GRAVITY
    velocities *= FRICTION

    # Move positions due to their new velocity
    positions[:active] += velocities[:active]
    
    reset_grid(grid_counts)

    # Calculate which grid cell each ball is located inside
    for i in range(active):
        cell_x = int(positions[i, 0] / grid_size)
        cell_y = int(positions[i, 1] / grid_size)
        if cell_x >= 0 and cell_x < num_cells_x and cell_y >= 0 and cell_y < num_cells_y:
            count = grid_counts[cell_x, cell_y]
            if count < MAX_BALLS_PER_CELL:
                grid[cell_x, cell_y, count] = i
                grid_counts[cell_x, cell_y] += 1

    # Calculate collisions with chamber walls
    for i in range(active):
        dist = np.sqrt((positions[i, 0] - centre_pos[0]) ** 2 + (positions[i, 1] - centre_pos[1]) ** 2)
        if dist + BALL_RADIUS > radius:
            normal = np.array([positions[i, 0] - centre_pos[0], positions[i, 1] - centre_pos[1]])
            normal /= dist
            vel_norm = dot(velocities[i], normal)
            velocities[i] -= 2 * vel_norm * normal * REST_COEFF

            overlap = dist + BALL_RADIUS - radius
            positions[i, 0] -= overlap * normal[0]
            positions[i, 1] -= overlap * normal[1]

    for i in range(num_cells_x):
        for j in range(num_cells_y):
            neighbors = [
                (i-1, j-1), (i, j-1), (i+1, j-1),
                (i-1, j),             (i+1, j),
                (i-1, j+1), (i, j+1), (i+1, j+1)
            ]
            count = grid_counts[i, j]
            for k in range(count):
                ball1 = grid[i, j, k]
                distances = []
                angles = []

                # Calculate collisions within grid cell
                for l in range(count):
                    if l == k: continue     # ignore current ball to avoid calculating self-collisions
                    ball2 = grid[i, j, l]
                    dx = positions[ball2, 0] - positions[ball1, 0]
                    dy = positions[ball2, 1] - positions[ball1, 1]
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    distances.append(distance)
                    angles.append(np.mod(np.arctan2(dx,dy),np.pi/3))

                    # Calculate collisions, avoiding repetition of those already calculated
                    if l > k and distance < MIN_DISTANCE:
                        velocities[ball1], velocities[ball2] = resolve_collision(
                            positions[ball1], velocities[ball1], positions[ball2], velocities[ball2]
                        )
                        velocities[ball1], velocities[ball2] = apply_repulsive_force(
                            positions[ball1], positions[ball2], velocities[ball1], velocities[ball2], REPULSIVE_FORCE
                        )

                # Calculate collisions with neighbouring grid cells
                for (nx, ny) in neighbors:
                    if 0 <= nx < num_cells_x and 0 <= ny < num_cells_y:
                        ncount = grid_counts[nx, ny]
                        for l in range(ncount):
                            ball2 = grid[nx, ny, l]
                            dx = positions[ball2, 0] - positions[ball1, 0]
                            dy = positions[ball2, 1] - positions[ball1, 1]
                            distance = np.sqrt(dx ** 2 + dy ** 2)
                            distances.append(distance)
                            angles.append(np.mod(np.arctan2(dx,dy),np.pi/3))

                            # Calculate collisions
                            if distance < MIN_DISTANCE:
                                velocities[ball1], velocities[ball2] = resolve_collision(
                                    positions[ball1], velocities[ball1], positions[ball2], velocities[ball2]
                                )
                                velocities[ball1], velocities[ball2] = apply_repulsive_force(
                                    positions[ball1], positions[ball2], velocities[ball1], velocities[ball2], REPULSIVE_FORCE
                                )
                if len(angles) > 0:
                    colours[ball1,0] = sum(angles) / len(angles)
                else:
                    colours[ball1,0] = 0

                # Get the nearest neighbours, up to six, and calculate the average nearest-neighbour distances.
                # Choose the colour of each ball based on its average nearest-neighbour spacing.
                closest_six = sorted(distances)[:6]
                if len(closest_six) > 0:
                    avg_distance = sum(closest_six)/len(closest_six)    # average the nearest-neighbours' distances
                    colours[ball1,1] = avg_distance - MIN_DISTANCE      # subtract the minimum distance to get the spacing
                else:
                    colours[ball1,1] = 1000   # default distance for isolated balls

    # Store data for the colour scheme of each ball
    colours[:,0] = np.clip(colours[:,0] * 255, 0 , 255)
    colours[:,1] = np.clip(colours[:,1] / BALL_RADIUS * 2550, 0 , 255)
    return colours


def remove_ball(pos, positions):
    global active
    for i in range(active):
        if np.linalg.norm(positions[i] - pos) < BALL_RADIUS:
            positions[i] = positions[active - 1]  # Replace with the last active ball
            active -= 1
            break

def set_caption():
    pygame.display.set_caption(CAPTION + " - " + CAPTION_VIEWS[show_angles])
    
# Initialize grid
grid = np.zeros((NUM_CELLS_X, NUM_CELLS_Y, MAX_BALLS_PER_CELL), dtype=np.int32)
grid_counts = np.zeros((NUM_CELLS_X, NUM_CELLS_Y), dtype=np.int32)

# Main loop
running = True
clock = pygame.time.Clock()

render_interval = 1000 // 30  # Milliseconds per frame
last_render_time = pygame.time.get_ticks()-1
last_calculation_time = last_render_time
i = 0

render_FPS = 0
calculation_FPS = 0
t3 = time.time()
print ("pre-loop", t3-t2)
set_caption()
while running:
    current_time = pygame.time.get_ticks()
    if i == 1:
        print ("first loop", time.time() - t3)
    if PRINT_FPS and i%100 == 1:
        print (f"{calculation_FPS=}, {render_FPS=}")

    i+=1
    try:
        fps = 1000/(current_time-last_calculation_time)
        calculation_FPS = calculation_FPS*0.99+fps*0.01
    except:
        pass
    last_calculation_time = current_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = np.array(event.pos)
            if event.button == 1:  # Left mouse button
                if active < NUM_BALLS:
                    if is_valid_position(pos, positions,active, BALL_RADIUS, RADIUS):
                        positions[active] = pos
                        active += 1
            elif event.button == 2:  # Middle mouse button: Toggle vibration on and off
                if VIBRATION_AMPLITUDE == 0:
                    VIBRATION_AMPLITUDE = 1
                    VIBRATION_ANGLE = np.arctan2(pos[0]-WIDTH//2, pos[1]-HEIGHT//2)
                else:
                    VIBRATION_AMPLITUDE = 0
            elif event.button == 3:  # Right mouse button
                remove_ball(pos, positions)
            elif event.button == 4:
                show_angles += 1
                show_angles = show_angles % 3
                set_caption()
            elif event.button == 5:
                show_angles -= 1
                show_angles = show_angles % 3
                set_caption()

    mouse_buttons = pygame.mouse.get_pressed()
    if mouse_buttons[0]:  # Left mouse button is being held down
        pos = pygame.mouse.get_pos()
        if active < NUM_BALLS and is_valid_position(np.array(pos), positions,active, BALL_RADIUS, RADIUS):
            positions[active] = pos
            active += 1

    # Calculate vibrating radius
    vibrating_radius = RADIUS #+ VIBRATION_AMPLITUDE * np.sin(VIBRATION_FREQUENCY * current_time * 0.001)
    vibrating_position = (WIDTH // 2 + 5 * np.sin(VIBRATION_ANGLE) * VIBRATION_AMPLITUDE * np.sin(VIBRATION_FREQUENCY * current_time * 0.001), HEIGHT // 2 + 5 * np.cos(VIBRATION_ANGLE) * VIBRATION_AMPLITUDE * np.sin(VIBRATION_FREQUENCY * current_time * 0.001))

    # Update and draw balls
    colours = update_positions(positions, velocities, grid, grid_counts, GRID_SIZE, NUM_CELLS_X, NUM_CELLS_Y, active, current_time, vibrating_position, vibrating_radius, colours, show_angles)

    if current_time - last_render_time >= render_interval:
        try:
            fps = 1000/(current_time-last_render_time)
            render_FPS = render_FPS*0.9+fps*0.1
        except:
            pass
        last_render_time = current_time

        last_time = current_time
        screen.fill(WHITE)
        
        # Draw container
        pygame.gfxdraw.aacircle(screen, int(vibrating_position[0]), int(vibrating_position[1]), int(vibrating_radius), RED)

        # Draw balls
        draw_balls(screen, positions, colours)
        if DRAW_GRID:
            draw_grid()

        pygame.display.flip()

    # Control the max tick rate for updating (if needed)
    if ENFORCE_MAX_CALCULATION_FPS:
        clock.tick(MAX_TICK_RATE)

pygame.quit()
