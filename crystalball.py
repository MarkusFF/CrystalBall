"""Crystal Ball

An educational ball-bearing simulation of crystal dynamics.
"""

# Standard library imports
import typing
import time
t0 = time.time()    # Initial timestamp

# Third-party imports
import pygame
import pygame.gfxdraw
import numpy as np
from numba import njit
from numba.experimental import jitclass

# Performance config
NJIT_PARALLEL = True
NJIT_CACHE = True
PRINT_FPS = True
ENFORCE_MAX_CALCULATION_FPS = False
MAX_TICK_RATE = 50                          # Max tick rate for updating
RENDER_INTERVAL = 1000 // 50                # Milliseconds per frame

# Constants
WIDTH, HEIGHT = 800, 800                    # On-screen window dimensions
BORDER = 30                                 # On-screen border width within window
BALL_RADIUS = 12                            # Radius of the ball bearings
NUM_BALLS = 5000                            # Maximum number of ball bearings
GEN_BALLS = 500                             # How many balls to show at the beginning
MAX_ATTEMPTS = 100                          # How many attempts to try to place each ball before moving on to the next
MAX_BALLS_PER_CELL = 10                     # Maximum number of balls that can be in one cell of the grid (arbitrary large number)
DRAW_GRID = False                           # Draw the grid? It's slow
USE_GRAVITY = True                          # Simulate gravity?
BALLS_REPELL = True                         # Simulate repulsive forces between balls?
GRAVITY = USE_GRAVITY * 0.001               # Gravity constant
REST_COEFF = 0.8                            # Restitution coefficient
FRICTION = 0.998                            # Friction coefficient
MIN_DISTANCE = 2 * BALL_RADIUS              # Minimum distance between ball centers
REPULSIVE_FORCE = BALLS_REPELL*0.85*1.0     # Strength of the repulsive force
VIBRATION_AMPLITUDE = 1                     # Amplitude of the boundary vibration when switched on
VIBRATION_FREQUENCY = 2                     # Frequency of the boundary vibration
CAPTION = "Ball Bearing Simulation with NumPy and Numba"
CAPTION_VIEWS = ("Simple view", "Hybrid view","Domains view","Defects and boundaries view","Heat map")
NUM_VIEWS = 5
HEATMAP_DECAY = 0.9                         # Retention coefficient to define the time-constant of the heatmap smoothing filter
HEATMAP_SLOPES = (40, 10, 200)              # The colour map gradient for each colour channel in order: RGB

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (150, 150, 255)
BLACK = (0, 0, 0)

# Grid size for spatial partitioning
GRID_SIZE = 2 * BALL_RADIUS
NUM_CELLS_X = WIDTH // GRID_SIZE
NUM_CELLS_Y = HEIGHT // GRID_SIZE

def main():

    # Initialize Pygame
    pygame.init()

    # Create the screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(CAPTION)

    container = Boundary(radius=WIDTH//2 - BORDER)    # Initial radius of the container

    # State variables
    active = 0              # Number of balls being simulated and displayed
    show_angles = 0         # The view being shown at the beginning
    vibration_angle = 0     # Angle/direction of the boundary vibration
    vibration_amplitude = 0 # Amplitude of the boundary vibration

    # Allocate for variables carrying information for each ball
    velocities = np.zeros((NUM_BALLS, 2))
    colours = np.zeros((NUM_BALLS, 2))
    heatmap = np.zeros((NUM_BALLS))

    # Initialize grid
    grid = np.zeros((NUM_CELLS_X, NUM_CELLS_Y, MAX_BALLS_PER_CELL), dtype=np.int32)
    grid_counts = np.zeros((NUM_CELLS_X, NUM_CELLS_Y), dtype=np.int32)

    # Initialize ball positions
    t1 = time.time()
    print ("pre-init", t1-t0)
    positions, active = initialize_positions(GEN_BALLS, BALL_RADIUS, container)
    t2 = time.time()
    print ("post-init", t2-t1)

    # Main loop
    running = True
    clock = pygame.time.Clock()

    last_render_time = pygame.time.get_ticks()-1
    last_calculation_time = last_render_time

    # Initialize performance counters
    i = 0
    render_FPS = 0
    calculation_FPS = 0

    # Update the window title to include the name of the starting view
    set_caption(show_angles)

    # Timestamp before starting the main loop
    t3 = time.time()
    print ("pre-loop", t3-t2)

    # Main loop
    while running:
        current_time = pygame.time.get_ticks()              # Check the clock

        if i == 1:
            print ("first loop", time.time() - t3)          # Print the time of the first loop only
        if PRINT_FPS and i%100 == 1:
            print (f"{calculation_FPS=}, {render_FPS=}")    # Print frames per second for all other iterations

        i += 1                                              # Increment loop counter

        # Calculate frames per second for the behind-the-scenes calculations
        try:
            fps = 1000/(current_time-last_calculation_time)
            calculation_FPS = calculation_FPS*0.99+fps*0.01
        except:
            pass

        last_calculation_time = current_time                # Update timestamps

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = np.array(event.pos)
                if event.button == 1:  # Left mouse button                  : Add a new ball if mouse is in a valid position
                    if active < NUM_BALLS:
                        if container.is_valid_position(pos, positions, active, BALL_RADIUS):
                            positions[active] = pos
                            active += 1
                elif event.button == 2:  # Middle mouse button              : Toggle vibration on and off
                    if vibration_amplitude == 0:
                        vibration_amplitude = VIBRATION_AMPLITUDE
                        vibration_angle = np.arctan2(pos[0]-container.centre[0], pos[1]-container.centre[1])
                    else:
                        vibration_amplitude = 0
                elif event.button == 3:  # Right mouse button               : Remove a ball
                    active = remove_ball(pos, positions, active)
                elif event.button == 4:  # Scroll-wheel forwards/up         : Change view
                    show_angles += 1
                    show_angles = show_angles % NUM_VIEWS
                    set_caption(show_angles)
                elif event.button == 5:  # Scroll-wheel backwards/down      : Change view
                    show_angles -= 1
                    show_angles = show_angles % NUM_VIEWS
                    set_caption(show_angles)

        # Check which mouse buttons are being held down
        mouse_buttons = pygame.mouse.get_pressed()
        
        if mouse_buttons[0]:  # Left mouse button is being held down        : Add more balls!
            pos = pygame.mouse.get_pos()
            if active < NUM_BALLS and container.is_valid_position(pos, positions, active, BALL_RADIUS):
                positions[active] = pos
                active += 1

        # Calculate vibrating radius
        # vibrating_radius = RADIUS #+ vibration_amplitude * np.sin(VIBRATION_FREQUENCY * current_time * 0.001)
        vibrating_position = (container.anchor[0] + 5 * np.sin(vibration_angle) * vibration_amplitude * np.sin(VIBRATION_FREQUENCY * current_time * 0.001), container.anchor[1] + 5 * np.cos(vibration_angle) * vibration_amplitude * np.sin(VIBRATION_FREQUENCY * current_time * 0.001))
        container.centre = vibrating_position

        # Update and draw balls
        positions, velocities, colours = update_positions(positions, velocities, grid, grid_counts, GRID_SIZE, NUM_CELLS_X, NUM_CELLS_Y, active, current_time, container, colours, show_angles)

        # Accelerate performance by rendering the screen after several iterations of the calculations
        if current_time - last_render_time >= RENDER_INTERVAL:

            # Calculate frames per second for the rendering
            try:
                fps = 1000/(current_time-last_render_time)
                render_FPS = render_FPS*0.9+fps*0.1
            except:
                pass

            # Update timestamps
            last_render_time = current_time

            # Start drawing new screen
            screen.fill(WHITE)

            # Draw container
            draw_boundary(screen, container)

            # Draw balls
            heatmap = draw_balls(screen, positions, velocities, colours, heatmap, active, show_angles)

            # Draw the grid onscreen (optional)
            if DRAW_GRID:
                draw_grid(screen, grid_counts)

            # Update the screen
            pygame.display.flip()

        # Control the max tick rate for updating (if needed)
        if ENFORCE_MAX_CALCULATION_FPS:
            clock.tick(MAX_TICK_RATE)

    pygame.quit()


###################################################################################################

# @jitclass
# class BoundaryCircle:
#     anchor: typing.Tuple[float, float]
#     centre: typing.Tuple[float, float]
#     radius: float

#     def __init__(self, radius):
#         self.centre = (WIDTH // 2, HEIGHT // 2)
#         self.anchor = self.centre
#         self.radius = radius

#     def is_valid_position(self, new_position, existing_positions, active, ball_radius):
#         for pos in existing_positions[:active]:
#             if norm(new_position - pos) < ball_radius:
#                 return False
#         return norm(new_position - np.array([self.centre[0], self.centre[1]])) < self.radius - ball_radius
#         # return norm((new_position[0] - self.centre[0],new_position[1] - self.centre[1])) < self.radius - ball_radius
    
#     def get_new_position(self, ball_radius):
#         return np.random.rand(2) * (2 * (self.radius - ball_radius)) + (self.centre[0] - self.radius + ball_radius)
    
#     def distance_outside(self, position):
#         return np.sqrt((position[0] - self.centre[0]) ** 2 + (position[1] - self.centre[1]) ** 2) - self.radius

#     def is_outside(self, position, ball_radius):
#         return self.distance_outside(position) + ball_radius > 0
    
#     def normal(self, position):
#         normal = np.array([position[0] - self.centre[0], position[1] - self.centre[1]])
#         return normal / (self.distance_outside(position) + self.radius)


# def draw_boundary(screen, bound):
#     pygame.gfxdraw.aacircle(screen, int(bound.centre[0]), int(bound.centre[1]), int(bound.radius), RED)

# Try a square boundary
@jitclass
class Boundary:
    anchor: typing.Tuple[float, float]
    centre: typing.Tuple[float, float]
    radius: float

    def __init__(self, radius):
        self.centre = (WIDTH // 2, HEIGHT // 2)
        self.anchor = self.centre
        self.radius = radius
    
    def is_valid_position(self, new_position, existing_positions, active, ball_radius):
        for pos in existing_positions[:active]:
            if norm((new_position[0] - pos[0],new_position[1] - pos[1])) < ball_radius:
                return False
        return not self.is_outside(new_position, ball_radius)
    
    def get_new_position(self, ball_radius):
        return np.random.rand(2) * (2 * (self.radius - ball_radius)) + (self.centre[0] - self.radius + ball_radius)
    
    def distance_outside(self, position):
        vector_outside = np.maximum(abs(position[0] - self.centre[0]),abs(position[1] - self.centre[1]))
        # print(vector_outside - self.radius)
        return vector_outside - self.radius

    def is_outside(self, position, ball_radius):
        #return (ball_radius - self.radius < (self.centre[0] - position[0]) < self.radius - ball_radius) and (ball_radius - self.radius < (self.centre[1] - position[1]) < self.radius - ball_radius)
        # print(self.distance_outside(position) + ball_radius > 0)
        return self.distance_outside(position) + ball_radius > 0
    
    def normal(self, position):
        normal = np.array([position[0] - self.centre[0], position[1] - self.centre[1]])
        return normal / (self.distance_outside(position) + self.radius)


def draw_boundary(screen, bound):
    rectangle = pygame.Rect(bound.centre[0]-bound.radius, bound.centre[1]-bound.radius, 2*bound.radius, 2*bound.radius)
    pygame.gfxdraw.rectangle(screen, rectangle, RED)


###################################################################################################


@njit(cache=NJIT_CACHE)
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

@njit(cache=NJIT_CACHE)
def norm(a):
    return np.sqrt(a[0] * a[0] + a[1] * a[1])

@njit(cache=NJIT_CACHE)
def initialize_positions(num_balls, ball_radius, container):
    positions_np = np.zeros((NUM_BALLS,2), dtype = np.float32)
    active = 0
    for _ in range(num_balls):
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            new_position = container.get_new_position(ball_radius)
            if container.is_valid_position(new_position, positions_np, active, ball_radius):
                positions_np[active,:] = new_position
                active += 1
                break
            attempts += 1
        if attempts == MAX_ATTEMPTS:
            break
    return positions_np, active

def draw_grid(screen, grid_counts):
    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            grid_colour = np.clip((255*grid_counts[i,j]//2, 0, (255*grid_counts[i,j]//2),128),0,255)
            pygame.gfxdraw.filled_polygon(screen, ((GRID_SIZE*i,GRID_SIZE*j),(GRID_SIZE*(i+1),GRID_SIZE*j),(GRID_SIZE*(i+1),GRID_SIZE*(j+1)),(GRID_SIZE*i,GRID_SIZE*(j+1))),grid_colour)

def draw_balls(screen, positions, velocities, colours, heatmap, active, show_angles):
    # For displaying the heatmap view:
    if show_angles == 4:
        # Calculate a rolling average for the heatmap
        heatmap[:active] = HEATMAP_DECAY * heatmap[:active] + np.sum(velocities[:active,:]**2, axis=1)
        # Calculate ball colour scheme
        heatmapR = np.clip(heatmap[:active] * HEATMAP_SLOPES[0] * 255, 0, 255)
        heatmapG = np.clip(heatmap[:active] * HEATMAP_SLOPES[1] * 255, 0, 255)
        heatmapB = np.clip(heatmap[:active] * HEATMAP_SLOPES[2] * 255, 0, 255)

    # Draw each ball
    for idx, pos in enumerate(positions[:active]):
        # Draw the fill of each ball dependent on view being displayed
        if show_angles == 0:
            # Simple view with no colour mapping
            pygame.gfxdraw.filled_circle(screen, int(pos[0]), int(pos[1]), BALL_RADIUS, (0,0,127))
        if show_angles == 1:
            # "Hybrid" view of distance and angle combined
            # pygame.gfxdraw.filled_circle(screen, int(pos[0]), int(pos[1]), BALL_RADIUS, (colours[idx,0],colours[idx,1],colours[idx,0]))
            pygame.gfxdraw.filled_circle(screen, int(pos[0]), int(pos[1]), BALL_RADIUS, (colours[idx,0],colours[idx,1],255-colours[idx,0]))
        elif show_angles == 2:
            # Mean nearest six neighbour distance
            pygame.gfxdraw.filled_circle(screen, int(pos[0]), int(pos[1]), BALL_RADIUS, (255-colours[idx,0],200-(colours[idx,0]*200)//255,255))
        elif show_angles == 3:
            # Crystal plane angle
            pygame.gfxdraw.filled_circle(screen, int(pos[0]), int(pos[1]), BALL_RADIUS, (255-colours[idx,1],200-(colours[idx,1]*200)//255,255))
        elif show_angles == 4:
            # Heat map
            # pygame.gfxdraw.filled_circle(screen, int(pos[0]), int(pos[1]), BALL_RADIUS, (int(heatmapB[idx]),0,255-int(heatmapB[idx])))
            pygame.gfxdraw.filled_circle(screen, int(pos[0]), int(pos[1]), BALL_RADIUS, (int(heatmapR[idx]),int(heatmapG[idx]),int(heatmapB[idx])))

        # Draw ball outline
        pygame.gfxdraw.aacircle(screen, int(pos[0]), int(pos[1]), BALL_RADIUS, BLACK)

    return heatmap

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
def update_positions(positions, velocities, grid, grid_counts, grid_size, num_cells_x, num_cells_y, active, time, container, colours, show_angles):
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
        if container.is_outside(positions[i], BALL_RADIUS):
            normal = container.normal(positions[i])
            vel_norm = dot(velocities[i], normal)
            velocities[i] -= 2 * vel_norm * normal * REST_COEFF

            overlap = container.distance_outside(positions[i]) + BALL_RADIUS
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
                    # (dx, dy) = positions[ball2] - positions[ball1]      # TEN TIMES SLOWER!?
                    distance = norm((dx,dy))
                    
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
                            # (dx, dy) = positions[ball2] - positions[ball1]      # TEN TIMES SLOWER!?
                            distance = norm((dx,dy))
                    
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

                # Normalise the angles and use them as a method of colouring in the balls
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
    return positions, velocities, colours

def remove_ball(pos, positions, active):
    for i in range(active):
        if np.linalg.norm(positions[i] - pos) < BALL_RADIUS:
            positions[i] = positions[active - 1]  # Replace with the last active ball
            active -= 1
            break
    return active

def set_caption(show_angles):
    pygame.display.set_caption(CAPTION + " - " + CAPTION_VIEWS[show_angles])

###################################################################################################

if __name__ == "__main__":
    main()