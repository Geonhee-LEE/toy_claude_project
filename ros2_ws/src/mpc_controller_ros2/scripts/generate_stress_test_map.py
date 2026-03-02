#!/usr/bin/env python3
"""
Generate a PGM map file for the MPPI stress test arena.

Arena: 20m x 20m with walls and static obstacles.
Resolution: 0.05 m/pixel -> 1000x1000 pixels.
Origin: (-25.0, -25.0) -> pixel (0,0) = world (-25, -25).

PGM P5 binary format, nav2 convention:
  - Row 0 = top of image = y_max in world
  - Row 999 = bottom of image = y_min in world
  - Value 0 = occupied (wall/obstacle)
  - Value 254 = free space
"""

import math
import os

# -- Map parameters --
WIDTH = 1000       # pixels
HEIGHT = 1000      # pixels
RESOLUTION = 0.05  # m/pixel
ORIGIN_X = -25.0   # world x at pixel column 0
ORIGIN_Y = -25.0   # world y at pixel row (HEIGHT-1) [bottom of image]

FREE_SPACE = 254
OCCUPIED = 0

# -- Output paths --
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR = os.path.join(SCRIPT_DIR, '..', 'maps')
PGM_PATH = os.path.join(MAPS_DIR, 'stress_test_arena.pgm')
YAML_PATH = os.path.join(MAPS_DIR, 'stress_test_map.yaml')


def world_to_pixel(wx: float, wy: float) -> tuple:
    """Convert world coordinates to pixel (column, row) in image space.

    px (column) = (wx - ORIGIN_X) / RESOLUTION
    py_from_bottom = (wy - ORIGIN_Y) / RESOLUTION
    row = (HEIGHT - 1) - py_from_bottom   (invert for top-to-bottom image)
    """
    px = int((wx - ORIGIN_X) / RESOLUTION)
    py_from_bottom = int((wy - ORIGIN_Y) / RESOLUTION)
    row = (HEIGHT - 1) - py_from_bottom
    return (px, row)


def draw_filled_rect(grid: bytearray, x_min: int, y_min: int,
                     x_max: int, y_max: int, value: int):
    """Draw a filled rectangle on the grid.

    Parameters are in pixel coordinates: (col_min, row_min) to (col_max, row_max).
    """
    for row in range(max(0, y_min), min(HEIGHT, y_max + 1)):
        for col in range(max(0, x_min), min(WIDTH, x_max + 1)):
            grid[row * WIDTH + col] = value


def draw_filled_circle(grid: bytearray, cx: int, cy: int,
                       radius: int, value: int):
    """Draw a filled circle on the grid.

    (cx, cy) = center in pixel coords (col, row).
    """
    for row in range(max(0, cy - radius), min(HEIGHT, cy + radius + 1)):
        for col in range(max(0, cx - radius), min(WIDTH, cx + radius + 1)):
            dx = col - cx
            dy = row - cy
            if dx * dx + dy * dy <= radius * radius:
                grid[row * WIDTH + col] = value


def draw_wall_horizontal(grid: bytearray, world_y: float,
                         world_x_min: float, world_x_max: float,
                         thickness_m: float):
    """Draw a horizontal wall (constant y) with given thickness.

    The wall is centered at world_y, spanning [x_min, x_max].
    Thickness is applied symmetrically around world_y.
    """
    half_t = thickness_m / 2.0
    # World corners
    x0, y0 = world_x_min, world_y - half_t
    x1, y1 = world_x_max, world_y + half_t

    # Convert to pixel
    col_min, row_max = world_to_pixel(x0, y0)  # lower-left -> higher row number
    col_max, row_min = world_to_pixel(x1, y1)  # upper-right -> lower row number

    draw_filled_rect(grid, col_min, row_min, col_max, row_max, OCCUPIED)


def draw_wall_vertical(grid: bytearray, world_x: float,
                       world_y_min: float, world_y_max: float,
                       thickness_m: float):
    """Draw a vertical wall (constant x) with given thickness.

    The wall is centered at world_x, spanning [y_min, y_max].
    """
    half_t = thickness_m / 2.0
    x0, y0 = world_x - half_t, world_y_min
    x1, y1 = world_x + half_t, world_y_max

    col_min, row_max = world_to_pixel(x0, y0)
    col_max, row_min = world_to_pixel(x1, y1)

    draw_filled_rect(grid, col_min, row_min, col_max, row_max, OCCUPIED)


def generate_map():
    """Generate the stress test arena PGM map."""
    # Initialize grid with free space
    grid = bytearray([FREE_SPACE] * (WIDTH * HEIGHT))

    # -- Arena boundary walls (20m x 20m, from -10 to +10) --
    wall_thickness = 0.2  # meters

    # North wall: y=10, x from -10 to 10
    draw_wall_horizontal(grid, world_y=10.0,
                         world_x_min=-10.0, world_x_max=10.0,
                         thickness_m=wall_thickness)

    # South wall: y=-10, x from -10 to 10
    draw_wall_horizontal(grid, world_y=-10.0,
                         world_x_min=-10.0, world_x_max=10.0,
                         thickness_m=wall_thickness)

    # East wall: x=10, y from -10 to 10
    draw_wall_vertical(grid, world_x=10.0,
                       world_y_min=-10.0, world_y_max=10.0,
                       thickness_m=wall_thickness)

    # West wall: x=-10, y from -10 to 10
    draw_wall_vertical(grid, world_x=-10.0,
                       world_y_min=-10.0, world_y_max=10.0,
                       thickness_m=wall_thickness)

    # -- Static obstacles (filled circles) --
    obstacle_radius_m = 0.3
    obstacle_radius_px = int(obstacle_radius_m / RESOLUTION)  # 6 pixels

    # Obstacle 1: world (2, 4)
    cx1, cy1 = world_to_pixel(2.0, 4.0)
    draw_filled_circle(grid, cx1, cy1, obstacle_radius_px, OCCUPIED)

    # Obstacle 2: world (-3, -3)
    cx2, cy2 = world_to_pixel(-3.0, -3.0)
    draw_filled_circle(grid, cx2, cy2, obstacle_radius_px, OCCUPIED)

    # -- Write PGM file (P5 binary format) --
    os.makedirs(os.path.dirname(PGM_PATH), exist_ok=True)

    with open(PGM_PATH, 'wb') as f:
        # Header
        header = f"P5\n{WIDTH} {HEIGHT}\n255\n"
        f.write(header.encode('ascii'))
        # Binary pixel data
        f.write(bytes(grid))

    print(f"[OK] PGM saved: {os.path.abspath(PGM_PATH)}")
    print(f"     Size: {WIDTH}x{HEIGHT} pixels, {RESOLUTION} m/pixel")
    print(f"     Arena: 20m x 20m walls, 2 obstacles")

    # -- Write YAML file --
    yaml_content = (
        "image: stress_test_arena.pgm\n"
        "resolution: 0.05\n"
        "origin: [-25.0, -25.0, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n"
    )

    with open(YAML_PATH, 'w') as f:
        f.write(yaml_content)

    print(f"[OK] YAML saved: {os.path.abspath(YAML_PATH)}")

    # -- Verification --
    verify_map(grid)


def verify_map(grid: bytearray):
    """Print verification stats."""
    occupied_count = sum(1 for v in grid if v == OCCUPIED)
    free_count = sum(1 for v in grid if v == FREE_SPACE)
    total = WIDTH * HEIGHT

    print(f"\n--- Verification ---")
    print(f"Total pixels:    {total}")
    print(f"Occupied (0):    {occupied_count}")
    print(f"Free (254):      {free_count}")
    print(f"Other:           {total - occupied_count - free_count}")

    # Check wall corners exist
    corners = [
        ("NW corner", -10.0, 10.0),
        ("NE corner", 10.0, 10.0),
        ("SW corner", -10.0, -10.0),
        ("SE corner", 10.0, -10.0),
    ]
    for name, wx, wy in corners:
        px, py = world_to_pixel(wx, wy)
        val = grid[py * WIDTH + px]
        status = "OK" if val == OCCUPIED else "FAIL"
        print(f"  {name} ({wx},{wy}) -> pixel ({px},{py}) = {val} [{status}]")

    # Check obstacles
    obstacles = [
        ("Obstacle 1", 2.0, 4.0),
        ("Obstacle 2", -3.0, -3.0),
    ]
    for name, wx, wy in obstacles:
        px, py = world_to_pixel(wx, wy)
        val = grid[py * WIDTH + px]
        status = "OK" if val == OCCUPIED else "FAIL"
        print(f"  {name} ({wx},{wy}) -> pixel ({px},{py}) = {val} [{status}]")

    # Check center is free
    px, py = world_to_pixel(0.0, 0.0)
    val = grid[py * WIDTH + px]
    status = "OK" if val == FREE_SPACE else "FAIL"
    print(f"  Center (0,0) -> pixel ({px},{py}) = {val} [{status}]")


if __name__ == '__main__':
    generate_map()
