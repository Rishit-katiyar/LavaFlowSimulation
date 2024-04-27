# lava_flow_simulation.py

# Import necessary libraries
from heapq import heappush, heappop, heapify
import sys
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# Constants for planetary surface
planet_average_surface_depth_km = 1  # Average surface depth of the planet in kilometers

# Calculate magma level based on average surface depth
magma_level = -planet_average_surface_depth_km

# Input and output file paths
input_image_path = "tolbachik_ali_2013045_geo.tif"  # Input image file path for Venus
output_image_path = "lava_flows.tif"  # Output image file path for Venus

# Optional parameters for volcanic flow mapping
random_seed = None  # Random seed for reproducibility
contrast = 500  # Contrast parameter for volcanic flow intensity generation
bit_depth = 16  # Bit depth of the output image
flow_width_factor = 100  # Factor to control volcanic flow width (increased significantly for super thick flows)
flow_detection_limit = 0  # Limit for volcanic flow detection

# Set random seed if provided
if random_seed:
    np.random.seed(seed=random_seed)

# Set recursion limit to avoid stack overflow
sys.setrecursionlimit(65536)

# Read the input image for the planet surface
print("Reading the planetary surface image...")
planet_surface_image = imageio.imread(input_image_path)

# Convert to grayscale if image is color
if len(planet_surface_image.shape) == 3 and planet_surface_image.shape[2] == 3:
    planet_surface_image = np.dot(planet_surface_image[..., :3], [0.2989, 0.5870, 0.1140])

# Get the heightmap from the input image for the planet surface
heightmap = np.array(planet_surface_image)
(X, Y) = heightmap.shape

# Print information about the input image for the planet surface
print("Dimensions of the input planetary surface image:", X, "x", Y)
print("Input image loaded successfully")

# Find eruption points on the planet
print("Identifying eruption points on the planet...")

# Initialize visited array to keep track of visited pixels on the planet
visited = np.zeros((X, Y), dtype=bool)

# List to store eruption points on the planet
eruption_points = []

# Function to add an eruption point to the list on the planet
def add_eruption_point(x, y):
    eruption_points.append((heightmap[x, y] + np.random.random(), x, y))
    visited[x, y] = True

# Counter for number of points to explore on the planet
to_explore = 0

# Iterate through each pixel to find eruption points on the planet
for x in range(1, X - 1):
    for y in range(1, Y - 1):
        # Check if the pixel is below magma level on the planet
        if heightmap[x, y] <= magma_level:
            continue
        to_explore += 1
        if to_explore % 1000000 == 0:
            print("Found", str(to_explore // 1000000), "millions points to explore")
        # Check if the pixel has a neighboring pixel below magma level on the planet
        if (heightmap[x - 1, y] <= magma_level or heightmap[x + 1, y] <= magma_level or
                heightmap[x, y - 1] <= magma_level or heightmap[x, y + 1] <= magma_level):
            add_eruption_point(x, y)

# Check pixels on the edges of the image on the planet
for x in range(X):
    if heightmap[x, 0] > magma_level:
        add_eruption_point(x, 0)
        to_explore += 1
    if heightmap[x, -1] > magma_level:
        add_eruption_point(x, Y - 1)
        to_explore += 1

for y in range(1, Y - 1):
    if heightmap[0, y] > magma_level:
        add_eruption_point(0, y)
        to_explore += 1
    if heightmap[-1, y] > magma_level:
        add_eruption_point(X - 1, y)
        to_explore += 1

# Print information about the eruption points on the planet
print("Found", str(len(eruption_points)), "eruption points on the planet")

# Create a heap from the eruption points list on the planet
heap = eruption_points[:]
heapify(heap)

# Print information about volcanic flow path construction on the planet
print("Constructing volcanic flow paths on the planet:", str(to_explore), "points to visit")

# Array to store flow directions on the planet
flow_directions = np.zeros((X, Y), dtype=np.int8)

# Function to try pushing a pixel to the heap on the planet
def try_push(x, y):
    if not visited[x, y]:
        h = heightmap[x, y]
        if h > magma_level:
            heappush(heap, (h + np.random.random(), x, y))
            visited[x, y] = True
            return True
    return False

# Function to process neighboring pixels and simulate volcanic flow on the planet
def process_neighbors(x, y):
    dirs = 0
    if x > 0 and try_push(x - 1, y):
        dirs += 1
    if y > 0 and try_push(x, y - 1):
        dirs += 2
    if x < X - 1 and try_push(x + 1, y):
        dirs += 4
    if y < Y - 1 and try_push(x, y + 1):
        dirs += 8
    # Simulate volcanic flow direction based on neighboring pixels on the planet
    flow_directions[x, y] = dirs

# Iterate until the heap is empty on the planet
while len(heap) > 0:
    t = heappop(heap)
    to_explore -= 1
    if to_explore % 1000000 == 0:
        print(str(to_explore // 1000000), "million points left", "Altitude:", int(t[0]), "Queue:", len(heap))
    process_neighbors(t[1], t[2])

# Cleanup visited and heightmap arrays on the planet
visited = None
heightmap = None

# Print information about volcanic flow intensity calculation on the planet
print("Calculating volcanic flow paths on the planet")

# Array to store volcanic flow intensity on the planet
volcanic_flow_intensity = np.ones((X, Y))

# Function to recursively set volcanic flow intensity for each pixel on the planet
def set_intensity(x, y):
    intensity = 1
    dirs = flow_directions[x, y]

    if dirs % 2 == 1:
        intensity += set_intensity(x - 1, y)
    dirs //= 2
    if dirs % 2 == 1:
        intensity += set_intensity(x, y - 1)
    dirs //= 2
    if dirs % 2 == 1:
        intensity += set_intensity(x + 1, y)
    dirs //= 2
    if dirs % 2 == 1:
        intensity += set_intensity(x, y + 1)
    volcanic_flow_intensity[x, y] = intensity
    return intensity

# Find the maximal volcanic flow intensity on the planet
max_intensity = 0
for start in eruption_points:
    intensity = set_intensity(start[1], start[2])
    if intensity > max_intensity:
        max_intensity = intensity

# Print information about maximal volcanic flow intensity on the planet
print("Maximal volcanic flow intensity:", str(max_intensity))

# Cleanup flow_directions array on the planet
flow_directions = None

# Print information about image generation on the planet
print("Generating volcanic flow intensity map")

# Calculate power for volcanic flow intensity transformation on the planet
power = 1 / contrast

# Generate volcanic flow intensity map based on the specified parameters on the planet
if bit_depth <= 8:
    bit_depth = 8
    dtype = np.uint8
elif bit_depth <= 16:
    bit_depth = 16
    dtype = np.uint16
elif bit_depth <= 32:
    bit_depth = 32
    dtype = np.uint32
else:
    bit_depth = 64
    dtype = np.uint64

max_value = 2 ** bit_depth - 1
if max_intensity != 0:
    coeff = max_value / (max_intensity ** power)
else:
    coeff = 0

# Calculate volcanic flow width based on volcanic flow intensity on the planet
flow_width = np.floor((volcanic_flow_intensity ** power) * (coeff * flow_width_factor)).astype(dtype)

# Cleanup volcanic_flow_intensity array on the planet
volcanic_flow_intensity = None

# Save the generated image to the output file on the planet
imageio.imwrite(output_image_path, flow_width.astype(np.uint16))  # Ensure correct data type for saving

# Print information about the output image on the planet
print("Output image saved to:", output_image_path)

# Display the input and output images on the planet
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Input image on the planet
axes[0].imshow(planet_surface_image, cmap='gray')
axes[0].set_title('Input Image (Planet Surface)')
axes[0].axis('off')

# Output image on the planet
axes[1].imshow(flow_width, cmap='hot')  # Use 'hot' colormap for intensity
axes[1].set_title('Generated Volcanic Flow Intensity Map')
axes[1].axis('off')

# Adjust layout to fit images on the planet
plt.tight_layout()

# Show the images on the planet
plt.show()
