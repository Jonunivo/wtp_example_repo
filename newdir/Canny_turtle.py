import cv2
import numpy as np
from gturtle import *
import sys
import os # Imported to check if file exists before overwriting

# --- CONFIGURATION ---
# Input file
IMAGE_PATH = "dog.jpg"

# Output file for the intermediate Canny result
OUTPUT_EDGE_IMAGE = "debug_canny_edges.png"

# Drawing settings
SIMPLIFICATION_FACTOR = 0.005  # Higher (e.g., 0.02) = simpler lines, faster drawing
SCALE_FACTOR = 1.0             # Adjust to fit the drawing on screen
DRAW_SPEED = -1                # -1 is instant, 1-10 is animated

# --- 1. COMPUTER VISION PART ---

print(f"Attempting to load: {IMAGE_PATH}")

# Load image in grayscale mode (needed for Canny)
img = cv2.imread(IMAGE_PATH, 0)

# --- ERROR HANDLING ---
if img is None:
    print("\n*** ERROR: Could not find or open image ***")
    print(f"Checked location: {os.path.abspath(IMAGE_PATH)}")
    print("Please ensure the filename is correct and is in the same folder as this script.")
    sys.exit()
print("Image loaded successfully.")

# --- EDGE DETECTION ---
# Use Canny to find edges.
# Threshold 1 (100): Below this, pixels are rejected.
# Threshold 2 (200): Above this, pixels are strong edges.
# Pixels between are included only if connected to a strong edge.
print("Performing Canny edge detection...")
edges = cv2.Canny(img, 100, 200)

# --- NEW FUNCTIONALITY: SAVE EDGE RESULT ---
# 'edges' is just a NumPy array image where 0 is black and 255 is white.
# We can save it directly to disk.
success = cv2.imwrite(OUTPUT_EDGE_IMAGE, edges)
if success:
    print(f"SUCCESS: Intermediate edge detection image saved as '{OUTPUT_EDGE_IMAGE}'")
else:
    print(f"WARNING: Failed to save intermediate image to '{OUTPUT_EDGE_IMAGE}'")


# --- CONTOUR FINDING ---
# This turns pixels into lists of connected points (vectors/strokes)
print("Vectorizing lines (finding contours)...")
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Get dimensions for centering the turtle drawing later
height, width = edges.shape
center_x = width // 2
center_y = height // 2

print(f"Found {len(contours)} distinct lines. Starting Turtle drawing...")

# --- 2. TURTLE DRAWING PART ---

makeTurtle()
hideTurtle()
speed(DRAW_SPEED)
clear()
setPenWidth(2) # Slightly thicker pen looks better for sketches
setPenColor("black")

# Loop through every continuous line (contour) found
for contour in contours:
    # Simplify the path to reduce points (makes it faster and smoother)
    # epsilon determines the maximum distance between the original curve and its approximation.
    epsilon = SIMPLIFICATION_FACTOR * cv2.arcLength(contour, False)
    approx = cv2.approxPolyDP(contour, epsilon, False)
    
    # Lift pen before moving to the start of a new line segment
    penUp()
    
    first_point_of_line = True
    
    for point in approx:
        # OpenCV coordinates data structure is [[[x, y]]]
        x, y = point[0]
        
        # --- Coordinate Transformation ---
        # 1. Shift origin from top-left (0,0) to center image (x - center_x)
        # 2. Flip Y axis because Turtle y-up is positive, Image y-down is positive
        # 3. Apply scaling
        tf_x = (x - center_x) * SCALE_FACTOR
        tf_y = (center_y - y) * SCALE_FACTOR 
        
        moveTo(tf_x, tf_y)
        
        if first_point_of_line:
             # We have moved to the start of the line, now put pen down
            penDown()
            first_point_of_line = False

# Move turtle out of the way when done
penUp()
setPos(-width/2 + 20, -height/2 + 20)
label("Drawing Complete")
print("Turtle drawing finished.")