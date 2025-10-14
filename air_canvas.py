# Step 1: Import the necessary libraries
import cv2          # OpenCV for all computer vision tasks
import numpy as np  # NumPy for handling arrays and numerical operations

# --- Global Variables & Setup ---

# This list will store the coordinates of the points to be drawn.
# It's a list containing other lists. Each inner list represents a single, continuous line.
# We start with one empty line.
points_to_draw = [[]]
# This index keeps track of which line we are currently adding points to.
current_line_index = 0

# Create a window with the name "Air Canvas" that will display our output.
# cv2.WINDOW_AUTOSIZE makes the window resizable.
cv2.namedWindow("Air Canvas", cv2.WINDOW_AUTOSIZE)

# Start video capture from the default webcam (which is usually at index 0).
cap = cv2.VideoCapture(0)

# --- HSV Color Tracker Setup ---
# This section defines the color range for the object we want to track.
# The default values are set for a bright blue color.
# You can change these np.array values to track a different color (e.g., green, red).
# Note: In OpenCV, Hue is in the range [0, 179], Saturation [0, 255], and Value [0, 255].
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])


# --- Main Application Loop ---
# This `while True` loop will run continuously, processing one frame from the webcam at a time.
while True:
    # 1. Read a new frame from the webcam.
    # `success` is a boolean that is True if the frame was read successfully.
    # `frame` is the actual image captured from the webcam.
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame from webcam. Exiting...")
        break # Exit the loop if we can't get a frame.

    # 2. Flip the frame horizontally.
    # This creates a mirror-like effect, which is much more intuitive for drawing.
    frame = cv2.flip(frame, 1)

    # 3. Convert the frame from BGR to HSV color space.
    # The HSV (Hue, Saturation, Value) model is better for color detection than the default BGR.
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 4. Create a "mask" for the specified color.
    # The mask is a black-and-white image. Pixels in the original frame that fall
    # within our blue color range will be white in the mask, and all other pixels will be black.
    color_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # 5. Find the contours (outlines) of the white areas in the mask.
    # cv2.RETR_EXTERNAL gets only the outermost contours.
    # cv2.CHAIN_APPROX_SIMPLE compresses contour segments, saving memory.
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Check if any contours were found.
    if contours:
        # Find the contour with the largest area. We assume this is our drawing tool.
        # This helps to ignore small patches of noise that might be the same color.
        largest_contour = max(contours, key=cv2.contourArea)
        
        # We only proceed if the detected object is of a reasonable size.
        if cv2.contourArea(largest_contour) > 500:
            # Calculate the center of the largest contour using image moments.
            moments = cv2.moments(largest_contour)
            # Avoid division by zero
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])

                # Add this new center point to the list for the current line.
                points_to_draw[current_line_index].append((center_x, center_y))

    # 7. Draw all the lines onto the frame.
    # We loop through each line in our main list of points.
    for line in points_to_draw:
        # We need at least two points to be able to draw a line segment.
        for i in range(1, len(line)):
            # Draw a blue line between the previous point (i-1) and the current point (i).
            # The last number (5) is the thickness of the line.
            cv2.line(frame, line[i - 1], line[i], (255, 0, 0), 5)

    # 8. Display helpful instructions on the screen.
    cv2.putText(frame, "Track a BLUE object to draw", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Press 'c' to clear", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 9. Show the final, modified frame in the "Air Canvas" window.
    cv2.imshow("Air Canvas", frame)

    # 10. Wait for 1 millisecond for a key press.
    # The `& 0xFF` is a bitwise AND operation, important for 64-bit machines.
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # If the user presses 'c', clear the canvas by resetting the points list.
        points_to_draw = [[]]
        current_line_index = 0
    elif key == ord('q'):
        # If the user presses 'q', break out of the while loop to end the program.
        break

# --- Cleanup ---
# This code runs after the loop has been broken.
# Release the webcam so other applications can use it.
cap.release()
# Close all the windows that OpenCV created.
cv2.destroyAllWindows()

