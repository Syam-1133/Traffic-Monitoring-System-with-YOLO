import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # YOLOv8 nano (lightweight)

# Open video file or webcam
video_path = "traffic_video.mp4"  # Change to 0 for webcam
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))  # Frames per second

# Output video file
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Tracking vehicle IDs
vehicle_count = 0
tracked_vehicles = {}

# Vehicle counters
vehicles_entered = 0
vehicles_exited = 0

# Define ROI (Region of Interest) for speed estimation
roi_y1 = frame_height // 3  # Start point for speed measurement
roi_y2 = (frame_height // 3) * 2  # End point for speed measurement
known_distance_meters = 10  # Adjust this based on your video

# Define pixel-to-meter ratio (tune this based on your video setup)
pixels_per_meter = 10  # For example, 10 pixels = 1 meter

# Function to smooth out speed readings
def smooth_speed(new_speed, prev_speed, alpha=0.2):
    return alpha * new_speed + (1 - alpha) * prev_speed

# Function to overlay text with a realistic background
def overlay_text(frame, text, position, font_scale, color, thickness, bg_color=(0, 0, 0), alpha=0.6):
    """Overlay text on the frame with a transparent background for better visibility"""
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x, text_y = position

    # Create a background rectangle for the text
    overlay = frame.copy()
    cv2.rectangle(overlay, (text_x - 5, text_y - text_size[1] - 5),
                  (text_x + text_size[0] + 5, text_y + 5), bg_color, -1)

    # Blend the overlay with the frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Add the text on top of the overlay
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# Initialize variables for FPS calculation
start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Increment frame count for FPS calculation
    frame_count += 1

    # Run YOLOv8 detection
    results = model(frame)

    # Count total vehicles detected in the current frame
    total_vehicles = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID

            # Detect vehicles (car, truck, motorcycle, bus)
            if cls in [2, 3, 5, 7]:
                total_vehicles += 1  # Increment total vehicle count

                # Assign an ID to each vehicle
                vehicle_id = f"V{vehicle_count}"

                # If vehicle crosses the first ROI, start timer and count it as entered
                if y1 > roi_y1 and vehicle_id not in tracked_vehicles:
                    tracked_vehicles[vehicle_id] = {
                        "start_time": time.time(),
                        "prev_speed": 0  # Initialize previous speed
                    }
                    vehicles_entered += 1  # Count vehicle as entered

                # If vehicle crosses the second ROI, calculate speed
                if y1 > roi_y2 and vehicle_id in tracked_vehicles:
                    time_taken = time.time() - tracked_vehicles[vehicle_id]["start_time"]

                    # Prevent division by zero
                    if time_taken > 0:
                        # Calculate speed in meters per second
                        speed_mps = known_distance_meters / time_taken  # meters per second

                        # Convert m/s to km/h
                        speed_kmph = speed_mps * 3.6
                        # Smooth speed to make it more realistic
                        smoothed_speed = smooth_speed(speed_kmph, tracked_vehicles[vehicle_id]["prev_speed"])

                        # Update previous speed
                        tracked_vehicles[vehicle_id]["prev_speed"] = smoothed_speed

                        # Display vehicle info on frame
                        overlay_text(frame, f"Speed: {int(smoothed_speed)} km/h", (x1, y2 + 20), 0.7, (0, 0, 255), 2)
                        overlay_text(frame, f"Class: {cls}", (x1, y2 + 40), 0.7, (0, 0, 255), 2)
                    else:
                        overlay_text(frame, "Speed: Too Fast", (x1, y2 + 20), 0.7, (0, 0, 255), 2)

                    # Remove vehicle from tracked vehicles after calculation
                    del tracked_vehicles[vehicle_id]  # After calculating speed, remove it

                    # Update vehicle exit count
                    vehicles_exited += 1

                # Draw bounding box and vehicle ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                overlay_text(frame, vehicle_id, (x1, y1 - 10), 0.5, (0, 255, 0), 2)

    # Draw ROI lines
    cv2.line(frame, (0, roi_y1), (frame_width, roi_y1), (255, 0, 0), 2)
    cv2.line(frame, (0, roi_y2), (frame_width, roi_y2), (255, 0, 0), 2)

    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps_current = frame_count / elapsed_time

    # Overlay additional information
    overlay_text(frame, f"FPS: {fps_current:.2f}", (10, 30), 0.8, (255, 255, 255), 2, bg_color=(0, 0, 0))  # Top-left
    overlay_text(frame, f"Total Vehicles: {total_vehicles}", (10, 70), 0.8, (255, 255, 255), 2, bg_color=(0, 0, 0))  # Top-left
    overlay_text(frame, f"Timestamp: {time.strftime('%H:%M:%S')}", (10, 110), 0.8, (255, 255, 255), 2, bg_color=(0, 0, 0))  # Top-left

    # Overlay the vehicle counts
    overlay_text(frame, f"Vehicles Entered: {vehicles_entered}", (frame_width - 300, 30), 0.8, (0, 255, 0), 2, bg_color=(0, 0, 0))  # Top-right
    overlay_text(frame, f"Vehicles Exited: {vehicles_exited}", (frame_width - 300, 70), 0.8, (0, 0, 255), 2, bg_color=(0, 0, 0))  # Top-right

    # Show frame with information
    cv2.imshow("Traffic Monitoring", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()