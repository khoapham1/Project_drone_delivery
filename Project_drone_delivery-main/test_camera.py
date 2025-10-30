import cv2
import numpy as np
from drone_control import start_camera, get_lastest_frame, stop_camera
import time

# Start the camera
print("Starting camera...")
start_camera()
print("Camera started. Waiting briefly for frame capture...")
time.sleep(2)  # Wait for camera to initialize and capture a frame

# Get the latest frame
frame = get_lastest_frame()

if frame is None:
    print("❌ No frame captured. Ensure the camera is working.")
else:
    # Decode the JPEG frame to an OpenCV image (BGR format)
    nparr = np.frombuffer(frame, np.uint8)
    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if cv_image is None:
        print("❌ Failed to decode frame.")
    else:
        # Display the frame
        cv2.imshow("Captured Frame", cv_image)
        print("Displaying frame. Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally, save the frame to a file for inspection
        output_path = "captured_frame.jpg"
        cv2.imwrite(output_path, cv_image)
        print(f"Frame saved to {output_path}")

# Stop the camera
print("Stopping camera...")
stop_camera()
print("Camera stopped.")