import cv2

# Connect to the Thorlabs camera
camera = cv2.VideoCapture(1)  # Use the appropriate camera index (0, 1, 2, etc.)

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()