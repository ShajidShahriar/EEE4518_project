import cv2
import mediapipe as mp
import time
import math
import serial

# Serial Communication Setup (Connect to Arduino)
serial_connected = False
desired_port = 'COM9'  # Change this to match your Arduino's COM port

try:
    arduino = serial.Serial(desired_port, 9600, timeout=1)
    time.sleep(2)  # Allow time for Arduino to initialize
    serial_connected = True
    print(f"Connected to Arduino on {desired_port}")
except Exception as e:
    print(f"Serial connection failed: {e}")
    serial_connected = False


# Functions
def make_square(image):
    h, w = image.shape[:2]
    size = max(h, w)
    padded_image = cv2.copyMakeBorder(
        image,
        (size - h) // 2,
        (size - h) - (size - h) // 2,
        (size - w) // 2,
        (size - w) - (size - w) // 2,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    return padded_image


def map_to_servo_angle(x, width):
    """ Maps the waist position to a servo angle between 60 and 120 degrees. """
    angle = 60 + int((x / width) * 60)
    return max(60, min(120, angle))  # Clamp between 60-120 degrees


def calculate_servo_point(base_x, base_y, angle, length=200):
    """ Calculate the end point for the servo visualization arrow. """
    rad = math.radians(angle)
    end_x = int(base_x + length * math.cos(rad))
    end_y = int(base_y - length * math.sin(rad))
    return end_x, end_y


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Video Capture
cap = cv2.VideoCapture(0)  # Change index if needed

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Set resolution and FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# Variables
prev_time = time.time()
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP
RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP
current_servo_angle = 90  # Start at center

# Main Loop
try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Can't receive frame.")
            break

        frame = make_square(frame)
        h, w = frame.shape[:2]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Get Waist Position
            left_hip = results.pose_landmarks.landmark[LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[RIGHT_HIP]
            waist_x = int((left_hip.x + right_hip.x) / 2 * w)
            waist_y = int((left_hip.y + right_hip.y) / 2 * h)

            # Calculate Servo Angle Directly Based on Waist Position
            current_servo_angle = map_to_servo_angle(waist_x, w)
            print(f"Tracking Waist: {waist_x}px -> Servo: {current_servo_angle}°")

            # Send Servo Command to Arduino
            if serial_connected:
                arduino.write(f"{current_servo_angle}\n".encode())

            # Draw Indicators
            cv2.putText(image, f"Servo Angle: {current_servo_angle}°", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.circle(image, (waist_x, waist_y), 5, (0, 0, 255), -1)

            # Servo Arrow Visualization
            end_x, end_y = calculate_servo_point(w // 2, h - 50, current_servo_angle)
            cv2.arrowedLine(image, (w // 2, h - 50), (end_x, end_y), (0, 255, 0), 5)
            cv2.circle(image, (w // 2, h - 50), 10, (255, 0, 0), -1)
        else:
            # No Player Detected -> Reset to Default Position
            current_servo_angle = 90
            if serial_connected:
                arduino.write(f"{current_servo_angle}\n".encode())
            cv2.putText(image, "No player detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show Image
        cv2.imshow('Ping Pong Launcher - Player Tracking', image)

        # Exit Condition
        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    if serial_connected:
        arduino.close()
        print("Serial connection closed.")
