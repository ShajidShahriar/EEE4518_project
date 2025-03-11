import cv2
import mediapipe as mp
import time
import serial
import serial.tools.list_ports
import random
import math

# Config
USE_MOCK_SERIAL = False
RETURN_TO_CENTER_AFTER_AIM = False

# COM port setup
def list_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

arduino = None
serial_connected = False
desired_port = 'COM5'  # Change this to your Arduino's COM port
available_ports = list_ports()
print(f"Available COM ports: {available_ports}")

if desired_port in available_ports:
    try:
        arduino = serial.Serial(desired_port, 9600, timeout=1)
        time.sleep(2)
        serial_connected = True
        print(f"Connected to Arduino on {desired_port}.")
    except:
        print(f"Couldn't connect to Arduino on {desired_port}")
        exit()
else:
    print(f"{desired_port} not found.")
    exit()

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
    angle = int((x / width) * 180)
    angle = max(0, min(180, angle))  # Clamp the angle between 0 and 180
    return angle

def calculate_target_x(waist_x, width, scale=1.2, rand=100):
    center_x = width / 2
    delta_x = waist_x - center_x
    target_x = waist_x + scale * delta_x
    target_x += random.randint(-rand, rand)
    target_x = max(0, min(target_x, width))
    return int(target_x)

def calculate_servo_angle_with_randomness(target_x, width):
    base_angle = map_to_servo_angle(target_x, width)
    random_offset = random.randint(-20, 20)
    servo_angle = base_angle + random_offset
    servo_angle = max(0, min(180, servo_angle))
    return servo_angle

def calculate_servo_point(base_x, base_y, angle, length=200):
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

# Video capture
cap = cv2.VideoCapture(0)  # Change index if needed

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Set resolution and FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Variables
prev_time = time.time()
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP
RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP
current_servo_angle = 90

if serial_connected and arduino and arduino.is_open:
    try:
        arduino.write(f"{current_servo_angle}\n".encode())
        print(f"Servo Angle Sent: {current_servo_angle}°")
    except:
        serial_connected = False

stability_buffer = []
stability_threshold = 30
cooldown_counter = 0

# Main loop
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
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            left_hip = results.pose_landmarks.landmark[LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[RIGHT_HIP]
            waist_x = int((left_hip.x + right_hip.x) / 2 * w)
            waist_y = int((left_hip.y + right_hip.y) / 2 * h)
            stability_buffer.append(waist_x)
            if len(stability_buffer) > 5:
                stability_buffer.pop(0)
            if cooldown_counter == 0 and len(stability_buffer) == 5:
                min_x = min(stability_buffer)
                max_x = max(stability_buffer)
                if (max_x - min_x) <= stability_threshold:
                    random_delay = random.uniform(1, 3)
                    print(f"Waiting for {random_delay:.2f} seconds")
                    delay_start = time.time()
                    while (time.time() - delay_start) < random_delay:
                        time.sleep(0.01)
                    target_x = calculate_target_x(waist_x, w)
                    servo_angle = calculate_servo_angle_with_randomness(target_x, w)
                    print(f"Servo Angle Calculated: {servo_angle}°")
                    current_servo_angle = servo_angle
                    if serial_connected and arduino and arduino.is_open:
                        try:
                            arduino.write(f"{current_servo_angle}\n".encode())
                            print(f"Servo Angle Sent: {current_servo_angle}°")
                            response = arduino.readline().decode().strip()
                            if response:
                                print(f"Arduino: {response}")
                        except:
                            serial_connected = False
                    cooldown_counter = random.randint(15, 30)
                    print(f"Cooldown: {cooldown_counter} frames")
            else:
                cooldown_counter = max(0, cooldown_counter - 1)
            cv2.putText(image, f"Servo Angle: {current_servo_angle}°", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.circle(image, (waist_x, waist_y), 5, (0, 0, 255), -1)
            target_x_visual = calculate_target_x(waist_x, w)
            cv2.circle(image, (target_x_visual, waist_y), 5, (0, 255, 255), -1)
            end_x, end_y = calculate_servo_point(w // 2, h - 50, current_servo_angle)
            cv2.arrowedLine(image, (w // 2, h - 50), (end_x, end_y), (0, 255, 0), 5)
            cv2.circle(image, (w // 2, h - 50), 10, (255, 0, 0), -1)
        else:
            cv2.putText(image, "No player detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            current_servo_angle = 90
            if serial_connected and arduino and arduino.is_open:
                try:
                    arduino.write(f"{current_servo_angle}\n".encode())
                    print(f"Servo Angle Sent: {current_servo_angle}°")
                except:
                    serial_connected = False

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(image, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(image, f"Resolution: {w}x{h}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow('Servo Aiming System', image)
        if cv2.waitKey(1) == ord('q'):
            print("Exiting...")
            break

except:
    print("An error occurred.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    if arduino and arduino.is_open:
        arduino.close()
        print("Serial connection closed.")
