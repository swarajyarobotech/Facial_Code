import RPi.GPIO as GPIO
import time
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import cv2

SERVO_PIN = 17 # GPIO 12
BUZZER_PIN = 21
RED_LED_PIN = 17
GREEN_LED_PIN = 9
TRIG_PIN = 18
ECHO_PIN = 24
BUTTON_PIN = 23


GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)
GPIO.setup(BUTTON_PIN, GPIO.IN,pull_up_down = GPIO.PUD_UP)


# Set up PWM for the servo motor
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz frequency
pwm.start(0)  # Initialize PWM with 0 duty cycle

def set_servo_angle(angle):
    """Sets the servo to the specified angle."""
    duty = 2 + (angle / 18)  # Duty cycle calculation for the angle
    GPIO.output(SERVO_PIN, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(SERVO_PIN, False)
    pwm.ChangeDutyCycle(0)
    
    
def buzz_alarm():
    """for 8 sec"""
    GPIO.output(BUZZER_PIN,True)
    time.sleep(8)
    GPIO.output(BUZZER_PIN,False)
    
def blink_Led(pin = 17,duration = 5):
    iterations = int(duration / 0.2)
    for i in range(iterations):
        GPIO.output(pin,True)
        time.sleep(0.2)
        GPIO.output(pin,False)
        time.sleep(0.4)
    
        
    

# Initialize 'currentname' to trigger only when a new person is identified
currentname = ["Person1", "Person2", "Person3", "Person4", "Person5", "Person6", "Person7", "Person8"]
encodingsP = "encodings.pickle"

# Load the known faces and embeddings along with OpenCV's Haar cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# Initialize the video stream and allow the camera sensor to warm up
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# Start the FPS counter
fps = FPS().start()

# State variables for door and timer
door_open = False
last_face_time = time.time()

# Loop over frames from the video file stream
try:
    while True:
        # Grab the frame from the threaded video stream
        frame = vs.read()
        frame = imutils.resize(frame, width=500)

        # Detect the face boxes
        boxes = face_recognition.face_locations(frame)
        encodings = face_recognition.face_encodings(frame, boxes)
        names = []

        face_detected = False

        # Loop over the facial embeddings
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"  # Default name for unrecognized faces
            
            if name =="Unknown":
                buzz_alarm()
                blink_Led(RED_LED_PIN,duration = 5)
                print("Unknown Person Detected")
            # Check if there's a match
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # Count votes for each recognized face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # Determine the name with the highest votes
                name = max(counts, key=counts.get)

            names.append(name)

            # If the person is in the currentname list and the door is not open
            if name in currentname:
                face_detected = True

                if not door_open:
                    print(f"[INFO] {name} detected. Opening the door.")
                    set_servo_angle(90)  # Open the door
                    door_open = True
                    last_face_time = time.time()  # Reset the timer when a valid face is detected

        # Draw bounding boxes and names on the frame
        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # If no valid face is detected, start the timer
        if not face_detected:
            if door_open and time.time() - last_face_time > 3:  # 3 seconds timeout
                print("[INFO] No valid face detected. Closing the door.")
                set_servo_angle(0)  # Close the door
                door_open = False
        else:
            # If a valid face is detected, reset the timer
            last_face_time = time.time()

        # Display the frame
        cv2.imshow("Facial Recognition is Running", frame)
        key = cv2.waitKey(1) & 0xFF

        # Quit when 'q' key is pressed
        if key == ord("q"):
            break

        # Update the FPS counter
        fps.update()

except KeyboardInterrupt:
    print("\n[INFO] Exiting program.")

finally:
    fps.stop()
    print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    vs.stop()
    pwm.stop()
    GPIO.cleanup()
