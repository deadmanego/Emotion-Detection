# Emotion-Detection
import cv2
from deepface import DeepFace
import time

# Initialize webcam
cap = cv2.VideoCapture(0)  # Change to 1 if you have multiple cameras

# Check if webcam opens correctly
if not cap.isOpened():
    print("❌ Error: Cannot access webcam")
    exit()

time.sleep(1)  # Give camera time to initialize

# Load Haar cascade once (instead of every frame)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()  # Read frame from webcam
    
    if not ret or frame is None:
        print("❌ Error: Empty frame received")
        continue  # Skip iteration instead of breaking

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]  # Extract face region

        try:
            # Use DeepFace to analyze emotion
            result = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            # Draw rectangle & label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        except Exception as e:
            print(f"⚠️ DeepFace Error: {e}")
            continue  # Skip processing for this face
    
    # Display frame
    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()