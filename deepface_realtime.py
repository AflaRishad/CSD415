from deepface import DeepFace
import cv2
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # face_recognition/
FACES_DIR = os.path.join(BASE_DIR, "faces")
TEMP_PATH = os.path.join(BASE_DIR, "temp.jpg")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save frame temporarily
    cv2.imwrite(TEMP_PATH, frame)

    # Default values
    name = "Unknown"
    face_area = None

    try:
        # Detect face
        faces = DeepFace.extract_faces(
            img_path=TEMP_PATH,
            enforce_detection=False,
            detector_backend='opencv'
        )

        if faces:
            face_area = faces[0]['facial_area']
            x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']

            # Try to find in database
            result = DeepFace.find(
                img_path=TEMP_PATH,
                db_path=FACES_DIR,
                enforce_detection=False,
                detector_backend='opencv'
            )

            if len(result) > 0 and len(result[0]) > 0:
                identity_path = result[0].iloc[0]['identity']
                name = os.path.basename(os.path.dirname(identity_path))

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw name *below* the box
            cv2.putText(frame, name, (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    except:
        pass

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Delete temp image
if os.path.exists(TEMP_PATH):
    os.remove(TEMP_PATH)
