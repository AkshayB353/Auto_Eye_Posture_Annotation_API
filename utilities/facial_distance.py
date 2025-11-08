import cv2
import mediapipe as mp
import numpy as np
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
LEFT_EYE_OUTER = 33   
RIGHT_EYE_OUTER = 362 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Face Width Calibrator @ 50 cm")
print("Sit 50 cm from camera. Look straight. Press 'q' when ready.")
widths = []
cv2.namedWindow("Calibration - Press Q", cv2.WINDOW_NORMAL)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    display = frame.copy()
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left = landmarks[33]
        right = landmarks[263]
        right2 = landmarks[362]
        left_px = (int(left.x * w), int(left.y * h))
        right_px = (int(right.x * w), int(right.y * h))
        right2_px = (int(right2.x * w), int(right2.y * h))
        cv2.circle(display, left_px, 6, (0, 255, 0), -1)
        cv2.circle(display, right_px, 6, (0, 255, 0), -1)
        cv2.line(display, left_px, right_px, (0, 255, 0), 2)
        cv2.line(display, left_px, right2_px, (0, 255, 255), 2)
        width_px = abs(right_px[0] - left_px[0])
        widths.append(width_px)
        cv2.putText(display, f"Width: {width_px:.1f}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "Press Q to save", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    else:
        cv2.putText(display, "Face not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Calibration - Press Q", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
if len(widths) < 10:
    print("Not enough samples. Try again.")
else:
    face_width_50cm = np.median(widths)
    print("\n" + "="*50)
    print(f" YOUR FACE WIDTH AT 50 CM: {face_width_50cm:.1f} pixels")
    print("="*50)
    print("\nUse this in your posture code:")
    print(f"FACE_WIDTH_AT_50CM = {face_width_50cm:.1f}")
    print(f"IDEAL_FACE_WIDTH_PX = {face_width_50cm:.1f}  # for 'Straight' posture")
    print("\nFor distance estimation:")
    print(f"FOCAL_LENGTH_PX = ({face_width_50cm:.1f} * 50) / 14.0  # if real width = 14 cm")
    print("="*50)