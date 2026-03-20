import cv2, mediapipe as mp
cap = cv2.VideoCapture(0)
mpfm = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
print("Camera opened:", cap.isOpened())
ret, frame = cap.read()
print("Frame captured:", bool(ret))
if ret:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mpfm.process(rgb)
    print("Face landmarks found:", bool(res.multi_face_landmarks))
cap.release()    