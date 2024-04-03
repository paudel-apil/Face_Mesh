import cv2
import time
import mediapipe as mp


win_name = "Face Mesh"
cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)

mpfaceMesh = mp.solutions.face_mesh
faceMesh = mpfaceMesh.FaceMesh(max_num_faces = 5)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness = 1,circle_radius = 1,color = (0,255,0))

pTime = 0


while True:
    has_frame, frame = cap.read()
    if not has_frame:
        break

    imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    # print(results.multi_face_landmarks)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame,face_landmarks,mpfaceMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)
            for id,lm in enumerate(face_landmarks.landmark):
                ih, iw, ic = frame.shape
                x,y = int(lm.x * iw), int(lm.y * ih)
                print(id,x,y)


    
    cTime = time.time()
    if cTime != pTime:
        fps = 1 / (cTime- pTime)
    else:
        fps = 0
    pTime = cTime

    cv2.putText(frame,str(int(fps)),(30,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),3)
    # frame = cv2.flip(frame,1)
    cv2.imshow(win_name,frame)

    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyWindow(win_name)

    