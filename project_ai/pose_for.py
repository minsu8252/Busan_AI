import pickle
import cv2
import os
import mediapipe as mp
import numpy as np
import pandas as pd

with open('pose.pkl', 'rb') as f:
    model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0) # 0번은 기본 내장 카메라 / 외부카메라에 따라 1,2,3으로 부여
# print(cap.get(3), cap.get(4)) # 카메라 가로, 세로 크기 출력

ret = cap.set(3,1920) #가로 크기 조절
ret = cap.set(4,1080) # 세로 크기 조절

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
        
        try:
            # 학습된 모델에 실시간 좌표 값들을 넣고 확인
            X = pd.DataFrame([list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten())])
            model_pose_class = model.predict(X)[0]
            model_pose_prob = model.predict_proba(X)[0]
                        
            # 웹캠 좌측 상단 직사각형 박스 생성
            cv2.rectangle(image, (0,0), (200,60), (245, 117, 16), -1)  # rectangle(캠, 좌측상당좌표, 우측하단좌표, 색깔(BRG), 색채우기(-1))
            
            # 유사도(PROB) 표시
            cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(model_pose_prob[np.argmax(model_pose_prob)],3)), (10,40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255,255,255), 1, cv2.LINE_AA)
            
            # CLASS 표시
            cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX,  #putText(캠, 클자, 시작좌표, 폰트, 크기, 색깔,굵기 , LINE_AA(곡선에서 좋은 선) )
                        0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, model_pose_class.split(' ')[0], (95,40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255,255,255), 1, cv2.LINE_AA)
            picture=cv2.imread(f'results/{model_pose_class}.png', cv2.IMREAD_COLOR)
            cv2.imshow('picture', picture)
            
        except:
            pass

        cv2.imshow('MediaPipe Pose', image)         
        
        # 캠 종료
        if cv2.waitKey(5) & 0xFF == 27: # ESC키
            break
            
cap.release()
cv2.destroyAllWindows()