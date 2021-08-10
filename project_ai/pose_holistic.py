import cv2
import mediapipe as mp
import os
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# 이미지를 pose estimation
IMAGE_FILES = os.listdir('musinsa_img_data')  # 폴더안의 파일 이름을 리스트로 만들기 / listdir에서 폴더 명만 수정하면 됨
with mp_holistic.Holistic(
    static_image_mode=True) as holistic:
    for idx, file in enumerate(IMAGE_FILES):
        try:
            image = cv2.imread('musinsa_img_data/' + file)   # listdir과 똑같은 폴더 명으로 수정
            image_height, image_width, _ = image.shape
            # BGR로 불러온 이미지(cv2이기 때문)를 RGB로 변경
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                print(
                    f'Nose coordinates: ('
                    f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
                    f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
                )
            # 포즈, 왼손,오른손, 얼굴에 점 찍기
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
            mp_drawing.draw_landmarks(
                annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            cv2.imwrite('new_musinsa_img_data/' + str(idx+1) + '.png', annotated_image) # 점 찍은 사진을 새로 저장 / ' '안에 새로운 폴더명으로 바꾸기
            # # Plot pose world landmarks.
            # # mp_drawing.plot_landmarks(
            # #     results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
        except:
            pass