from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import pickle

# 학습모델 불러오기
with open('pose.pkl', 'rb') as f:
    model = pickle.load(f)


mp_darwing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

app = Flask(__name__)

ret = cap.set(3,1280) # 캠 가로 크기
ret = cap.set(4,720) # 캠 세로 크기


# 프레임에서 사람 인식하여 다시 표시
def gen_frames():
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
        while cap.isOpened():
            ret , image = cap.read() #프레임을 제대로 읽으면 ret = true 아니면 false / 읽은 프레임은 image
            if not ret:
                print('camera failed!!!')
                continue

            original_image = image.copy()

            back = np.full((720, 1280, 3), 0, np.uint8) # 겹칠 이미지 배경

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            mp_darwing.draw_landmarks(image, results.pose_landmarks,mp_pose.POSE_CONNECTIONS)

            try:
                X = [list(np.array([[landmark.x , landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten())]
                model_pose_class = model.predict(X)[0] # 클래스 예측
                model_pose_prob = model.predict_proba(X)[0] # 유사도 측정
                prob = round(model_pose_prob[np.argmax(model_pose_prob)] * 100)

                # 유사도 및 클래스 화면에 표시(나중에는 없애야하는 것)
                cv2.rectangle(image, (0,0), (200,60), (245, 117, 16), -1)

                # putText(캠, 글자, 시작좌표, 폰트, 크기, 색깔,굵기 , LINE_AA(곡선에서 좋은 선) )
                image = cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                image = cv2.putText(image, str(prob), (10,40), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,255,255), 1, cv2.LINE_AA)

                image = cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                image =cv2.putText(image, model_pose_class, (95,40), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,255,255), 1, cv2.LINE_AA)
                
                landmarks = results.pose_landmarks.landmark    
        
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                center_hip= [round((left_hip[0] + right_hip[0])/2*1280), round((left_hip[1] + right_hip[1])/2*720)]

                # 추천 이미지 겹치기
                pic=cv2.imread(f'results/{model_pose_class}.png', cv2.IMREAD_COLOR)
                pic = cv2.flip(pic, 1)
                width = pic.shape[1]
                height = pic.shape[0]
                r_width = round(width*1.8) # 겹칠 이미지 너비
                r_height = round(height*1.8) # 겹칠 이미지 높이
                pic2 = cv2.resize(pic, (r_width, r_height))

                # 겹칠 이미지 시작 좌표
                y = round(center_hip[0] - r_width/2 - 10) # 가로
                x = round(center_hip[1] - r_height/2) # 세로
                back[x:x+r_height, y:y+r_width] = pic2
                image = cv2.addWeighted(image, 0.8, back, 0.2, 0)


            except:
                pass

            # 웹캠의 프레임을 jpg로 인코딩 후 tobytes로 변환해서 yield로 해야 웹에 프레임이 띄워진다.
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/menu')
def menu():
    return render_template('menu.html')

@app.route('/service')
def services():
    return render_template('service.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/image_feed')
# def image_feed():
#     global model_pose_class
    # if model_pose_class == 'up':
    #     return render_template('index.html', image_file = "results/up.png")

    # elif model_pose_class == 'middle':
    #     return render_template('index.html', image_file = "results/middle.png")

    # else:
    #     return render_template('index.html', image_file = "results/down.png")

    # if model_pose_class == 'up':
    #     return send_file("results/up.png", mimetype='image/png')

    # elif model_pose_class == 'middle':
    #     return send_file("results/middle.png", mimetype='image/png')

    # else:
    #     return send_file("results/down.png", mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True) # debug=True 이면 웹페이지만 새로고침하면 변경된 사항을 볼 수 있다.(서버가 계속 실행되는 동안)


"""
해야할 것

1. 추천 사진 크기 조절 / 사진 저장
2. html 슬라이드 메뉴 / 앵커 태그

3. index를 꾸미기
4. yield 부분 이해하기 / return과 차이 
5. 발표자료 (ppt, 판넬) 수정


"""

"""
가로 1280
세로 720

center_hip[0] x축 가로
center_hip [1] y축 세로

r_width 이미지 너비 (가로)
r_height 이미지 높이 (세로)

y = 가로
x = 세로

y = round(center_hip[0] - r_width/2)
x = round(center_hip[1] - r_height/2)

"""
