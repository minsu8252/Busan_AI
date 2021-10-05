from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import pickle

mp_darwing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

with open('pose.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
        while cap.isOpened():
            ret , image = cap.read() #프레임을 제대로 읽으면 ret = true 아니면 false / 읽은 프레임은 image
            if not ret:
                print('camera failed!!!')
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            mp_darwing.draw_landmarks(image, results.pose_landmarks,mp_pose.POSE_CONNECTIONS)

            try:
                X = [list(np.array([[landmark.x , landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten())]
                model_pose_class = model.predict(X)[0]
                model_pose_prob = model.predict(X)[0]
                model_pose_prob_data = int(model_pose_prob[np.argmax(model_pose_prob)] * 100)
                
            except:
                pass

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()
        cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)

