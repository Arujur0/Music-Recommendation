from flask import Flask, Response, render_template, request, redirect, url_for
import cv2
from models import Emotional
import torch
from recommendation import recommend_song_by_mood
from torchvision import transforms as T
import pandas as pd
import json

app = Flask(__name__)
camera = cv2.VideoCapture(0)
emote = {0 : "Angry", 1 : "Disgusted", 2 : "Afraid", 3 : "Happy", 4 : "Sad", 5 : "Surprised", 6 : "Neutral"}
features = ['danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Emotional(1, 7).to(device)
model.load_state_dict(torch.load('checkpoints\emotionnet_14.onnx'))
mood = None
song = None
song_id = None
df = pd.read_csv('readthisinsteadofscrapping.csv')
df = df.drop(columns=['Unnamed: 0'])
dfa = df.drop(columns=['artist', 'album', 'track_name', 'track_id', 'key', 'mode', 'duration_ms', 'time_signature'])
model.eval()


def gen_frames():
    global mood
    global song
    global song_id
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        else:
            detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
            for (fx, fy, fw, fh) in face:
                # Crop the face and run it through the emotion detection model
                faces = gray[fx : fy + fh, fx : fx + fw]
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
                crop = T.Compose([T.ToTensor(), T.Resize((48, 48))])
                cropped_image = crop(faces).to(device)
                cropped_image = torch.unsqueeze(cropped_image, dim=0)
                pred = model(cropped_image)
                class_label = int(torch.argmax(pred))
                mood = class_label
                song, song_id = recommend_song_by_mood(df, dfa, mood, features)
                cv2.putText(frame, emote[class_label],  (fx+20, fy-60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_4)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')        


@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    # Get the user's confirmation on the predicted emotion
    global song
    global song_id
    confirmation = request.form.get('confirmation')
    moods = request.form.get('moods')
    if confirmation == 'y':
        song, song_id = recommend_song_by_mood(df, dfa, moods, features)

        link = "https://open.spotify.com/track/" + song['track_id']
        df.drop(index=song_id, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return link
    else:
        return render_template('index.html')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='localhost', debug=True, port=8000)
        