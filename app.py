from flask import Flask, request, render_template, jsonify
import os
from utils.face_infer import predict_emotion_from_face
from utils.voice_infer import predict_emotion_from_voice
from utils.text_infer import predict_emotion_from_text
from fusion import fuse_emotions

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    results = {}

    face_file = request.files.get('face_image')
    voice_file = request.files.get('voice_audio')
    text_input = request.form.get('text_input')

    if face_file:
        face_path = os.path.join(UPLOAD_FOLDER, face_file.filename)
        face_file.save(face_path)
        results['face'] = predict_emotion_from_face(face_path)

    if voice_file:
        voice_path = os.path.join(UPLOAD_FOLDER, voice_file.filename)
        voice_file.save(voice_path)
        results['voice'] = predict_emotion_from_voice(voice_path)

    if text_input:
        results['text'] = predict_emotion_from_text(text_input)

    fused = fuse_emotions(results)
    return jsonify({"individual": results, "fused_emotion": fused})

if __name__ == '__main__':
    app.run(debug=True)