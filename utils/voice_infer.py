import torch
import librosa
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

feature_extractor = AutoFeatureExtractor.from_pretrained("./models/voice")
model = Wav2Vec2ForSequenceClassification.from_pretrained("./models/voice")
model.eval()

def predict_emotion_from_voice(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    emotion = emotion_labels[predicted_id] if predicted_id < len(emotion_labels) else "unknown"
    return emotion
