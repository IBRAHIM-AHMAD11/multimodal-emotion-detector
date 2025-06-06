# Multimodal Emotion Detector

This app uses facial image, voice tone, and text sentiment to predict emotions.

## Features
- CNN/ResEmoteNet for facial emotion recognition
- Wav2Vec2 for voice emotion recognition
- DistilBERT for text sentiment classification
- Flask-based web interface

## Run it:
```bash
pip install -r requirements.txt
python app.py
```
Then open: [http://localhost:5000](http://localhost:5000)