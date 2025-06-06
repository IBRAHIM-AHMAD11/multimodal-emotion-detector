from transformers import pipeline, AutoFeatureExtractor, AutoModelForImageClassification, Wav2Vec2ForSequenceClassification
import os

# Create folders for each model

if not os.path.exists("./models"):
    os.makedirs("./models/text")
    os.makedirs("./models/voice")
    os.makedirs("./models/face")

# Text model
text_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
text_model.model.save_pretrained("./models/text")
text_model.tokenizer.save_pretrained("./models/text")

# Voice model
voice_model_name = "superb/wav2vec2-base-superb-er"
voice_extractor = AutoFeatureExtractor.from_pretrained(voice_model_name)
voice_model = Wav2Vec2ForSequenceClassification.from_pretrained(voice_model_name)
voice_extractor.save_pretrained("./models/voice")
voice_model.save_pretrained("./models/voice")

# Face model
face_model_name = "carlosleao/FER-Facial-Expression-Recognition"
face_extractor = AutoFeatureExtractor.from_pretrained(face_model_name)
face_model = AutoModelForImageClassification.from_pretrained(face_model_name)
face_extractor.save_pretrained("./models/face")
face_model.save_pretrained("./models/face")