from transformers import pipeline

classifier = pipeline("text-classification", model="./models/text")

def predict_emotion_from_text(text):
    result = classifier(text)[0]
    return result['label']
