from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from PIL import Image
import torchvision.transforms as T

feature_extractor = AutoFeatureExtractor.from_pretrained("./models/face")
model = AutoModelForImageClassification.from_pretrained("./models/face")
model.eval()

emotion_labels = model.config.id2label

transform = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

def predict_emotion_from_face(image_path):
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    return emotion_labels[pred]
