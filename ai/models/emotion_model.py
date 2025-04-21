import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class EmotionModel:
    def __init__(self, model_name: str = "cointegrated/rubert-tiny-sentiment-balanced"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.labels = ['negative', 'neutral', 'positive']  # можно уточнить по документации модели

    def predict_emotion(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[0][pred_idx].item()
        return {
            "emotion": self.labels[pred_idx],
            "confidence": confidence
        }
