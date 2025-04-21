import torch
from transformers import BertForSequenceClassification, BertTokenizer
import pickle
from typing import Dict

class IntentClassifier:
    def __init__(self, model_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BertForSequenceClassification.from_pretrained(f"{model_dir}/model").to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(f"{model_dir}/model")
        self.model.eval()

        with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

    def predict(self, text: str) -> Dict:
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs).item()

        return {
            "intent": self.label_encoder.inverse_transform([pred_idx])[0],
            "confidence": probs[0][pred_idx].item()
        }
