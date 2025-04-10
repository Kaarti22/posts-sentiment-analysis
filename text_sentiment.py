from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_text_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
    
    max_idx = int(np.argmax(probs))
    return {
        "label": labels[max_idx],
        "score": float(probs[max_idx])
    }
