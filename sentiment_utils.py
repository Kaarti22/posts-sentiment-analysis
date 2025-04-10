from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

emotion_labels = [
    "a photo expressing happiness",
    "a photo expressing sadness",
    "a photo expressing anger",
    "a photo expressing fear",
    "a photo expressing surprise",
    "a photo expressing neutrality"
]

def map_emotion_to_sentiment(emotion_label: str) -> str:
    if "happiness" in emotion_label:
        return "POSITIVE"
    elif any(word in emotion_label for word in ["sadness", "anger", "fear"]):
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def predict_image_sentiment_clip(image_path: str):
    image = Image.open(image_path).convert("RGB")
    
    inputs = clip_processor(text=emotion_labels, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)

    logits_per_image = outputs.logits_per_image  # shape (1, num_prompts)
    probs = logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]

    results = [{"label": label, "score": float(score)} for label, score in zip(emotion_labels, probs)]
    top_sentiment = max(results, key=lambda x: x["score"])
    
    return {
        "image_sentiment": results,
        "final_image_sentiment": top_sentiment["label"]
    }
