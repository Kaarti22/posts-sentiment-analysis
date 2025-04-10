from sentiment_utils import predict_image_sentiment_clip, map_emotion_to_sentiment
from text_sentiment import get_text_sentiment
from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os

app = FastAPI()

@app.post("/analyze/")
async def analyze_post(text: str = Form(...), image: UploadFile = File(...)):
    temp_image_path = f"temp_{image.filename}"
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    text_result = get_text_sentiment(text)
    image_result = predict_image_sentiment_clip(temp_image_path)
    image_sentiment = map_emotion_to_sentiment(image_result["final_image_sentiment"])

    os.remove(temp_image_path)

    if text_result["label"] == "NEUTRAL":
        final_sentiment = image_sentiment
    elif image_sentiment == "NEUTRAL":
        final_sentiment = text_result["label"]
    elif text_result["label"] == image_sentiment:
        final_sentiment = text_result["label"]
    else:
        if text_result["score"] >= 0.85:
            final_sentiment = text_result["label"]
        else:
            final_sentiment = image_sentiment

    return {
        "text_sentiment": text_result,
        "image_sentiment": image_result,
        "final_sentiment": final_sentiment
    }
