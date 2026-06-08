# -*- coding: utf-8 -*-
"""
FastAPI — REST API для распознавания эмоций на лицах.
Модель: dima806/facial_emotions_image_detection (Hugging Face ViT)
Автор: Кузнецов Никита

Запуск:
    uvicorn api:app --reload

Swagger UI: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
import io
import uvicorn

app = FastAPI(
    title="Emotion Recognition API",
    description="API для распознавания эмоций на фотографиях лиц. "
                "Модель: dima806/facial_emotions_image_detection (Hugging Face ViT).",
    version="1.0.0",
)

EMOTION_RU = {
    "angry":    "Злость",
    "disgust":  "Отвращение",
    "fear":     "Страх",
    "happy":    "Радость",
    "neutral":  "Нейтрально",
    "sad":      "Грусть",
    "surprise": "Удивление",
}

print("Загрузка модели...")
classifier = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
)
print("Модель готова!")


class EmotionResult(BaseModel):
    label: str
    label_ru: str
    score: float


class PredictResponse(BaseModel):
    top_emotion: str
    top_emotion_ru: str
    confidence: float
    all_emotions: list[EmotionResult]


@app.get("/", summary="Проверка работоспособности")
def root():
    return {"status": "ok", "model": "dima806/facial_emotions_image_detection"}


@app.get("/health", summary="Health check")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse, summary="Распознать эмоцию на фото")
async def predict(file: UploadFile = File(..., description="Фото лица (jpg, png, webp)")):
    """
    Принимает фотографию лица и возвращает предсказанные эмоции.

    - **file**: изображение в формате JPG, PNG или WEBP
    - Возвращает топ-эмоцию с уверенностью и список всех 7 эмоций с вероятностями
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

    results = classifier(image)

    top = results[0]
    top_label = top["label"].lower()

    all_emotions = [
        EmotionResult(
            label=r["label"].lower(),
            label_ru=EMOTION_RU.get(r["label"].lower(), r["label"]),
            score=round(r["score"], 4),
        )
        for r in results
    ]

    return PredictResponse(
        top_emotion=top_label,
        top_emotion_ru=EMOTION_RU.get(top_label, top_label),
        confidence=round(top["score"], 4),
        all_emotions=all_emotions,
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
