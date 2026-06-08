# -*- coding: utf-8 -*-
"""
Приложение: Распознавание эмоций на лицах
Модель: dima806/facial_emotions_image_detection (Hugging Face ViT)
Автор: Кузнецов Никита
"""

import os
from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline
from PIL import Image

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 МБ

ALLOWED = {"png", "jpg", "jpeg", "webp"}

# Загружаем модель один раз при старте приложения
print("Загрузка модели... (первый запуск может занять 1-2 минуты)")
classifier = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
)
print("Модель загружена!")

EMOTION_RU = {
    "angry":   "Злость",
    "disgust": "Отвращение",
    "fear":    "Страх",
    "happy":   "Радость",
    "neutral": "Нейтрально",
    "sad":     "Грусть",
    "surprise":"Удивление",
}

EMOTION_EMOJI = {
    "angry":   "😠",
    "disgust": "🤢",
    "fear":    "😨",
    "happy":   "😊",
    "neutral": "😐",
    "sad":     "😢",
    "surprise":"😲",
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if not file or not allowed_file(file.filename):
        return render_template("index.html", error="Загрузите изображение (jpg, png, webp)")

    filename = file.filename
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    image = Image.open(save_path).convert("RGB")
    results = classifier(image)

    top = results[0]
    label = top["label"].lower()
    confidence = round(top["score"] * 100, 1)

    predictions = [
        {
            "label": r["label"].lower(),
            "label_ru": EMOTION_RU.get(r["label"].lower(), r["label"]),
            "emoji": EMOTION_EMOJI.get(r["label"].lower(), ""),
            "score": round(r["score"] * 100, 1),
        }
        for r in results
    ]

    return render_template(
        "result.html",
        image_path=save_path,
        top_emotion=EMOTION_RU.get(label, label),
        top_emoji=EMOTION_EMOJI.get(label, ""),
        confidence=confidence,
        predictions=predictions,
    )


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
