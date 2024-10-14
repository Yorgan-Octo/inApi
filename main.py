from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import cv2
import io
import torch
import os
import uuid

app = FastAPI()

# Загрузка вашей обученной модели
model_path = 'yolov5/runs/train/bcd/weights/best.pt'  # Путь к вашей обученной модели
device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    # Загрузка модели YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=device)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")


# Модель данных
class Message(BaseModel):
    message: str

# Пример данных
data = {
    "message": "Hello, World!",
}

@app.get("/")
async def read_root():
    return data

@app.get("/api/data")
async def get_data():
    return data

@app.post("/api/data")
async def update_data(msg: Message):
    data["message"] = msg.message
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)