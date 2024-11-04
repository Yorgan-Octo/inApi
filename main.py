import sys
import subprocess
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
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
    print(f"Error: {e}")
    try:
        # Добавляем директорию YOLOv5 в путь поиска
        yolov5_path = os.path.join(os.getcwd(), 'yolov5')  # или абсолютный путь к yolov5
        sys.path.append(yolov5_path)

        from yolov5.models.common import DetectMultiBackend  # Импорт из YOLOv5

        model = DetectMultiBackend(model_path, device=device)
    except Exception as r:
        print(f"Error2: {r}")




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



# Папка для сохранения изображений
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Чтение изображения из загруженного файла
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Генерация уникального имени для файла
        image_filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(output_dir, image_filename)

        # Сохранение изображения в папку
        cv2.imwrite(image_path, image)

        # Выполнение предсказания модели на сохраненном изображении
        results = model(image_path)

        # Печать результатов (для отладки)
        results.print()

        # Отрисовка bounding box'ов на изображении
        results.render()  # Это обновляет изображение с отрисованными bounding box'ами

        # Получение обработанного изображения с отрисованными результатами
        rendered_image = results.ims[0]  # Извлечение обработанного изображения как NumPy массив

        # Преобразование NumPy массива в байтовый поток
        _, buffer = cv2.imencode('.png', rendered_image)  # Кодирование в PNG формат
        output_buffer = io.BytesIO(buffer)  # Создание буфера
        output_buffer.seek(0)  # Перемещаем указатель в начало буфера

        # Возвращаем изображение как поток
        return StreamingResponse(output_buffer, media_type="image/png")

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.post("/predictnew/")
async def predict(file: UploadFile = File(...)):
    try:
        # Чтение изображения из загруженного файла
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Генерация уникального имени для файла
        image_filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(output_dir, image_filename)

        # Сохранение изображения в папку
        cv2.imwrite(image_path, image)

        # Задаем свою директорию для сохранения
        custom_output_dir = os.path.join(output_dir, "yolo_results")

        # Запуск YOLOv5 через subprocess для предсказания
        command = [
            'python', 'yolov5/detect.py',
            '--weights', model_path,
            '--source', image_path,
            '--conf-thres', '0.25',
            '--project', custom_output_dir,  # Ваша директория для сохранения
            '--name', 'result',  # Имя подпапки
            '--save-txt', '--save-conf'
        ]

        # Запуск команды и захват её вывода
        result = subprocess.run(command, capture_output=True, text=True)

        # Логируем результат выполнения команды
        print("Subprocess result:", result.stdout)
        print("Subprocess error (if any):", result.stderr)

        # Проверяем сохраненное изображение в папке custom_output_dir
        annotated_image_path = os.path.join(custom_output_dir, "result", image_filename)

        if not os.path.exists(annotated_image_path):
            raise HTTPException(status_code=500, detail="Model failed to process the image")

        # Чтение аннотированного изображения
        rendered_image = cv2.imread(annotated_image_path)

        # Преобразование аннотированного изображения в байтовый поток
        _, buffer = cv2.imencode('.png', rendered_image)
        output_buffer = io.BytesIO(buffer)
        output_buffer.seek(0)

        # Возвращаем изображение как поток
        return StreamingResponse(output_buffer, media_type="image/png")

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)