from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

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