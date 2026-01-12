from fastapi import FastAPI
from pydantic import BaseModel
from gpt4all import GPT4All
import os
import requests

# -------------------------
# Descargar modelo si no existe
# -------------------------
MODEL_PATH = "gpt4all-lora-quantized.bin"
MODEL_URL = "https://gpt4all.io/models/gpt4all-lora-quantized.bin"

if not os.path.exists(MODEL_PATH):
    print("Descargando modelo...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Modelo descargado")

# -------------------------
# Inicializar FastAPI y modelo
# -------------------------
app = FastAPI()
model = GPT4All(MODEL_PATH)

# -------------------------
# Request y endpoint
# -------------------------
class Request(BaseModel):
    prompt: str

@app.post("/chat")
def chat(req: Request):
    response = model.generate(req.prompt)
    return {"completion": response}
