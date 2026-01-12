# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # <--- importar
from pydantic import BaseModel
from transformers import pipeline

# -------------------------
# Inicializar FastAPI
# -------------------------
app = FastAPI(title="Lightweight GPT API")

# -------------------------
# Configurar CORS
# -------------------------
origins = [
    "*",  # permite cualquier origen, para pruebas
    # "http://localhost:8000",  # podrías restringirlo a tu frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Endpoint de request
# -------------------------
class Request(BaseModel):
    prompt: str
    max_length: int = 100

# -------------------------
# Cargar modelo ligero distilgpt2
# -------------------------
print("Cargando modelo...")
generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")
print("Modelo cargado.")

# -------------------------
# Endpoint POST /chat
# -------------------------
@app.post("/chat")
def chat(req: Request):
    output = generator(
        req.prompt,
        max_length=req.max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
    )
    return {"completion": output[0]["generated_text"]}
