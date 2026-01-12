# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -------------------------
# Inicializar FastAPI
# -------------------------
app = FastAPI(title="Lightweight GPT API")

# -------------------------
# Endpoint de request
# -------------------------
class Request(BaseModel):
    prompt: str
    max_length: int = 100  # límite de tokens generados

# -------------------------
# Cargar modelo y tokenizer
# -------------------------
# Esto puede tardar unos segundos al iniciar
print("Cargando modelo EleutherAI/gpt-neo-125M...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
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
    # El pipeline devuelve una lista de diccionarios
    return {"completion": output[0]["generated_text"]}
