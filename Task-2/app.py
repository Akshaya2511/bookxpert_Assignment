# app.py
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI(title="Local Recipe Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# Load your fine-tuned recipe model
MODEL_NAME = "./recipe_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Request body
class QueryInput(BaseModel):
    ingredients: str

def generate_recipe(ingredients: str) -> str:
    # Make the prompt *exactly* like the examples used for fine-tuning
    prompt = prompt = f"Ingredients: {ingredients}\nRecipe: Please give a short, step-by-step recipe (2–4 lines)."
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate: use max_new_tokens, set pad/eos and sampling params
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=120,            # generate up to 120 new tokens
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process: extract the part after the "Recipe:" marker
    if "Recipe:" in text:
        recipe_part = text.split("Recipe:", 1)[1].strip()
    else:
        # fallback: remove the prompt and keep what is generated
        recipe_part = text.replace(prompt, "").strip()

    # Trim to the first double newline or if too long, keep only first 2 sentences
    # (adjust heuristics to your taste)
    if "\n\n" in recipe_part:
        recipe_part = recipe_part.split("\n\n", 1)[0].strip()
    else:
        # naive sentence limiter (first 2 sentences)
        sentences = recipe_part.split(". ")
        recipe_part = ". ".join(sentences[:2]).strip()
        if not recipe_part.endswith("."):
            recipe_part += "."

    return recipe_part


@app.post("/ask/")
async def ask_recipe(query: QueryInput):
    recipe = generate_recipe(query.ingredients)
    return {"ingredients": query.ingredients, "suggested_recipe": recipe}
