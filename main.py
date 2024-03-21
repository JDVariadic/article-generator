from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
import torch

#Credits to https://www.kaggle.com/datasets/fabiochiusano/medium-articles for the dataset

app = FastAPI()
TOKEN_LIMIT = 400
TOP_K_LIMIT = 50
async def generate_text(title, max_length=TOKEN_LIMIT, top_k=TOP_K_LIMIT, model_dir="./model/custom-gpt2-model", tokenizer_dir="./model/custom-gpt2-tokenizer"):
    if max_length > TOKEN_LIMIT:
        return {"error": "Limit for max_length is 400 tokens."}
    if top_k > TOP_K_LIMIT:
        return {"error": "Limit for top_k is 50."}
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    input_text = f"[TITLE] {title} [/TITLE]"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=input_ids,
            pad_token_id=tokenizer.pad_token_id,
            max_length=max_length,
            do_sample=True,
            top_k=top_k,
        )
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True) 
    return generated_text

class RequestParams(BaseModel):
    title: str
    max_length: int = TOKEN_LIMIT
    top_k: int = TOP_K_LIMIT

@app.post("/generate-article")
async def handle_request(request: RequestParams):
    generated_article = await generate_text(request.title, request.max_length, request.top_k)
    return {"generated_article": generated_article}