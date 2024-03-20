"""
Here’s the exam:
1. Select a Causal language Model
2. ⁠You can freely train/fine-tune/or use it outside the box into what use-case you prefer 
3. ⁠Deploy that to heroku, render, or any free deployment platforms (free only) using Fast API. 
4. ⁠Must be able to do post requests remotely.
5. Upload it to github with a short readme on how to install and infer on your endpoint
"""
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig
from pydantic import BaseModel
import torch

#Credits to https://www.kaggle.com/datasets/fabiochiusano/medium-articles for the dataset

app = FastAPI()

async def generate_text(title, max_length=1000, top_k=50, model_dir="./model/custom-gpt2-model", tokenizer_dir="./model/custom-gpt2-tokenizer"):
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
            early_stopping=True,
        )
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True) 
    return generated_text

class RequestParams(BaseModel):
    title: str
    max_length: int = 1000
    top_k: int = 50

@app.post("/generate-article")
async def handle_request(request: RequestParams):
    generated_article = await generate_text(request.title, request.max_length, request.top_k)
    return {"generated_article": generated_article}