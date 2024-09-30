import asyncio
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import os
from functools import lru_cache
from typing import List, Dict
import base64
from transformers import AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Use a thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

# Load model once at startup and move to GPU
model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis1.6-Gemma2-9B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    multimodal_max_length=2048,
    trust_remote_code=True
).cuda()

# JIT compile the model for faster inference
model = torch.jit.script(model)

text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

# Cache for language files
@lru_cache(maxsize=None)
def read_text_file(file_name: str) -> str:
    language_dir = 'prompt/language/'
    file_path = os.path.join(language_dir, file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        fallback_path = os.path.join(language_dir, 'en.txt')
        with open(fallback_path, 'r', encoding='utf-8') as file:
            return file.read()

# Cache for base64 encoded images
@lru_cache(maxsize=100)
def image_to_base64(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode('utf-8')

async def process_image(image_content: bytes) -> Image.Image:
    return await asyncio.to_thread(Image.open, io.BytesIO(image_content))

async def run_model_inference(input_ids, pixel_values, attention_mask):
    with torch.inference_mode():
        gen_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": False,
            "eos_token_id": model.generation_config.eos_token_id,
            "pad_token_id": text_tokenizer.pad_token_id,
            "use_cache": True
        }
        output_ids = await asyncio.to_thread(
            model.generate,
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **gen_kwargs
        )
        return text_tokenizer.decode(output_ids[0], skip_special_tokens=True)

@app.post("/process/")
async def process_image_and_text(
    image: UploadFile = File(...),
    text: str = Form(...),
    lang: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    image_content = await image.read()
    
    # Process image asynchronously
    pil_image = await process_image(image_content)
    
    # Read language file (cached)
    prompt = read_text_file(f"{lang}.txt") + text
    query = f'<image>\n{prompt}'
    
    # Preprocess inputs
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, [pil_image])
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    
    # Move tensors to GPU
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]
    
    # Run model inference asynchronously
    output = await run_model_inference(input_ids, pixel_values, attention_mask)
    
    # Encode image to base64 in the background
    base64_image = await asyncio.to_thread(image_to_base64, image_content)
    
    return JSONResponse(content={"output": output, "image": base64_image})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)