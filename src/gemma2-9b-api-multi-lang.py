from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM,BitsAndBytesConfig
import io
import os
from time import perf_counter
from utils import timer

app = FastAPI()

model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis1.6-Gemma2-9B",
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto",
                                             multimodal_max_length=2048,
                                             trust_remote_code=True).cuda()


text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

@timer
def read_text_file(file_name):
    language_dir = '../prompt/language/'
    file_path = os.path.join(language_dir, file_name)

    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        fallback_path = os.path.join(language_dir, 'en.txt')
        if os.path.isfile(fallback_path):
            with open(fallback_path, 'r', encoding='utf-8') as file:
                return file.read()

@app.post("/process/")
async def process_image_and_text(image: UploadFile = File(...), text: str = Form(...),lang: str = Form(...)):
    image_content = await image.read()
    pil_image = Image.open(io.BytesIO(image_content))
    prompt = read_text_file(f"{lang}.txt")+text
    query = f'<image>\n{prompt}'
    
    t0 = perf_counter()
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, [pil_image])
    print("-"*50)
    t1 = perf_counter()
    print(f"Time took to process the inputs: {t1-t0:.4f}")
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    print("-"*50)
    t2 = perf_counter()
    print(f"Time took to process the attention mask: {t2-t1:.4f}")
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]
    print("-"*50)
    t3 = perf_counter()
    print(f"Time took to move everything to the GPU: {t3-t2:.4f}")
    print("-"*50)
    

    with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True
            )

            output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = text_tokenizer.decode(output_ids, skip_special_tokens=True)

    t4 = perf_counter()
    print(f"Time took to generate the output: {t4-t3:.4f}")
    print("-"*50)
    
    return JSONResponse(content={"output": output})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)