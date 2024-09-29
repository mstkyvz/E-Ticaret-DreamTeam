from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM,AutoModelForSequenceClassification,AutoTokenizer
import io
import os

app = FastAPI()

model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis1.6-Gemma2-9B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=1024,
                                             trust_remote_code=True).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()


model_ckpt = "papluca/xlm-roberta-base-language-detection"
tokenizer_lang = AutoTokenizer.from_pretrained(model_ckpt)
model_lang = AutoModelForSequenceClassification.from_pretrained(model_ckpt)



def get_lang(text):
    lang = tokenizer_lang(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model_lang(**lang).logits

    preds = torch.softmax(logits, dim=-1)
    id2lang = model_lang.config.id2label
    vals, idxs = torch.max(preds, dim=1)


    first_key = id2lang[idxs[0].item()]
    return first_key


def read_text_file(file_name):
    language_dir = 'prompt/language/'
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
async def process_image_and_text(image: UploadFile = File(...), text: str = Form(...)):
    
    
    image_content = await image.read()
    pil_image = Image.open(io.BytesIO(image_content))
    lang=get_lang(text)
    prompt = read_text_file(f"{lang}.txt")+text
    
    query = f'<image>\n{prompt}'
    
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, [pil_image])
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]
    
    
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
    
    return JSONResponse(content={"output": output})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)