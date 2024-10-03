from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM,BitsAndBytesConfig
import io
import os
from time import perf_counter
from utils import timer
# manuel quantization icin
from optimum.quanto import quantize, freeze, qfloat8, qint4, qint8
import gc

app = FastAPI()

model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis1.6-Gemma2-9B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=1024,
                                             max_memory={0: "16GB"},
                                             trust_remote_code=True).cuda()

image_info="""given the description of a product, generate a prompt of keywords that I can use to generate white background professional-looking product image of the given product. the prompt you generated will then be used in a Stable Diffusion pipeline for Image 2 Image generation. The label can be several words as long as it defines the object accurately. the label should define the objects' physical appearance in the image as well

description:{}"""


# # attentiondaki ve mlpdeki lineer katmanlari float8 quantize et
# # 19 gb VRAM -> 7(emin degilim?)gb VRAM (idle) -> ~12gb VRAM (islemdeyken)
# for layer in model.llm.model.layers:
#     quantize(layer.self_attn, weights=qfloat8, activations=qfloat8)
#     freeze(layer.self_attn)
#     quantize(layer.mlp, weights=qfloat8, activations=qfloat8)
#     freeze(layer.mlp)

# for layer in model.visual_tokenizer.backbone.vision_model.encoder.layers:
#     quantize(layer.self_attn, weights=qfloat8, activations=qfloat8)
#     freeze(layer.self_attn)
#     quantize(layer.mlp, weights=qfloat8, activations=qfloat8)
#     freeze(layer.mlp)

# # visual tokenizerdaki headde de sequential linear katman var, bunlar da quantize edilebiliyor
# quantize(model.visual_tokenizer.head, weights=qfloat8, activations=qfloat8)
# freeze(model.visual_tokenizer.head)
# quantize(model.visual_tokenizer.backbone.vision_model.head, weights=qfloat8, activations=qfloat8)
# freeze(model.visual_tokenizer.backbone.vision_model.head)

# gc.collect()
# torch.cuda.empty_cache()

# Get the data type of the model's parameters

# eger model pipeline ile wraplenirse
# model.enable_model_cpu_offload() falan kullanilabilir 
# o da VRAM kullanimini azaltir modeli gerektigi zaman CPUya offladigi icin. 
# belki bazi katmanlar qint4'e de maplenebilir ama hangi katmanlar uyumlu onu bilmiyorum
# modelde de zaten 8bit quantization uygulayabilecegim kimse kalmadi

# o zaman, kalan secenekler:
# - KV cache boyutunu kucultmek
# - islenen token miktarini azaltmak -> islenen pixel patch sayisini azaltmak
# - daha dusuk quantization secenekleri (qint8 ve qint4 var)
# - batch normlarda ogrenilen parametreleri de mi quantize etsek? 

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
    model.cuda()
    model_dtype = next(model.parameters()).dtype
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    model.visual_tokenizer = model.visual_tokenizer.to(model.device)
    pil_image = Image.open(image_content)
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
            print(type(pixel_values), type(attention_mask), type(input_ids))
            output_ids = model.generate(input_ids.to(model.device), pixel_values=pixel_values, attention_mask=attention_mask.to(model.device), **gen_kwargs)[0]
            output = text_tokenizer.decode(output_ids, skip_special_tokens=True)

            
            
    t4 = perf_counter()
    del text_tokenizer, visual_tokenizer, input_ids, attention_mask, pixel_values, output_ids
    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Time took to generate the output: {t4-t3:.4f}")
    print("-"*50)
    return JSONResponse(content={"output": output})



# @app.post("/process/")
# async def process_image_and_text(image: UploadFile = File(...), text: str = Form(...),lang: str = Form(...)):
#     image_content = await image.read()
#     model.cuda()
#     model_dtype = next(model.parameters()).dtype
#     text_tokenizer = model.get_text_tokenizer()
#     visual_tokenizer = model.get_visual_tokenizer()

#     model.visual_tokenizer = model.visual_tokenizer.to(model.device)
#     pil_image = Image.open(io.BytesIO(image_content))
#     prompt = read_text_file(f"{lang}.txt")+text
#     query = f'<image>\n{prompt}'
    
#     prompt2=image_info.format(text)
    
#     query2 = f'<image>\n{prompt}'

#     t0 = perf_counter()
#     prompt, input_ids, pixel_values = model.preprocess_inputs(query, [pil_image])
#     print("-"*50)
#     t1 = perf_counter()
#     print(f"Time took to process the inputs: {t1-t0:.4f}")
#     attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
#     print("-"*50)
#     t2 = perf_counter()
#     print(f"Time took to process the attention mask: {t2-t1:.4f}")
#     input_ids = input_ids.unsqueeze(0).to(device=model.device)
#     attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
#     pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]
    
#     print("-"*50)
#     t3 = perf_counter()
#     print(f"Time took to move everything to the GPU: {t3-t2:.4f}")
#     print("-"*50)
#     gen_kwargs = dict(
#                 max_new_tokens=1024,
#                 do_sample=False,
#                 top_p=None,
#                 top_k=None,
#                 temperature=None,
#                 repetition_penalty=None,
#                 eos_token_id=model.generation_config.eos_token_id,
#                 pad_token_id=text_tokenizer.pad_token_id,
#                 use_cache=True
#             )
#     with torch.inference_mode():
#             output_ids = model.generate(input_ids.to(model.device), pixel_values=pixel_values, attention_mask=attention_mask.to(model.device), **gen_kwargs)[0]
#             output = text_tokenizer.decode(output_ids, skip_special_tokens=True)

#     del input_ids, attention_mask, pixel_values, output_ids
    
#     prompt, input_ids, pixel_values = model.preprocess_inputs(query2, [pil_image])
#     print("-"*50)
#     t1 = perf_counter()
#     print(f"Time took to process the inputs: {t1-t0:.4f}")
#     attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
#     print("-"*50)
#     t2 = perf_counter()
#     print(f"Time took to process the attention mask: {t2-t1:.4f}")
#     input_ids = input_ids.unsqueeze(0).to(device=model.device)
#     attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
#     pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]
#     with torch.inference_mode():
#             output_ids = model.generate(input_ids.to(model.device), pixel_values=pixel_values, attention_mask=attention_mask.to(model.device), **gen_kwargs)[0]
#             output_for_image = text_tokenizer.decode(output_ids, skip_special_tokens=True)
            
#     t4 = perf_counter()
#     del text_tokenizer, visual_tokenizer, input_ids, attention_mask, pixel_values, output_ids
#     model.cpu()
#     torch.cuda.empty_cache()
#     gc.collect()
#     print(f"Time took to generate the output: {t4-t3:.4f}")
#     print("-"*50)
#     return JSONResponse(content={"output": output,"output_for_image":output_for_image})


@app.post("/image_info/")
async def image_and_text(image: UploadFile = File(...), text: str = Form(...),lang: str = Form(...)):
    image_content = await image.read()
    pil_image = Image.open(image_content)
    prompt = text
    query = f'<image>\n{prompt}'
    
    model.cuda()
    model_dtype = next(model.parameters()).dtype
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    
    model.visual_tokenizer = model.visual_tokenizer.to(model.device)

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
            print(type(pixel_values), type(attention_mask), type(input_ids))
            output_ids = model.generate(input_ids.to(model.device), pixel_values=pixel_values, attention_mask=attention_mask.to(model.device), **gen_kwargs)[0]
            output = text_tokenizer.decode(output_ids, skip_special_tokens=True)

            
            
    t4 = perf_counter()
    del text_tokenizer, visual_tokenizer, input_ids, attention_mask, pixel_values, output_ids
    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Time took to generate the output: {t4-t3:.4f}")
    print("-"*50)
    return JSONResponse(content={"output": output})


if __name__ == "__main__":
     # test_post()
     import uvicorn
     uvicorn.run(app, host="0.0.0.0", port=8000)