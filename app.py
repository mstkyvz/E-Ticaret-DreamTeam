import streamlit as st
from PIL import Image
import requests
import json
import re
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import torch
import os,io
from src.Translater import llm
from src.utils import timer,image_to_base64
from src.html_templates import (
    get_page_style, get_header, preview_container_template, 
    output_view_template, get_additional_styles,start_text
)
from config.config import Config
from src.workflow import generate_image
from src.GgufManager import GgufManager
from prompt import prompt
import time
import hashlib
import numpy as np
from io import BytesIO
import warnings
import cv2
warnings.filterwarnings("ignore")

print(start_text)

config=Config()

MODEL_CKPT = config.model_ckpt
API_URL = config.api_url
SUPPORTED_LANGUAGES = config.supported_languages
LANGUAGE_MAP = config.language_map


tokenizer_lang = AutoTokenizer.from_pretrained(MODEL_CKPT)
model_lang = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT)

ggufManager=GgufManager()

@timer
def extract_and_convert_to_json(input_string):
    input_string = input_string.replace("```json", "").replace("`", "").replace(""", "\"").replace(""", "\"")
    pattern = r'\{([^}]*)\}'
    matches = re.findall(pattern, input_string)
    combined = '{' + '}{'.join(matches) + '}'
    
    try:
        return json.loads(combined)
    except json.JSONDecodeError:
        return "Geçerli bir JSON oluşturulamadı."

@timer
def get_lang(text):
    lang = tokenizer_lang(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model_lang(**lang).logits
    preds = torch.softmax(logits, dim=-1)
    id2lang = model_lang.config.id2label
    _, idxs = torch.max(preds, dim=1)
    return id2lang[idxs[0].item()]

@timer
def setup_page():
    st.set_page_config(layout="wide")
    st.markdown(get_page_style(), unsafe_allow_html=True)
    st.markdown(get_header(), unsafe_allow_html=True)
    
@timer
def get_flags():
    return {lang: Image.open(config.flags_path / f"{lang}.png") for lang in SUPPORTED_LANGUAGES}
flags = get_flags()

@timer
def preview_container(uploaded_file, description, flag=None):
    image = Image.open(uploaded_file) if uploaded_file else Image.open(config.empty_image_path)
    description = description or "Açıklama"
    image_flag = image_to_base64(flag if flag else flags["tr"])
    image_base64 = image_to_base64(image)
    
    st.markdown(
        preview_container_template().format(
            image_flag=image_flag,
            image_base64=image_base64,
            description=description
        ),
        unsafe_allow_html=True,
    )
    
@timer
def output_view(urun_adi, urun_aciklama, base64_image, flag):
    flag_base64 = image_to_base64(flag)
    st.markdown(
        output_view_template().format(
            flag_base64=flag_base64,
            urun_adi=urun_adi,
            base64_image=base64_image,
            urun_aciklama=urun_aciklama
        ),
        unsafe_allow_html=True,
    )



@timer
def get_lang_data(text, lang, progress_bar):
    
    for trans_lang in SUPPORTED_LANGUAGES:
        if trans_lang != lang:
            try:
                response = extract_and_convert_to_json(llm(text, LANGUAGE_MAP[lang], LANGUAGE_MAP[trans_lang]))
                data = {
                    "product_name": response['product_name'],
                    "product_detailed_description": response['product_detailed_description'],
                    "flag": flags[trans_lang]
                }
                yield trans_lang, data
            except Exception as e:
                print(f"Error processing {trans_lang}: {e}")
                yield trans_lang, None
            finally:
                progress_bar.progress((SUPPORTED_LANGUAGES.index(trans_lang) + 1) / len(SUPPORTED_LANGUAGES))
                
@timer                
def main_lang(uploaded_file,description,flag_lang):
    files = {'image': uploaded_file}
    data = {'text': description, 'lang': flag_lang}
    print(files)
    response = requests.post(API_URL, files=files, data=data)
    
    output = response.json()['output']
    image = Image.open(uploaded_file)
    try:
        output = extract_and_convert_to_json(output)
        urun_adi = output['product_name']
        urun_aciklama = output['product_detailed_description']
    except Exception as e:
        print(f"Error processing output: {e}")
        urun_adi = ""
        urun_aciklama = output
    return  urun_adi,urun_aciklama,output,image



@timer                
def get_image_info(uploaded_file,description):
    files = {'image': uploaded_file}
    data = {'text': description, 'lang': "en"}
    print(files)
    response = requests.post(API_URL, files=files, data=data)
    
    output = response.json()['output']
    return output


@timer                
def get_sam_mask(file,des,mask_id):
    url = "http://localhost:8002/process_image/"
    files = {"file": file}
    data = {"description": des}
    print(type(file),type(des))
    response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        with open(f"/home/jupyter/temp/mask/{mask_id}.png", "wb") as f:
            f.write(response.content)
        print("Image processed and saved as processed_image.png")
        return response.content
    else:
        print(f"Error: {response.status_code}, {response.text}")


def hash_string(s):
    return hashlib.sha256(s.encode()).hexdigest()
    
@timer
def main():
    setup_page()
    col1, col2 = st.columns([1, 1])

    with col1:
        with st.container():
            st.header("Görsel Yükle")
            uploaded_file = st.file_uploader("Görsel Yükle", type=["jpg", "jpeg", "png"])
            description = st.text_input("Açıklama")
            gonder = st.button("Gönder", use_container_width=True)
            alert = st.empty()
            
            
        flag_lang = get_lang(description) if description else "tr"
        flag_lang = flag_lang if flag_lang in SUPPORTED_LANGUAGES else "tr"
        flag = flags[flag_lang]

    with col2:
        if gonder:
                if not uploaded_file:
                    alert.warning('Resimi Tekrar Yükleyin', icon="⚠️")
                    preview_container(uploaded_file, description)
                else:
                    byte_file=uploaded_file.read()
                    file_bytes = np.frombuffer(byte_file, np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    image_id = hash_string(str(image))
                    file_path = f"/home/jupyter/temp/in/{image_id}.png"
                    cv2.imwrite(file_path, image)
                    urun_adi,urun_aciklama,output,image=main_lang(file_path,description,flag_lang)
                    if f"{image_id}.png" in os.listdir(f"/home/jupyter/temp/out/"):
                        img = Image.open(f"/home/jupyter/temp/out/{image_id}.png")
                    else:
                        get_sam_mask(open(file_path,"rb"),description,image_id)
                        output_for_image=get_image_info(file_path,prompt.image_info.format(description))
                        output_for_image=prompt.image_prompt.format(output_for_image)
                        img=generate_image(output_for_image,file_path)
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        file_path = f"/home/jupyter/temp/out/{image_id}.png"
                        img.save(file_path, format="PNG")
                        
                    base64_image=image_to_base64(img)
                    tabs = st.tabs(["_"] + SUPPORTED_LANGUAGES)
                    
                    
                    with tabs[0]:
                        output_view(urun_adi, urun_aciklama, base64_image, flag)

                    with tabs[SUPPORTED_LANGUAGES.index(flag_lang) + 1]:
                        output_view(urun_adi, urun_aciklama, base64_image, flag)
                    ggufManager.run()
                    time.sleep(5)
                    progress_bar = st.progress(0)
                    for trans_lang, lang_data in get_lang_data(output, flag_lang, progress_bar):
                        if lang_data:
                            with tabs[SUPPORTED_LANGUAGES.index(trans_lang) + 1]:
                                output_view(lang_data['product_name'], lang_data['product_detailed_description'], base64_image, lang_data['flag'])
                        else:
                            with tabs[SUPPORTED_LANGUAGES.index(trans_lang) + 1]:
                                st.write(f"Data not available for {trans_lang}")

                    progress_bar.empty()
                    ggufManager.kill()
        else:
            preview_container(uploaded_file, description, flag)

    st.markdown(get_additional_styles(), unsafe_allow_html=True)

if __name__ == "__main__":
    main()