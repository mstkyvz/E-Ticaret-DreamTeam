import streamlit as st
from PIL import Image
import requests
import json
import re
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
from src.Translater import llm
from src.utils import timer
from src.html_templates import (
    get_page_style, get_header, preview_container_template, 
    output_view_template, get_additional_styles,start_text
)
import warnings
warnings.filterwarnings("ignore")

print(start_text)


MODEL_CKPT = "papluca/xlm-roberta-base-language-detection"
API_URL = "http://0.0.0.0:8000/process"
SUPPORTED_LANGUAGES = ["en", "zh", "es", "fr", "tr", "ar", "ru", "it"]
LANGUAGE_MAP = {
    "en": "english", "zh": "chinese", "es": "spanish", "fr": "french",
    "tr": "turkish", "ar": "arabic", "ru": "russian", "it": "italian"
}


tokenizer_lang = AutoTokenizer.from_pretrained(MODEL_CKPT)
model_lang = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT)

@timer
def image_to_base64(image):
    import io
    import base64
    
    max_width, max_height = 1080, 720
    
    original_width, original_height = image.size
    
    if original_width > max_width or original_height > max_height:
        ratio = min(max_width / original_width, max_height / original_height)
        new_size = (int(original_width * ratio), int(original_height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

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
    return {lang: Image.open(f"static/flag/{lang}.png") for lang in SUPPORTED_LANGUAGES}
flags = get_flags()

@timer
def preview_container(uploaded_file, description, flag=None):
    image = Image.open(uploaded_file) if uploaded_file else Image.open("static/empty.png")
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
    print(files,data)
    response = requests.post(API_URL, files=files, data=data)
    
    output = response.json()['output']
    base64_image = image_to_base64(Image.open("static/temiz.png"))

    try:
        output = extract_and_convert_to_json(output)
        urun_adi = output['product_name']
        urun_aciklama = output['product_detailed_description']
    except Exception as e:
        print(f"Error processing output: {e}")
        urun_adi = ""
        urun_aciklama = output
    return  urun_adi,urun_aciklama,output,base64_image           
                
                
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
                    print("---"*10,flag_lang)
                    urun_adi,urun_aciklama,output,base64_image=main_lang(uploaded_file,description,flag_lang)
                
                    tabs = st.tabs(["_"] + SUPPORTED_LANGUAGES)

                    with tabs[0]:
                        output_view(urun_adi, urun_aciklama, base64_image, flag)

                    with tabs[SUPPORTED_LANGUAGES.index(flag_lang) + 1]:
                        output_view(urun_adi, urun_aciklama, base64_image, flag)

                    progress_bar = st.progress(0)
                    for trans_lang, lang_data in get_lang_data(output, flag_lang, progress_bar):
                        if lang_data:
                            with tabs[SUPPORTED_LANGUAGES.index(trans_lang) + 1]:
                                output_view(lang_data['product_name'], lang_data['product_detailed_description'], base64_image, lang_data['flag'])
                        else:
                            with tabs[SUPPORTED_LANGUAGES.index(trans_lang) + 1]:
                                st.write(f"Data not available for {trans_lang}")

                    progress_bar.empty()
        else:
            preview_container(uploaded_file, description, flag)

    st.markdown(get_additional_styles(), unsafe_allow_html=True)

if __name__ == "__main__":
    main()