import streamlit as st
from PIL import Image
import requests
import json
import re
from transformers import AutoModelForCausalLM,AutoModelForSequenceClassification,AutoTokenizer
import torch
import os
from Translater import *

model_ckpt = "papluca/xlm-roberta-base-language-detection"
tokenizer_lang = AutoTokenizer.from_pretrained(model_ckpt)
model_lang = AutoModelForSequenceClassification.from_pretrained(model_ckpt)

url="http://0.0.0.0:8000/process"

all_lang=["en","zh","es","fr","tr","ar","ru","it"]

all_lang_map={"en":"english",
              "zh":"chinese",
              "es":"spanish",
              "fr":"french",
              "tr":"turkish",
              "ar":"arabic",
              "ru":"russian",
              "it":"italian"}

def image_to_base64(image):
    import io
    import base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode('utf-8')

def extract_and_convert_to_json(input_string):
    input_string=input_string.replace("```json","").replace("`","").replace("“","\"").replace("”","\"")
    pattern = r'\{([^}]*)\}'
    matches = re.findall(pattern, input_string)
    combined = '{' + '}{'.join(matches) + '}'
    
    try:
        json_output = json.loads(combined)
        return json_output
    except json.JSONDecodeError:
        return "Geçerli bir JSON oluşturulamadı."
    
def get_lang(text):
    lang = tokenizer_lang(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model_lang(**lang).logits

    preds = torch.softmax(logits, dim=-1)
    id2lang = model_lang.config.id2label
    vals, idxs = torch.max(preds, dim=1)


    first_key = id2lang[idxs[0].item()]
    return first_key

st.set_page_config(layout="wide")


hide_streamlit_style = """
            <style>
                header {visibility: hidden;}
                .streamlit-footer {display: none;}
                .st-emotion-cache-uf99v8 {display: none;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown(
        f"""
        <div style="display:flex; flex-direction: column; justify-content:  flex-start; align-items: center; width: 100%; margin-top:-100px;">
            <h1>DREAM TEAM</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


col1, col2 = st.columns([1, 1])


def preview_container(uploaded_file,description,flag=None):
    if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
            image = Image.open("empty.png")

    if not description:
            description="Açıklama"
    
    st.markdown(
            f"""
            <div style="display:flex; flex-direction: column;  justify-content:  flex-start; border-radius: 5px; align-items: center; width: 100%; height: 600px; box-shadow: rgba(100, 100, 111, 0.2) 0px 7px 29px 0px;">
                <div style="display:flex; align-items: start; width: 100%;">
                <div style="width: 40%;">
                <div style="width: 100%;">
                <div style="margin: 30px; width: 100px; height: 50px;">
                    <img src="data:image/png;base64,{image_to_base64(flag)}" 
                         style="width: 100%; height: 100%; object-fit: contain;"/>
                </div>
                </div>
                </div>
                <h2>Ön İzleme</h2>
                </div>
                <div style="width: 90%; height: 600px; overflow: hidden;">
                    <img src="data:image/png;base64,{image_to_base64(image)}" style="width: 100%; height: 100%; object-fit: contain;"/>
                </div>
               <h5 style="margin:30px;" >{description}</h5> 
            </div>
            """,
            unsafe_allow_html=True,
        )


def output_view(urun_adi,urun_aciklama,base64_image,flag):
    st.markdown(
            f"""
            <div style="display:flex; flex-direction: column; justify-content: flex-start; border-radius: 5px; align-items: center; width: 100%; height: 600px; box-shadow: rgba(100, 100, 111, 0.2) 0px 7px 29px 0px;">
                <div style="display:flex; align-items: start; width: 100%;">
                <div style="width: 40%;">
                <div style="width: 100%;">
                <div style="margin: 30px;width: 100px; height: 50px;">
                <img src="data:image/png;base64,{image_to_base64(flag)}" style="width: 100%; height: 100%; object-fit: contain;"/>
                </div>
                </div>
                </div>
                <h2>Çıktı</h2>
                </div>
            <h4>{urun_adi}</h4>
                <div style="width: 90%; height: 600px; overflow: hidden;">
                    <img src="data:image/png;base64,{base64_image}" style="width: 100%; height: 100%; object-fit: contain;"/>
                </div>
            <h5 style="margin:30px;" >{urun_aciklama}</h5>    
            </div>
            """,
            unsafe_allow_html=True,
        )


def get_flags():
    data={}
    for lang in all_lang:
        flag = os.path.join("static/flag/", f"{lang}.png")
        flag=Image.open(flag)
        data[lang]=flag
    return data

flags=get_flags()    
    
def get_lang_data(text,lang):
    data={}
    for tans_lang in all_lang:
        if tans_lang!=lang:
            try:
                response=extract_and_convert_to_json(llm(text,all_lang_map[lang],all_lang_map[tans_lang]))
                print(response)
                product_name=response['product_name']
                product_detailed_description=response['product_detailed_description']
                data[tans_lang]={"product_name":product_name,"product_detailed_description":product_detailed_description,"flag":flags[tans_lang]}
            except:
                pass
    return data
with col1:
    with st.container():
        st.header("Görsel Yükle")
        uploaded_file = st.file_uploader("Görsel Yükle", type=["jpg", "jpeg", "png"])
        description = st.text_input("Açıklama")
        gonder = st.button("Gönder",use_container_width=True)
        alert=st.empty()
    if description:
        flag_lang=get_lang(description)
        if not flag_lang in all_lang:
            flag_lang="tr"
    else:
        flag_lang="tr"
    flag=flags[flag_lang]
    
    st.markdown(
        """
    <style>
      div[data-testid="stVerticalBlockBorderWrapper"] {
           display: flex; 
           justify-content: space-between; 
           align-items: center; 
           width: 100%; 
           height: 600px;
           margin-bottom:30px;
        }

   div[data-testid="stHorizontalBlock"]{
           gap: 2rem; 
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    

with col2:
    if gonder:
        files = {'image': uploaded_file}
        data = {'text': description,'lang':flag_lang}
        if not uploaded_file:
            alert.warning('Resimi Tekrar Yükleyin', icon="⚠️")
            preview_container(uploaded_file,description)
        else:
            
            image = Image.open("temiz.png")
            response = requests.post(url, files=files, data=data)
            output=response.json()['output']
            base64_image = image_to_base64(image)
            try:
                output=extract_and_convert_to_json(output)
                urun_adi=output['product_name']
                urun_aciklama=output['product_detailed_description']
            except Exception as e:
                print(e)
                urun_adi=""
                urun_aciklama=output
            _,en,zh,es,fr,tr,ar,ru,it=st.tabs(["_","  en  ","  zh  ","  es  ","  fr  ","  tr  ","  ar  ","  ru  ","  it  "])
            with _:
                output_view(urun_adi,urun_aciklama,base64_image,flag)
            tabs=[en,zh,es,fr,tr,ar,ru,it]
            tabs_name=all_lang
            lang_datas=get_lang_data(output,flag_lang)
            for lang_name,lang_tab in zip(tabs_name,tabs):
                if lang_name!=flag_lang:
                    with lang_tab:
                        try:
                            lang_data=lang_datas[lang_name]
                            product_name=lang_data['product_name']
                            product_detailed_description=lang_data['product_detailed_description']
                            flag=lang_data['flag']
                            output_view(product_name,product_detailed_description,base64_image,flag)
                        except:
                            pass
                else:
                    with lang_tab:
                        try:
                                product_name=output['product_name']
                                product_detailed_description=output['product_detailed_description']
                                flag=lang_data['flag']
                                output_view(product_name,product_detailed_description,base64_image,flag)
                        except:
                                pass

    else:
        
        preview_container(uploaded_file,description,flag)

