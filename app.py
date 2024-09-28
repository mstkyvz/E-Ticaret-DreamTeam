import streamlit as st
from PIL import Image
import requests
import json

url="http://0.0.0.0:8000/process"

def image_to_base64(image):
    import io
    import base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode('utf-8')


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
        <div style="display:flex; flex-direction: column; justify-content:  flex-start; align-items: center; width: 100%;">
            <h1>DREAM TEAM</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


col1, col2 = st.columns([1, 1])


with col1:
    with st.container():
        st.header("Görsel Yükle")
        uploaded_file = st.file_uploader("Görsel Yükle", type=["jpg", "jpeg", "png"])
        description = st.text_input("Açıklama")
        gonder = st.button("Gönder",use_container_width=True)


    st.markdown(
        """
    <style>
      div[data-testid="stVerticalBlockBorderWrapper"] {
           display: flex; 
           justify-content: space-between; 
           align-items: center; 
           width: 100%; 
           height: 600px;
           margin-bottom:100px;2
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
        data = {'text': description}
        image = Image.open("temiz.png")
        response = requests.post(url, files=files, data=data)
        print(response.json())
        output=response.json()['output']
        base64_image = image_to_base64(image)
        try:
            output=output.replace("```json","").replace("`","")
            output=json.loads(output)
            urun_adi=output['urun_adi']
            urun_aciklama=output['urun_detayli_aciklamasi']
        except Exception as e:
            print(e)
            urun_adi=""
            urun_aciklama=output
        st.markdown(
        f"""
        <div style="display:flex; flex-direction: column; justify-content: flex-start; border-radius: 5px; align-items: center; width: 100%; height: 600px; box-shadow: rgba(100, 100, 111, 0.2) 0px 7px 29px 0px;">
        <h2>Çıktı</h2>
        <h4>{urun_adi}</h4>
            <div style="width: 90%; height: 600px; overflow: hidden;">
                <img src="data:image/png;base64,{base64_image}" style="width: 100%; height: 100%; object-fit: contain;"/>
            </div>
        <h5 style="margin:30px;" >{urun_aciklama}</h5>    
        </div>
        """,
        unsafe_allow_html=True,
    )
        
    else:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = Image.open("empty.png")

        if not description:
            description="Açıklama"
        st.markdown(
            f"""
            <div style="display:flex; flex-direction: column;  justify-content:  flex-start; border-radius: 5px; align-items: center; width: 100%; height: 600px; box-shadow: rgba(100, 100, 111, 0.2) 0px 7px 29px 0px;">
                <h2>Preview</h2>
                <div style="width: 90%; height: 600px; overflow: hidden;">
                    <img src="data:image/png;base64,{image_to_base64(image)}" style="width: 100%; height: 100%; object-fit: contain; "/>
                </div>

               <h5 style="margin:30px;" >{description}</h5> 

            </div>
            """,
            unsafe_allow_html=True,
        )

