start_text="""
########################################################################################################
#         _____                            _______                                 ___      __         #
#        |  __ \                          |__   __|                               / _ \    /_ |        #
#        | |  | |_ __ ___  __ _ _ __ ___     | | ___  __ _ _ __ __ _   __   __   | | | |    | |        #
#        | |  | | '__/ _ \/ _` | '_ ` _ \    | |/ _ \/ _` | '_ ` _  |  \ \ / /   | | | |    | |        #
#        | |__| | | |  __/ (_| | | | | | |   | |  __/ (_| | | | | | |   \ V /    | |_| |    | |        #
#        |_____/|_|  \___|\__,_|_| |_| |_|   |_|\___|\__,_|_| |_| |_|    \_/      \___/ (_) |_|        #
#                                                                                                      #
########################################################################################################
"""


def get_page_style():
    return """
        <style>
            header {visibility: hidden;}
            .streamlit-footer {display: none;}
            .st-emotion-cache-uf99v8 {display: none;}
        </style>
    """

def get_header():
    return """
        <div style="display:flex; flex-direction: column; justify-content: flex-start; align-items: center; width: 100%; margin-top:-100px;">
            <h1>DREAM TEAM</h1>
        </div>
    """

def preview_container_template():
    return """
        <div style="display:flex; flex-direction: column; justify-content: flex-start; border-radius: 5px; align-items: center; width: 100%; height: 600px; box-shadow: rgba(100, 100, 111, 0.2) 0px 7px 29px 0px;">
            <div style="display:flex; align-items: start; width: 100%;">
                <div style="width: 40%;">
                    <div style="width: 100%;">
                        <div style="margin: 30px; width: 100px; height: 50px;">
                            <img src="data:image/png;base64,{image_flag}" 
                                 style="width: 100%; height: 100%; object-fit: contain;"/>
                        </div>
                    </div>
                </div>
                <h2>Ön İzleme</h2>
            </div>
            <div style="width: 90%; height: 600px; overflow: hidden;">
                <img src="data:image/png;base64,{image_base64}" style="width: 100%; height: 100%; object-fit: contain;"/>
            </div>
           <h5 style="margin:30px;">{description}</h5> 
        </div>
    """

def output_view_template():
    return """
        <div style="display:flex; flex-direction: column; justify-content: flex-start; border-radius: 5px; align-items: center; width: 100%; height: 600px; box-shadow: rgba(100, 100, 111, 0.2) 0px 7px 29px 0px;">
            <div style="display:flex; align-items: start; width: 100%;">
                <div style="width: 40%;">
                    <div style="width: 100%;">
                        <div style="margin: 30px;width: 100px; height: 50px;">
                            <img src="data:image/png;base64,{flag_base64}" style="width: 100%; height: 100%; object-fit: contain;"/>
                        </div>
                    </div>
                </div>
                <h2>Çıktı</h2>
            </div>
            <h4>{urun_adi}</h4>
            <div style="width: 90%; height: 600px; overflow: hidden;">
                <img src="data:image/png;base64,{base64_image}" style="width: 100%; height: 100%; object-fit: contain;"/>
            </div>
            <h5 style="margin:30px;">{urun_aciklama}</h5>    
        </div>
    """

def get_additional_styles():
    return """
        <style>
            div[data-testid="stVerticalBlockBorderWrapper"] {
                display: flex; 
                justify-content: space-between; 
                align-items: center; 
                width: 100%; 
                height: 600px;
                margin-bottom: 30px;
            }
            div[data-testid="stHorizontalBlock"] {
                gap: 2rem; 
            }
        </style>
    """