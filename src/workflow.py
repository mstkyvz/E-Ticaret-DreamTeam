import websocket
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
from io import BytesIO
import io
import base64


def generate_image(prompt, file_path, server_address="127.0.0.1:8188"):
    client_id = str(uuid.uuid4())
    
    with open("./src/workflow.json", "r") as f:
        prompt_text = json.load(f)
    # Update the prompt_text dictionary
    prompt_text['89']['inputs']['text'] = prompt
    prompt_text['63']['inputs']['image'] = file_path
    def queue_prompt(prompt):
        p = {"prompt": prompt, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{server_address}/view?{url_values}") as response:
            return response.read()

    def get_history(prompt_id):
        with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def get_images(ws, prompt):
        prompt_id = queue_prompt(prompt)['prompt_id']
        output_images = {}
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break  # Execution is done
            else:
                continue  # previews are binary data
        history = get_history(prompt_id)[prompt_id]
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            images_output = []
            if 'images' in node_output:
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(Image.open(BytesIO(image_data)))
            output_images[node_id] = images_output
        return output_images

    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
    images = get_images(ws, prompt_text)
    ws.close()
    return images['115'][0]



# def generate_image(prompt, image, server_address="127.0.0.1:8188"):
#     client_id = str(uuid.uuid4())
    
#     with open("./src/workflow.json", "r") as f:
#         prompt_text = json.load(f)
    
#     image = Image.open(image)
#     buffered = BytesIO()
#     image.save(buffered, format="PNG")

#     file_path = f"/home/jupyter/temp/{client_id}.png"
#     image.save(file_path, format="PNG")

#     # Update the prompt_text dictionary
#     prompt_text['89']['inputs']['text'] = prompt
#     prompt_text['63']['inputs']['image'] = file_path
#     def queue_prompt(prompt):
#         p = {"prompt": prompt, "client_id": client_id}
#         data = json.dumps(p).encode('utf-8')
#         req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
#         return json.loads(urllib.request.urlopen(req).read())

#     def get_image(filename, subfolder, folder_type):
#         data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
#         url_values = urllib.parse.urlencode(data)
#         with urllib.request.urlopen(f"http://{server_address}/view?{url_values}") as response:
#             return response.read()

#     def get_history(prompt_id):
#         with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
#             return json.loads(response.read())

#     def get_images(ws, prompt):
#         prompt_id = queue_prompt(prompt)['prompt_id']
#         output_images = {}
#         while True:
#             out = ws.recv()
#             if isinstance(out, str):
#                 message = json.loads(out)
#                 if message['type'] == 'executing':
#                     data = message['data']
#                     if data['node'] is None and data['prompt_id'] == prompt_id:
#                         break  # Execution is done
#             else:
#                 continue  # previews are binary data
#         history = get_history(prompt_id)[prompt_id]
#         for node_id in history['outputs']:
#             node_output = history['outputs'][node_id]
#             images_output = []
#             if 'images' in node_output:
#                 for image in node_output['images']:
#                     image_data = get_image(image['filename'], image['subfolder'], image['type'])
#                     images_output.append(Image.open(BytesIO(image_data)))
#             output_images[node_id] = images_output
#         return output_images

#     ws = websocket.WebSocket()
#     ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
#     images = get_images(ws, prompt_text)
#     ws.close()
#     return images['115'][0]