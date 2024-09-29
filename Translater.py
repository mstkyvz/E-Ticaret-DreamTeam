import openai

client = openai.OpenAI(
    base_url="http://localhost:8001/v1", 
    api_key = "-"
)


def llm(text,lang1,lang2):
    system_prompt="""
Translate the given text from """+lang1+""" language to """+lang2+""" language. While translating, please adhere to the following rules:

1. Return the output in JSON format.
2. Do not add anything extra to the output.
3. Do not deviate from the content of the input text.
4. Do not translate the JSON keys "product_name" and "product_detailed_description".
5. Translate carefully
6. When translating, make exact matching translations. Be careful

Example Input:

{
  "product_name": "RTX 4090 Ekran kart",
  "product_detailed_description": "RTX 4090, NVIDIA tarafından üretilen bir video oyunu kartıdır. Yüksek kaliteli bir performans sunar ve oyunlar için mükemmel bir seçenektir."
}

Example Output:  
{
  "product_name": "Tarjeta de vídeo RTX 4090",
  "product_detailed_description": "RTX 4090 es una tarjeta de videojuegos fabricada por NVIDIA. Ofrece un rendimiento de alta calidad y es una gran opción para jugar."
}
"""
    text="Input:\n"+str(text)
    
    completion = client.chat.completions.create(
    model="DreamTeam",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    )

    return completion.choices[0].message.content