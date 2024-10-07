from groq import Groq
import base64
import pandas as pd
from database_creation.pinecone_upsert import upsert_chunks

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_caption(img_path, img_text, groq_api_key):
    image_path = img_path
    base64_image = encode_image(image_path)

    client = Groq(api_key=groq_api_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", 
                     "text": """
                                Given the image and a one line description, geneate a description for this image in 3-4 lines. 
                                Output only the image description without any fancy formatting. Dont output anything other than the image description.

                                One line description: 
                                {0}
                                """.format(img_text)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="llama-3.2-11b-vision-preview",
    )

    return chat_completion.choices[0].message.content

def upsert_images(pc, index_name, image_list_path, img_path, groq_api_key, embedding_mdl = "multilingual-e5-large"):
    image_list = pd.read_csv(image_list_path)

    id_list = []
    caption_list = []

    for _, row in image_list.iterrows():
        image_id = row['id']
        image_path = img_path + row['path']
        short_desc = row['desc']

        img_caption = short_desc + ". " + get_image_caption(image_path, short_desc, groq_api_key)

        id_list.append(str(image_id))
        caption_list.append(img_caption)

    upsert_chunks(pc, index_name, caption_list, ids=id_list, embedding_mdl=embedding_mdl)