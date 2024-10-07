from transformers import AutoTokenizer
from pinecone import Pinecone, ServerlessSpec

from database_creation.pdf_extractor import extract_and_save_text
from database_creation.chunking_logic import chunk_texts_from_folder
from database_creation.pinecone_upsert import upsert_chunks
from config import PINECONE_API_KEY, GROQ_API_KEY, LANGCHAIN_API_KEY
from database_creation.image_data import upsert_images

pdf_folder = 'data/pdfs/'
text_folder = 'data/extracted_text/'
chunk_map_path = 'data/chunks/mapping.pickle'
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
tokens_per_chunk = 500
chunk_overlap = 0.2
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "textbook-rag"
image_list_path = "data/img_data/index.txt"
image_path = "data/img_data/imgs/"



chunks = chunk_texts_from_folder(tokenizer, text_folder, tokens_per_chunk, chunk_overlap)
print("Total Chunks:", len(chunks))

response = upsert_chunks(pc, index_name, chunks, chunk_map_path)
print("Upserted Chunks:", response.upsertedCount)

respone = upsert_images(pc, "textbook-rag-images", image_list_path, image_path, GROQ_API_KEY)
print("Upserted Imaged")