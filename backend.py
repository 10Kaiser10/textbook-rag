###### RAG code here ########
from pinecone import Pinecone
import pickle
from rag.pinecone_query import PineconeRetriever
from rag.pipeline import RAG_Piepline
from config import PINECONE_API_KEY, GROQ_API_KEY, LANGCHAIN_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)
text_index_name = "textbook-rag"
mapping_path = "data/chunks/mapping.pickle"
embedding_mdl = "multilingual-e5-large"
agent_mdl = "llama-3.2-90b-vision-preview"
k = 3

with open(mapping_path, 'rb') as f:
    mapping = pickle.load(f)

retriever = PineconeRetriever(pc=pc, index_name=text_index_name, embedding_mdl=embedding_mdl, k=k, mapping=mapping)
rag = RAG_Piepline(retriever, GROQ_API_KEY, LANGCHAIN_API_KEY)


###### Agent Code here #######

from agent.agent import Agent
from rag.pinecone_query import PineconeImageRetriever

image_index_name = "textbook-rag-images"
img_k = 1
img_fldr = "data/img_data/imgs/"
mapping_csv_path = "data/img_data/index.txt"

img_retr = PineconeImageRetriever(pc, image_index_name, img_fldr, mapping_csv_path)
agent = Agent(GROQ_API_KEY, LANGCHAIN_API_KEY, retriever, img_retr, model=agent_mdl)


####### FastAPI Code ########

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import requests
from config import SARVAM_API_KEY
import json

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify your frontend URL here instead of "*" for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request body model
class QueryRequest(BaseModel):
    query: str

# Define the response model (optional)
class RagQueryResponse(BaseModel):
    query: str
    answer: str

class AgentQueryResponse(BaseModel):
    query: str
    answer: str
    img_path: str
    img_scr: float

class TTSRequest(BaseModel):
    text: str

class AgentTTSResponse(BaseModel):
    audios: list[str]

# Create a POST route to handle the query and generate a response
@app.post("/ask_rag", response_model=RagQueryResponse)
async def ask_question(request: QueryRequest):
    query = request.query
    try:
        # Call the RAG function with the query
        answer = rag.get_response(query)
        return RagQueryResponse(query=query, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Create a POST route to handle the query and generate a response
@app.post("/ask_agent", response_model=AgentQueryResponse)
async def ask_agent(request: QueryRequest):
    query = request.query
    try:
        # Call the RAG function with the query
        print(query)
        reply, img_data = agent.query(query)
        print(reply)
        if img_data[0] is None:
            img_data = ("", 0)
        return AgentQueryResponse(query=query, answer=reply, img_path=img_data[0], img_scr=img_data[1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def split_string(text):
    text_splits = text.split('.')
    text_chunks = []

    for split in text_splits:
        for i in range(0, len(split), 500):
            text_chunks.append(split[i:i+500])

    return text_chunks

def get_voice(chunks):
    voice_outs = []

    for i in range(0, len(chunks), 3):
        try:
            url = "https://api.sarvam.ai/text-to-speech"
            payload = {
                "inputs": chunks[i:i+3],
                "target_language_code": "en-IN",
                "speaker": "meera",
                "pitch": 0,
                "pace": 1,
                "loudness": 1.5,
                "speech_sample_rate": 8000,
                "enable_preprocessing": True,
                "model": "bulbul:v1"
            }
            headers = {"Content-Type": "application/json", "api-subscription-key": SARVAM_API_KEY}
            response = requests.request("POST", url, json=payload, headers=headers)
            voice_audio = json.loads(response.text)['audios']
            voice_outs += voice_audio
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return voice_outs

#route for text to speech
@app.post("/tts", response_model=AgentTTSResponse)
async def tts(request: TTSRequest):
    text_chunks = split_string(request.text)
    audios = get_voice(text_chunks)
    return AgentTTSResponse(audios=audios)

# To run the server locally (with uvicorn)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
