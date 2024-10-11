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
agent = Agent(GROQ_API_KEY, LANGCHAIN_API_KEY, retriever, img_retr, model="mixtral-8x7b-32768")


####### FastAPI Code ########

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

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
        reply, img_data = agent.query(query)
        if img_data[0] is None:
            img_data = ("", 0)
        return AgentQueryResponse(query=query, answer=reply, img_path=img_data[0], img_scr=img_data[1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the server locally (with uvicorn)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
