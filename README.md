
# QueryWise: An AI-Powered RAG System for TextBook querying

**QueryWise** is an AI-powered tool designed to answer queries related to sound waves from the NCERT textbook. It uses Retrieval-Augmented Generation (RAG) to fetch relevant information from a vectorstore and intelligently decides whether to answer queries using the textbook context or directly from an LLM (Language Learning Model).

  

[Demo Link](https://textbook-rag-frontend.onrender.com)

  

##  **Features**:

-  **RAG-Driven Responses**: Answers complex questions by looking up the NCERT textbook summary.

-  **Smart Query Handling**: Routes simple or unrelated queries to the LLM directly.

-  **Context Aware Image Retrieval**: Retrieves related images from the textbook and add image context to the answer.

-  **Voice Response:** Answer query with voice output.

  

##  **How it Works**

![Pipeline](https://github.com/10Kaiser10/textbook-rag/blob/main/frontend/media/pipeline.jpg)

  

####  **Text Retrieval:**

1. The system uses an LLM with structured outputs to decide whether a query should be handled by the RAG vectorstore or directly by the LLM.

2. When relevant, a RAG pipeline queries document chunks from a Pinecone vectorstore and generates answers using the retrieved context.

  

####  **Image Retrieval:**

1. For image-related queries, the system sends the image and its associated caption from the book to a multi-modal LLM to generate detailed image descriptions.

2. Text-based queries are matched against these image descriptions using Pinecone, retrieving the most relevant image and modifying the answer to include image context.

  

##  **Technologies Used**:

  

####  **RAG Pipeline:**

-  **LangChain**: Manages agent logic and query tracing with LangSmith.

-  **Pinecone**: Provides a fast and simple vectorstore for document and image retrieval.

-  **Groq**: Utilized for free LLM access.

-  **Llama 3.2 11B Vision**: Generates image captions.

-  **Llama 3 70B**: Handles LLM text responses.

-  **pymupdf**: Used for parsing PDF content.

-  **sarvam**: To convert text responses to speech.

  

####  **Backend and Frontend:**

-  **FastAPI**: Serves as the backend to handle API requests.

-  **Uvicorn**: ASGI server for running the FastAPI app.

-  **Render**: For deploying both the frontend and backend.