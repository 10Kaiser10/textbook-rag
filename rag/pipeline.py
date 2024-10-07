from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os

class RAG_Piepline:
    def __init__(self, retriever, groq_api_key, langchain_api_key, model='llama-3.1-8b-instant', langchain_project="textbook-rag", custom_template=None, enable_tracing='true', langchain_endpoint="https://api.smith.langchain.com"):
        os.environ["LANGCHAIN_TRACING_V2"] = enable_tracing
        os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = langchain_project

        if custom_template is None:
            template = """
            You are a helpful AI agent that answers questions based on provided context. Use only the information
            present in the context for answering. If relevant information to answer the question does not exist in the context,
            just say that the information is not present in the doucments. Keep the answers short and concise.

            Context:
            {context}

            Question: {question}
            """
        else:
            template = custom_template

        prompt_template = PromptTemplate.from_template(template)

        llm = ChatGroq(model=model, temperature=0.5, max_retries=2, api_key=groq_api_key)

        self.rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_response(self, query):
        reply = self.rag_chain.invoke(query)
        return reply