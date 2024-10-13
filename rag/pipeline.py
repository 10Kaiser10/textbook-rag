from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from pydantic import BaseModel, Field
from typing import Literal

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
    
class ResponseRouteQuery(BaseModel):
    """Route the current response by checking if it satisfactorily answers the question"""
    answer_quality: Literal["satisfactory", "not satisfactory"] = Field(
        ...,
        description="Given a user question and the current response, figure out if the response satisfactorily answers the question",
    )

class ReActRAG:
    def __init__(self, retriever, generator_llm, router_llm, langchain_api_key, num_updates = 3):
        self.init_langsmith(langchain_api_key)

        self.retriever = retriever
        self.generator_llm = generator_llm
        self.router_llm = router_llm

        self.num_updates = num_updates

    def init_langsmith(self, langchain_api_key, langchain_project="textbook-rag", enable_tracing='true', langchain_endpoint="https://api.smith.langchain.com"):
        os.environ["LANGCHAIN_TRACING_V2"] = enable_tracing
        os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = langchain_project

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def update_response(self, question, answer, context):
        prompt = """
            You are an AI agent tasked with updating and refining an existing answer based solely on the provided context.

            Instructions:
            1. You are given a question, an existing answer (which may be incomplete or empty), and context to help complete the answer.
            2. Your job is to **fill in missing information** **using only the information from the context**. Dont change any already present information.
            3. **Do not change the overall structure** of the existing answer. If it is empty, generate a new answer, but only using the context.
            4. **No outside information** or facts are allowed. Only use the context, question, and existing answer.
            5. If the relevant information to answer the question does not exist in the context, simply state: "The information is not present in the documents."
            6. **Do not output any filler lines** such as "Here is the updated answer..." or other extraneous text.
            7. **Follow any and all instructions mentioned in the question**

            Context:
            {2}

            Question:
            {0}

            Existing answer:
            {1}

            If the answer is empty, generate an answer from scratch. If the answer already exists, only update parts of it (dont change the overall structure).
        """.format(question, answer, context)

        return self.generator_llm.invoke(prompt).content
    
    def check_router(self, question, response):
        structured_llm_router = self.router_llm.with_structured_output(ResponseRouteQuery)
        prompt = """
            You are an expert evaluator, tasked with determining whether a response fully and satisfactorily answers a given question. You must evaluate the response **solely based on the information in the question and answer**, without using any outside facts or knowledge.

            You are given a question and its corresponding response. Your task is to critically assess the response and determine whether it meets the following criteria:
            1. **If the response is empty**, automatically classify it as **Not Satisfactory**.
            2. **If the response fails to fully address the question**, classify it as **Not Satisfactory**.
            3. **If the response contains lines mentioning that the information is not present in the context/documents**, classify it as **Not Satisfactory**.
            4. **If the response addresses all aspects of the question**, classify it as **Satisfactory**.

            **Do not use any external information to fact-check the response**. Focus only on whether the response is complete and answers the question based on the information provided.

            Make sure your judgment is based on:
            - **Completeness**: Does the response cover all key parts of the question?

            Please classify the response accordingly.

            Question:
            {0}

            Response:
            {1}
        """.format(question, response)

        return structured_llm_router.invoke(prompt).answer_quality
    
    def get_context(self, question, response):
        prompt = """
            You are tasked with generating 2-4 lines of text that will be used to query (lookup) a vector database to find relevant documents for retrieval-augmented generation (RAG). 

            Important Instructions:
                1. You are given a question and an incomplete or blank answer.
                2. You must **only** use the exact information present in the question and the existing answer. **Do not add any new information**.
                3. If there is insufficient information, focus solely on rephrasing or using keywords already provided.
                4. **Do not speculate**, **assume**, or introduce any new facts, context, or knowledge.
                5. The goal is to create a concise and relevant query that strictly adheres to the given question and answer.
                6. The query should be plain text and should contain no fancy formatting or structure.
                7. If the answer is empty, focus on mentioning the information required to answer the question correctly.
                8. If the answer is present but incomplete, focus on mentioning the missing information.

            Question: {0}
            Existing (Incomplete) Answer: {1}

            Generate only 2-4 lines of text. These lines should mention the information required to answer the question sufficiently.
        """.format(question, response)

        lookup = self.generator_llm.invoke(prompt).content
        print("lookup:\n", lookup)
        documents = self.retriever.invoke(lookup)
        context = self.format_docs(documents)
        return context

    def get_response(self, question):
        n = self.num_updates

        current_response = ""

        while(n>0):
            print('iters left:', n)
            context = self.get_context(question, current_response)
            print('context:\n', context)
            current_response = self.update_response(question, current_response, context)
            print('response:\n', current_response)
            router_out = self.check_router(question, current_response)
            print('router:\n', router_out)

            if router_out == 'satisfactory':
                break

            n -= 1

        return current_response