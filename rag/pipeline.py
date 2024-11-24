from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from pydantic import BaseModel, Field
from typing import Literal
from collections import Counter
from langchain_core.documents import Document

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
    answer_quality: Literal["Satisfactory", "Not Satisfactory"] = Field(
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

            ### Instructions:
            1. **Inputs:**
            - A user question.
            - An existing answer (which may be incomplete or empty).
            - A set of context documents containing relevant information.

            2. **Task:**
            - If the existing answer is empty, generate a new answer **entirely from the context provided**.
            - If the existing answer is incomplete, fill in the missing information **using only the context** without altering the existing structure or already present information.

            3. **Strict Rules:**
            - The context provided below is the only source of truth.
            - If the necessary information to answer a part of the question (or the full question) is not present in the context, respond by saying that this part is not answerable based on the context.
            - Do not add any unnecessary preambles, explanations, or filler text like "Here is the updated answer..."
            - The goal is to answer the question sufficiently. Dont add unneccessary details or information which is not relevant to the question.

            4. **Formatting:**
            - Maintain the same structure as the existing answer. If it’s empty, create a new structured answer from scratch.
            - Keep your response concise and directly relevant to the question.

            ---

            ### Inputs:
            - **Context:**  
            {2}

            - **Question:**  
            {0}

            - **Existing Answer:**  
            {1}

            ---

            Update or generate the answer based on the above inputs.

        """.format(question, answer, context)

        return self.generator_llm.invoke(prompt).content
    
    def check_router(self, question, response):
        structured_llm_router = self.router_llm.with_structured_output(ResponseRouteQuery)
        prompt = """
            You are an expert evaluator, tasked with determining whether a response fully and satisfactorily answers a given question. You must evaluate the response **solely based on the information in the question and answer**, without using any outside facts or knowledge.

            You are given a question and its corresponding response. Your task is to critically assess the response and determine whether it meets the following criteria:
            - **If the response is empty**, automatically classify it as **Not Satisfactory**.
            - **If the response fails to fully address the question**, classify it as **Not Satisfactory**.
            - **If the response contains lines mentioning that the information is not present in the context/documents**, classify it as **Not Satisfactory**.
            - **If the response contains lines mentioning "Not answerable based on the context provided"**, classify it as **Not Satisfactory**.
            - **If the response addresses all aspects of the question**, classify it as **Satisfactory**.

            **Do not use any external information to fact-check the response**. Focus only on whether the response is complete and answers the question based on the information provided.

            Question:
            {0}

            Response:
            {1}
        """.format(question, response)

        return structured_llm_router.invoke(prompt).answer_quality
    
    def vectorstore_lookup(self, lookup_query):
        queries = lookup_query.split('\n')
        print(queries)

        all_documents = []

        for query in queries:
            if not query.strip():
                continue
            
            print(query)
            documents = self.retriever.invoke(query)
            
            all_documents += [x.page_content for x in documents]
        
        freq = Counter(all_documents)
        filtered_docs = [Document(page_content=num) for num, _ in freq.most_common(3)]

        print("all docs\n", all_documents)
        print("filtered docs\n", filtered_docs)

        return filtered_docs


    def get_context(self, question, response):
        prompt = """
            You are tasked with generating **three distinct queries** that will be used to query a vector database to retrieve relevant chunks of a document for answering a user's question. The goal is to find context that fills any gaps in an incomplete answer or addresses the question fully if the current answer is empty.

            ### Important Instructions:
            1. **Input Information:**
            - You are given a question and an existing answer, which may be empty or incomplete.
            2. **Query Generation Goals:**
            - Each query must focus on retrieving information to address the question or improve the answer.
            - If the answer is empty, focus on extracting information to answer the question entirely.
            - If the answer is incomplete, focus on retrieving information to fill the gaps in the existing answer.
            - Each query should address the information that is currently missing from answer.
            3. **Guidelines for Query Creation:**
            - Use only the information provided in the question and the answer. **Do not add any new information or assumptions.**
            - Rephrase or extract keywords to form concise and specific queries.
            - Ensure that the three queries are distinct and explore different aspects or keywords from the given inputs.
            - Avoid redundancy between the queries.
            4. **Format:**
            - Generate exactly three lines seperate by newline character
            - Each line should contain one query.
            - Keep each query concise and specific, ideally in 1-4 sentences.
            - Avoid any formatting other than plain text.
            - Seperate the queries using newline character to make it easy to split them. Dont add any empty lines between two lines.
            - Dont add any filler texts like "here are three distinct queries to retrieve relevant context"

            ### Inputs:
            - **Question:** {0}
            - **Existing (Incomplete) Answer:** {1}

            ### Task:
            Generate **three distinct queries** based on the provided inputs. Each query should be focused on retrieving relevant context to answer the question sufficiently.
        """.format(question, response)

        lookup = self.generator_llm.invoke(prompt).content
        print("lookup:\n", lookup)
        #documents = self.retriever.invoke(lookup)
        documents = self.vectorstore_lookup(lookup)
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

            if router_out == 'Satisfactory':
                break

            n -= 1

        return current_response