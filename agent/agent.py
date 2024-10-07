import os
from langchain_groq import ChatGroq
from data.summaries.sound_summary import SOUND_SUMMARY
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from rag.pipeline import RAG_Piepline
from langchain_core.runnables import Runnable
from operator import itemgetter

def generate_response(ques):
    return "response"

@tool(parse_docstring=True)
def rag_tool(question: str) -> str:
    """
    Generating responses about topics related to:
    Sound from a science textbook, explaining the basic principles of sound production, propagation, and characteristics. 
    It covers topics such as mechanical and longitudinal waves, frequency, amplitude, pitch, and the speed of sound in various media. 
    Key concepts like reflection of sound, echoes, reverberation, and the human auditory range are explored. 
    The text also discusses practical applications of sound, including ultrasound in medical imaging and sonar technology. 
    Exercises and examples help reinforce the understanding of sound waves and their properties.

    Args:
        question: Message for which response is to be generated.

    Returns:
        Generated response.
    """

    return generate_response(question)

def get_summary(inp):
        return SOUND_SUMMARY

class Agent:
    def __init__(
            self, groq_api_key, langchain_api_key, 
            rag_retriever,
            model='llama-3.1-8b-instant',
            langchain_project="textbook-rag", enable_tracing='true', langchain_endpoint="https://api.smith.langchain.com"
            ):
        
        self.init_langsmith(langchain_api_key=langchain_api_key, langchain_project=langchain_project, enable_tracing=enable_tracing, langchain_endpoint=langchain_endpoint)

        self.rag = RAG_Piepline(rag_retriever, groq_api_key, langchain_api_key)
        self.llm = ChatGroq(model=model, temperature=0.5, max_retries=2, api_key=groq_api_key)
        self.llm_rag_router = ChatGroq(model=model, temperature=0.5, max_retries=2, api_key=groq_api_key)
        #self.llm_rag_router = self.llm_rag_router.bind_tools([rag_tool])
        
    def init_langsmith(self, langchain_api_key, langchain_project="textbook-rag", enable_tracing='true', langchain_endpoint="https://api.smith.langchain.com"):
        os.environ["LANGCHAIN_TRACING_V2"] = enable_tracing
        os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = langchain_project

    def rag_router_call(self, query, llm):
        # template = """
        #     You are an expert in determining whether a incoming message is related to textbook or not.
        #     The summary, structure and topics in the textbook are mentioned below. You are to determine if the textbook will be helpfull in answering the incoming message.
        #     If the message relates to the textbook, output "1" else "0". Only output either of these two numbers and nothing else.

        #     Summary, structure and topics in the textbook:
        #     {topics}

        #     Incoming Message:
        #     {message}
        # """

        template = """
            You have access to a summary of a textbook. This summary contains the following:
                - A short summary of the textbook.
                - A detailed structure of the textbook with all the chapters, topics and subtopics.
                - A list of keywords, names, and concepts covered in the textbook.

            You are given a query. Your task is to decide whether the query should be answered by looking up context from the textbook (RAG), or if it can be answered without using the textbook.
                - If the query requires context from the textbook (i.e., it relates to the topics, subtopics, or keywords covered in the textbook), return "1".
                - If the query can be answered without referring to the textbook, return "0".

            Here is the list of topics and keywords from the textbook:
            {topics}

            Query: {message}

            Respond with only "1" or "0" based on your decision. Dont output anything else.
        """

        prompt_template = PromptTemplate.from_template(template)

        inputs = {"topics": SOUND_SUMMARY, "message": query}

        chain = (
            {"topics": itemgetter("topics"), "message": itemgetter("message")}
            | prompt_template
            | llm
        )

        return chain.invoke(inputs)

    def query(self, text_input):
        rag_router_response = self.rag_router_call(text_input, self.llm_rag_router)

        print(rag_router_response.content)
        print(rag_router_response.tool_calls)

        #if len(rag_router_response.tool_calls) > 0:
        if rag_router_response.content.lower() == "1":
            response = self.rag.get_response(text_input)
        else:
            response = self.llm.invoke(text_input).content

        return response