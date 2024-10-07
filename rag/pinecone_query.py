import pickle
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pinecone import Pinecone

class PineconeRetriever(BaseRetriever):
    pc:Pinecone
    index_name:str
    embedding_mdl:str
    k:int
    mapping:dict

    def pinecone_query(self, text, pc, index_name, embedding_mdl = "multilingual-e5-large", k=3):
        index = pc.Index(index_name)

        embedding = pc.inference.embed(
            model= embedding_mdl,
            inputs= [text,],
            parameters= {"input_type": "query", "truncate": "NONE"}
        )

        result = index.query(
            vector=embedding[0].values,
            top_k=k,
            include_values=False,
            include_metadata=True
        )

        return result

    def query_text(self, text, mapping, pc, index_name, embedding_mdl = "multilingual-e5-large", k=3):
        result = self.pinecone_query(text, pc, index_name, embedding_mdl, k)

        output_documents = []

        for match in result['matches']:
            match_id = match['id']
            doc = Document(page_content=mapping[match_id])
            output_documents.append(doc)

        return output_documents
    
    def _get_relevant_documents(self, query, *, run_manager):
        return self.query_text(query, self.mapping, self.pc, self.index_name, self.embedding_mdl, self.k)