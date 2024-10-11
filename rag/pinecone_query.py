import pickle
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pinecone import Pinecone
import pandas as pd

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
    
class PineconeImageRetriever():
    def __init__(self, pc, index_name, image_fldr, mapping_csv_path, embedding_mdl = "multilingual-e5-large", k = 1):
        self.pc = pc
        self.index_name = index_name
        self.embedding_mdl = embedding_mdl
        self.k = k
        self.image_fldr = image_fldr

        mapping_csv = pd.read_csv(mapping_csv_path)
        self.path_mapping = dict(zip(mapping_csv['id'].astype(str).to_list(), mapping_csv['path'].to_list()))
        self.desc_mapping = dict(zip(mapping_csv['id'].astype(str).to_list(), mapping_csv['desc'].to_list()))

    def pinecone_query(self, text, pc, index_name, embedding_mdl = "multilingual-e5-large", k=1):
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
    
    def get_relevant_image(self, query):
        result = self.pinecone_query(query, self.pc, self.index_name, self.embedding_mdl, self.k)

        output_list = []

        for match in result['matches']:
            match_id = match['id']
            img_desc = self.desc_mapping[match_id]
            img_path = self.image_fldr + self.path_mapping[match_id]
            img_scr = match['score']

            output_list.append((img_path, img_desc, img_scr))

        return output_list