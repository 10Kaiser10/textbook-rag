import pickle

def pinecone_query(text, pc, index_name, embedding_mdl = "multilingual-e5-large", k=3):
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

def query_text(text, mapping_path, pc, index_name, embedding_mdl = "multilingual-e5-large", k=3):
    result = pinecone_query(text, pc, index_name, embedding_mdl, k)

    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)

    output_chunks = []

    for match in result['matches']:
        match_id = match['id']
        output_chunks.append(mapping[match_id])

    return output_chunks