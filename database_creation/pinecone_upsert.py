def upsert_chunks(pc, index_name, chunks, embedding_mdl = "multilingual-e5-large"):
    index = pc.Index(index_name)

    embeddings = pc.inference.embed(
        model= embedding_mdl,
        inputs= chunks,
        parameters= {"input_type": "passage", "truncate": "NONE"}
    )

    vectors = []
    for idx, emb in enumerate(embeddings):
        vectors.append({
            "id": str(idx),
            "values": emb['values'],
        })

    response = index.upsert(
        vectors=vectors
    )

    return response
