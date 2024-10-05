import pickle

def create_chunk_map(ids, chunks, map_path):
    mapping = dict(zip(ids, chunks))

    with open(map_path, 'wb') as f:
        pickle.dump(mapping, f, pickle.HIGHEST_PROTOCOL)

    return mapping


def upsert_chunks(pc, index_name, chunks, map_path, embedding_mdl = "multilingual-e5-large"):
    index = pc.Index(index_name)

    embeddings = pc.inference.embed(
        model= embedding_mdl,
        inputs= chunks,
        parameters= {"input_type": "passage", "truncate": "NONE"}
    )

    ids = [str(n) for n in range(len(chunks))]

    vectors = []
    for idx, emb in enumerate(embeddings):
        vectors.append({
            "id": ids[idx],
            "values": emb['values'],
        })

    response = index.upsert(
        vectors=vectors
    )

    create_chunk_map(ids, chunks, map_path)

    return response