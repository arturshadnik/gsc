from typing import List
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

def create_embeddings(text: List[str]) -> np.ndarray:
    return model.encode(text, convert_to_numpy=True)

def transform_query(text: str) -> str:
    return "Represent this sentence for searching relevant passages: " + text

if __name__ == "__main__":

    docs = [
        "Artur drives a Toyota",
        "Artur works for hcl",
        "Artur likes fish"
    ]
    em_docs = [create_embeddings(d) for d in docs]

    query = transform_query("what does artur drive")
    em_query = create_embeddings(query)

    # cs = cos_sim(em_query, em_docs)
    cs = [cosine_similarity([em_query], [em_d])[0][0] for em_d in em_docs]

    i = cs.index(max(cs))
    print(cs)
    print(i)
