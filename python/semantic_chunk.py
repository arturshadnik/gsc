"""
Basic semantic chunker implementation. Ignores page numbers etc, just considers raw text
Returns a list of N chunks and a corresponding NxM numpy array of embeddings
"""

from typing import List, Optional, Union, Coroutine, Any
import asyncio
from enum import Enum
from openai import AsyncOpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingModels(Enum):
    small = "text-embedding-3-small"
    large = "text-embedding-3-large"


async def process_text(text: str, model: EmbeddingModels) -> Coroutine[Any, Any, Union[List[str], List[np.array]]]:
    sentences = chunk_by_sentece(text)

    tasks = [get_embeddings(sentence, model) for sentence in sentences]

    sentence_embeddings = asyncio.gather(*tasks)

    for i, sentence in enumerate(sentence_embeddings):
        pass



def chunk_by_sentece(text: str) -> List[str]:
    return text.split(". ")

async def get_embeddings(text: str, model: EmbeddingModels) -> List[float]:
    aclient = AsyncOpenAI()

    response = await aclient.embeddings.create(
        model=model.value,
        input=text,
    )

    return response.data[0].embeddings

if __name__ == "__main__":
    file = input("Path to PDF: ")
