"""
Basic semantic chunker implementation. Ignores page numbers etc, just considers raw text
Returns a list of N chunks and a corresponding NxM numpy array of embeddings

Based on an implementation by Greg Kamradt https://www.youtube.com/watch?v=8OJC21T2SL4&t=843s
"""

from typing import List, Optional, Union, Coroutine, Any
import asyncio
from enum import Enum
from openai import AsyncOpenAI
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingModels(Enum):
    small = "text-embedding-3-small"
    large = "text-embedding-3-large"

class ThresholdType(Enum):
    percentile = "percentile"

async def process_text(text: str, model: EmbeddingModels) -> Coroutine[Any, Any, Union[List[str], List[np.array]]]:
    sentences = chunk_by_sentece(text)

    sentences = combine_sentences(sentences)

    tasks = [get_embeddings(sentence, model) for sentence in sentences[combine_sentences]]

    embeddings = asyncio.gather(*tasks)
    
    for i, sentence in enumerate(sentences):
        sentence['embedding'] = embeddings[i]

    distances, sentences = calculate_cosine_distances(sentence)

    threshold = calculate_threshold(distances, "percentile", percentile=95)

    idx_above_thresh = [i for i, dist, in enumerate(distances) if dist > threshold]

    # take those indeces and create minibatches of sentence chunks from idx[0] -> idx[1] and so on
    # create tasks [create_chunk for minibatch]
    # async process

def chunk_by_sentece(text: str) -> List[dict]:
    sentence_list = re.split(r'(?<=[.?!])\s+', text)
    sentences = [{'sentence': s, 'index': i} for i, s, in enumerate(sentence_list)]
    return sentences

def combine_sentences(sentences: List[dict], buffer_size: int = 1) -> List[dict]:
    for i in range(len(sentences)):
        combined_sentence = ''

        # Before target sentence
        for j in range(i - buffer_size, i):
            if j >= 0:
                combined_sentence += sentences[j]['sentence']

        combined_sentence += sentences[i]['sentence']

        # After target sentence
        for j in range(i+1, i+1+buffer_size):
            if j < len(sentences):
                combined_sentence += ' ' + sentences[j]['sentence']

        sentences[i]['combined_sentences'] = combined_sentence

    return sentences

async def get_embeddings(text: str, model: EmbeddingModels) -> List[float]:
    aclient = AsyncOpenAI()

    response = await aclient.embeddings.create(
        model=model.value,
        input=text,
    )

    return response.data[0].embeddings

def calculate_cosine_distances(sentences: List[dict]) -> (List[float], List[dict]):
    distances = []

    for i in range(len(sentences) - 1):
        em_current = sentences[i]['embedding']
        em_next = sentences[i+1]['embedding']

        similarity = cosine_similarity([em_current], [em_next])[0][0]

        distance = 1 - similarity

        distances.append(distance)

        sentences[i]['dist_to_next'] = distance

    return distances, sentences

def calculate_threshold(distances: List[float], thresh_type: ThresholdType.name, **kwargs) -> float:
    if thresh_type == ThresholdType.percentile.name:
        percentile = kwargs['percentile']
        threshold = np.percentile(distances, percentile)
    else:
        raise NotImplementedError("Invalid threshold algorithm")
    return threshold

async def create_chunk(mini_batch: List[dict]):
    pass


if __name__ == "__main__":
    file_path = input("Path to PDF: ")
    
