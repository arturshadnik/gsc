"""
Basic semantic chunker implementation. Ignores page numbers etc, just considers raw text
Returns a list of N chunks and a corresponding NxM numpy array of embeddings

Based on an implementation by Greg Kamradt https://www.youtube.com/watch?v=8OJC21T2SL4&t=843s
"""

from dotenv import load_dotenv
load_dotenv()
from typing import List, Tuple, Coroutine, Any
import asyncio
from openai import AsyncOpenAI
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

async def process_text(text: str, model: str, exit_on_fail: bool = False) -> Coroutine[Any, Any, Tuple[List[str], List[List[float]]]]:
    try:
        sentences = chunk_by_sentece(text)
        sentences = combine_sentences(sentences)

        print("creating embeddings of sentences")
        tasks = [get_embeddings(sentence['combined_sentences'], model) for sentence in sentences]

        embeddings = await asyncio.gather(*tasks, return_exceptions=True)
        if exit_on_fail:
            if any(isinstance(em, Exception) for em in embeddings):
                raise Exception("One or more sub-tasks failed")
        else:
            embeddings = [em for em in embeddings if not isinstance(em, Exception)]

        for i, s in enumerate(sentences):
            s['embedding'] = embeddings[i]

        print("calculating distances")
        distances, sentences = calculate_cosine_distances(sentences)
        threshold = calculate_threshold(distances, "percentile", percentile=80)

        idx_above_thresh = [i for i, dist, in enumerate(distances) if dist > threshold]

        chunks = create_final_chunks(sentences, idx_above_thresh)
        tasks = [get_embeddings(chunk, model) for chunk in chunks]
        embedded_chunks = await asyncio.gather(*tasks)

        return chunks, embedded_chunks
    except Exception as e:
        print("something went wrong", e)
        return [], []
    

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

async def get_embeddings(text: str, model: str) -> List[float]:
    aclient = AsyncOpenAI()

    try:
        response = await aclient.embeddings.create(
            model=model,
            input=text,
        )

        return response.data[0].embedding
    except Exception as e:
        print("Error getting embeddings", e)
        raise

def calculate_cosine_distances(sentences: List[dict]) -> Tuple[List[float], List[dict]]:
    distances = []
    try:
        for i in range(len(sentences) - 1):
            em_current = sentences[i]['embedding']
            em_next = sentences[i+1]['embedding']

            similarity = cosine_similarity([em_current], [em_next])[0][0]

            distance = 1 - similarity
            distances.append(distance)

            sentences[i]['dist_to_next'] = distance

    except Exception as e:
        print(e)
    return distances, sentences

def calculate_threshold(distances: List[float], thresh_type: str, **kwargs) -> float:
    if thresh_type == "percentile":
        percentile = kwargs['percentile']
        threshold = np.percentile(distances, percentile)
    else:
        raise NotImplementedError("Invalid threshold algorithm")
    return threshold

def create_final_chunks(sentences: List[dict], idx: List[int]) -> List[str]:
    idx.insert(0, 0)
    chunks = []

    mini_batches = [sentences[idx[i]:idx[i+1]] for i in range(len(idx)-1)]
    mini_batches.append(sentences[idx[-1]:])

    for mb in mini_batches:
        chunk = ' '.join([s['sentence'] for s in mb])
        chunks.append(chunk)
    
    return chunks


if __name__ == "__main__":
    file_path = input("enter a path to a pdf: ")
    
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        text = ''

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            text += page_text

        print("successfully parsed pdf")
    chunks, embeddings = asyncio.run(process_text(text, "text-embedding-3-small", False))

    

