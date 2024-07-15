import torch
from transformers import RobertaTokenizer, RobertaModel
import faiss
import numpy as np
from typing import List, Dict, Tuple
from utils import chunk_code

# Initialize models
model_name = "microsoft/codebert-base"

tokenizer_cache_dir = "cache"
model_cache_dir = "cache"

tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir=tokenizer_cache_dir)
model = RobertaModel.from_pretrained(model_name, cache_dir=model_cache_dir)

# Global variables for chunked data and index
CHUNKED_DATA = None
INDEX = None


def get_embedding(text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()


def get_code_embedding(code: str) -> torch.Tensor:
    return get_embedding(code)


def get_query_embedding(query: str) -> torch.Tensor:
    return get_embedding(query)


def index_chunked_code(code_files: List[str]) -> Tuple[List[Dict], faiss.IndexFlatL2]:
    chunked_data = []
    index = faiss.IndexFlatL2(768)  # CodeBERT embedding dimension

    for file in code_files:
        with open(file, 'r') as f:
            code = f.read()
        chunks = chunk_code(code)
        for i, chunk in enumerate(chunks):
            embedding = get_code_embedding(chunk)
            index.add(np.array([embedding]))
            chunked_data.append({
                "file": file,
                "chunk_id": i,
                "content": chunk
            })

    return chunked_data, index


def perform_code_rag_search(query: str, chunked_data: List[Dict], index, top_k: int = 5) -> List[Dict]:
    query_embedding = get_query_embedding(query).numpy()
    distances, indices = index.search(np.array([query_embedding]), top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        chunk_info = chunked_data[idx]
        similarity = 1 / (1 + distances[0][i])  # Convert distance to similarity
        results.append({
            "file": chunk_info["file"],
            "chunk_id": chunk_info["chunk_id"],
            "content": chunk_info["content"],
            "similarity": float(similarity)
        })

    return sorted(results, key=lambda x: x['similarity'], reverse=True)


def initialize_rag_system(code_files: List[str]):
    global CHUNKED_DATA, INDEX
    CHUNKED_DATA, INDEX = index_chunked_code(code_files)


def code_rag_search(query: str, top_k: int = 5) -> List[Dict]:
    global CHUNKED_DATA, INDEX
    return perform_code_rag_search(query, CHUNKED_DATA, INDEX, top_k)
