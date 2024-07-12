import autogen
from typing import List, Dict, Tuple
import glob
import torch
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize models
code_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
code_model = RobertaModel.from_pretrained("microsoft/codebert-base")
query_model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper functions
def get_code_embedding(code_snippet: str) -> np.ndarray:
    inputs = code_tokenizer(code_snippet, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = code_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def get_query_embedding(query: str) -> np.ndarray:
    return query_model.encode([query])[0]

def chunk_code(code: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    lines = code.split('\n')
    chunks = []
    for i in range(0, len(lines), chunk_size - overlap):
        chunk = '\n'.join(lines[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

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

def code_rag_search(query: str, chunked_data: List[Dict], index: faiss.IndexFlatL2, top_k: int = 5) -> List[Dict]:
    query_embedding = get_query_embedding(query)
    _, I = index.search(np.array([query_embedding]), top_k)
    
    results = []
    for idx in I[0]:
        chunk_info = chunked_data[idx]
        similarity = float(faiss.vector_to_array(index.reconstruct(idx)).dot(query_embedding))
        results.append({
            "file": chunk_info["file"],
            "chunk_id": chunk_info["chunk_id"],
            "content": chunk_info["content"],
            "similarity": similarity
        })
    
    return sorted(results, key=lambda x: x['similarity'], reverse=True)

# Global variables for chunked data and index
CHUNKED_DATA = None
INDEX = None

# Function to initialize the RAG system
def initialize_rag_system(code_files: List[str]):
    global CHUNKED_DATA, INDEX
    CHUNKED_DATA, INDEX = index_chunked_code(code_files)

# Agent definitions
user_proxy = autogen.UserProxyAgent(
    name="Human",
    system_message="A human user interacting with the AI project analysis system.",
    code_execution_config={"work_dir": "coding"}
)

assistant = autogen.AssistantAgent(
    name="Assistant",
    system_message="""You are an AI assistant helping users analyze an AI project by using code RAG to find relevant code snippets.
    Use the code_rag_search function to find relevant code and provide insights based on the retrieved snippets.""",
    llm_config={"config_list": [{"model": "gpt-4"}]}
)

code_analyst = autogen.AssistantAgent(
    name="CodeAnalyst",
    system_message="""You are an expert in code analysis, specializing in AI and machine learning systems. 
    Analyze code snippets retrieved by the RAG system and provide insights on data preprocessing, model architecture, and evaluation metrics.""",
    llm_config={"config_list": [{"model": "gpt-4"}]}
)

# Function caller agent
function_caller = autogen.AssistantAgent(
    name="FunctionCaller",
    system_message="""You are responsible for calling the code_rag_search function when requested.
    Use the following format for function calls:
    FUNCTION_CALL: code_rag_search(query="<query>", chunked_data=CHUNKED_DATA, index=INDEX, top_k=<number>)
    """,
    llm_config={"config_list": [{"model": "gpt-4"}]}
)

# Group chat setup
groupchat = autogen.GroupChat(
    agents=[user_proxy, assistant, code_analyst, function_caller],
    messages=[],
    max_round=50
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": [{"model": "gpt-4"}]})

# Main function
def main():
    # Initialize the RAG system
    code_files = glob.glob("/path/to/your/project/**/*.py", recursive=True)
    initialize_rag_system(code_files)
    
    # Start the conversation
    user_proxy.initiate_chat(
        manager,
        message="Can you provide an overview of our AI system's data preprocessing techniques and model architecture based on the code?"
    )

if __name__ == "__main__":
    main()