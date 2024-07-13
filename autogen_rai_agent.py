import json

import autogen
from typing import List, Dict, Tuple, Optional, Union, Any
import glob
import torch
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

TERMINATION_MESSAGE = "TERMINATE_CONVERSATION"


class FunctionCallingAgent(autogen.AssistantAgent):
    def __init__(self, name, system_message, functions):
        super().__init__(name=name, system_message=system_message, llm_config={"config_list": [{"model": "gpt-4"}]})
        self.functions = functions
        self.last_message = None

    def generate_reply(self, messages=None, sender=None, config=None):
        print(f"FunctionCallingAgent generate_reply called")
        print(f"Messages: {messages}")
        print(f"Sender: {sender}")
        print(f"Last stored message: {self.last_message}")

        if messages is None:
            messages = [self.last_message] if self.last_message else []

        if not messages:
            return "No messages to process."

        last_message = messages[-1] if isinstance(messages, list) else messages
        last_message_content = last_message.get('content', str(last_message)) if isinstance(last_message,
                                                                                            dict) else str(last_message)

        print(f"Processing message content: {last_message_content}")

        if "FUNCTION_CALL:" in last_message_content:
            function_call = last_message_content.split("FUNCTION_CALL:", 1)[1].strip()
            function_name, args_str = function_call.split("(", 1)
            function_name = function_name.strip()
            args_str = args_str.rsplit(")", 1)[0]

            print(f"Function name: {function_name}")
            print(f"Arguments: {args_str}")

            if function_name in self.functions:
                try:
                    args = eval(f"dict({args_str})")
                    result = self.functions[function_name](**args)
                    return f"Function '{function_name}' called successfully. Result: {result}"
                except Exception as e:
                    return f"Error calling function '{function_name}': {str(e)}"
            else:
                return f"Function '{function_name}' not found."
        else:
            return "No function call detected in the message."

    def receive(self, message, sender, request_reply=False, silent=False):
        self.last_message = message
        print(f"FunctionCallingAgent received message: {message}")
        return super().receive(message, sender, request_reply, silent)


# Initialize models
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base", cache_dir="cache")
model = RobertaModel.from_pretrained("microsoft/codebert-base", cache_dir="cache")


# Helper functions
def get_embedding(text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()


# Use this function for both code and query embedding
def get_code_embedding(code: str) -> torch.Tensor:
    return get_embedding(code)


def get_query_embedding(query: str) -> torch.Tensor:
    return get_embedding(query)


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


# Global variables for chunked data and index
CHUNKED_DATA = None
INDEX = None


# Function to initialize the RAG system
def initialize_rag_system(code_files: List[str]):
    global CHUNKED_DATA, INDEX
    CHUNKED_DATA, INDEX = index_chunked_code(code_files)


# Define the functions that can be called
def code_rag_search(query: str, top_k: int = 5) -> List[Dict]:
    global CHUNKED_DATA, INDEX
    return perform_code_rag_search(query, CHUNKED_DATA, INDEX, top_k)


# Create the function calling agent
function_caller = FunctionCallingAgent(
    name="FunctionCaller",
    system_message="""You are responsible for calling functions when requested. 
    Available functions: code_rag_search(query: str, top_k: int = 5)
    When you receive a function call request, execute it and return the results.
    Do not provide recommendations in your responses.""",
    functions={"code_rag_search": code_rag_search}
)

assistant = autogen.AssistantAgent(
    name="Assistant",
    system_message="""You are an AI assistant analyzing an AI project's code using a RAG system. To search for code, 
    use: FUNCTION_CALL: code_rag_search(query="<your query>", top_k=5).
    
    Always base responses on the retrieved code, not general AI knowledge.Do not provide recommendations in your responses.""",
    llm_config={"config_list": [{"model": "gpt-4o"}]}
)

user_proxy = autogen.UserProxyAgent(
    name="Human",
    system_message="A human user interacting with the AI project analysis system.",
    human_input_mode="NEVER",
    code_execution_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE_CONVERSATION" in msg["content"],

    # code_execution_config={"work_dir": "coding"}
)

# Group chat setup with initial context-setting messages
initial_messages = [
    {
        "role": "system",
        "content": """This conversation is about analyzing a specific AI project's code using a RAG system. 
        The Assistant can request code searches using the FunctionCaller agent with the command:
        FUNCTION_CALL: code_rag_search(query="<your query>", top_k=<number>)

        Always base responses on the retrieved code, not general AI knowledge."""
    }
]

EMPTY_MESSAGE_LIMIT = 1  # Number of consecutive empty messages before termination


class ImprovedAssistantAgent(autogen.AssistantAgent):
    def __init__(self, name, system_message, llm_config):
        super().__init__(name=name, system_message=system_message, llm_config=llm_config)
        self.empty_message_count = 0
        self.conversation_summary = ""

    def generate_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[autogen.Agent] = None,
            config: Optional[Any] = None
    ) -> Union[str, Dict, None]:
        if messages is None:
            messages = self.chat_messages[sender]

        if messages and messages[-1].get('content', '').strip() == '':
            self.empty_message_count += 1
            if self.empty_message_count >= EMPTY_MESSAGE_LIMIT:
                return self.generate_summary()
        else:
            self.empty_message_count = 0

        # Use the public method to generate a reply
        reply = super().generate_reply(messages, sender)

        if reply:
            self.conversation_summary += f"{reply}\n\n"

        return reply

    def generate_summary(self):
        summary_prompt = f"""Based on the conversation so far, please provide a concise summary of the AI system 
        analyzed. Include key points discussed and any notable insights. Here's the conversation summary to base your 
        response on:

        {self.conversation_summary}

        Please provide a clear and structured summary of the AI system's implementation details and characteristics.
        """

        summary_messages = [{"role": "user", "content": summary_prompt}]
        summary = super().generate_reply(summary_messages, None)
        return f"CONVERSATION_SUMMARY: {summary}"


# Main function
def main():
    # Initialize the RAG system
    code_files = glob.glob("code/**/*.py", recursive=True)
    initialize_rag_system(code_files)

    assistant = ImprovedAssistantAgent(
        name="Assistant",
        system_message="""You are an AI assistant analyzing an AI project's code using a RAG system. 
            To search for code, use: FUNCTION_CALL: code_rag_search(query="<your query>", top_k=5)
            Provide a comprehensive analysis of the AI system based on the code. Focus on key implementation details, 
            algorithms, and design choices. Avoid providing recommendations in your response.""",
        llm_config={"config_list": [{"model": "gpt-4"}]}
    )

    # Update the group chat with the new user proxy
    groupchat = autogen.GroupChat(
        agents=[user_proxy, assistant, function_caller],
        messages=[],
        max_round=5
    )

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": [{"model": "gpt-4o"}]})
    # Start the conversation
    conversation_result = user_proxy.initiate_chat(
        manager,
        # message="To what extent is the decision-making process of the AI system explainable and interpretable",
        # message="What data was used to build the AI system",
        message="""Describe the steps taken to identify and mitigate potential biases in your AI system.
                    What fairness metrics were used?
                    How were different subgroups evaluated?""",
        max_turns=3
    )

    return conversation_result


if __name__ == "__main__":
    chat_result = main()

    # Extract the conversation history
    if hasattr(chat_result, 'chat_history'):
        conversation = chat_result.chat_history
    elif hasattr(chat_result, 'messages'):
        conversation = chat_result.messages
    else:
        conversation = []
        print("Unexpected ChatResult structure. Unable to extract conversation history.")

    # Extract the summary
    summary = next(
        (msg['content'] for msg in reversed(conversation) if msg['content'].startswith("CONVERSATION_SUMMARY:")), None)

    # Prepare the result dictionary
    result_dict = {
        "chat_history": conversation,
        "summary": summary,
        "cost": getattr(chat_result, 'cost', None),
    }

    # Write the conversation to a JSON file
    with open('conversation_result.json', 'w') as f:
        json.dump(result_dict, f, indent=2)

    print("Conversation result has been saved to 'conversation_result.json'")
