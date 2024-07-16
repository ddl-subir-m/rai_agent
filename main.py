import json
import glob
from typing import Optional, Union

from rag_system import initialize_rag_system
from agents import CodeAnalysisAgent, UserProxyAgent, FunctionCallingAgent
from config import LLM_CONFIG
import autogen


def get_user_question():
    print("Please enter your question about the code (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        elif lines:
            return '\n'.join(lines)
        else:
            print("Question cannot be empty. Please try again.")


def process_question(question: Union[str, tuple]) -> str:
    if isinstance(question, tuple):
        return ' '.join(question)
    elif isinstance(question, str):
        return question
    else:
        raise ValueError(f"Unexpected question type: {type(question)}")


def main(code_path: str = "code/**/*.py", recursive: bool = True, question: Optional[str] = None):
    """
    Main function to run the code analysis system.

    Args:
    code_path (str): Glob pattern for the code files to analyze.
                     Defaults to "code/**/*.py".
    recursive (bool): Whether to search for files recursively.
                      Defaults to True.
    question (str, optional): The question to initiate the analysis with.
      If None, the user will be prompted to provide a question.
    """

    if question is None:
        question = get_user_question()
    else:
        question = process_question(question)

    code_files = glob.glob(code_path, recursive=recursive)

    if not code_files:
        print(f"No code files found matching the pattern: {code_path}")
        return

    print(f"Analyzing {len(code_files)} files...")

    initialize_rag_system(code_files)

    assistant = CodeAnalysisAgent(
        name="CodeAnalyst",
        system_message="""You are an AI assistant analyzing an AI project's code using a RAG system. To search for 
        code, use: FUNCTION_CALL: code_rag_search(query="<your query>", top_k=5) Provide a comprehensive analysis of 
        the AI system based on the code. Focus on key implementation details, algorithms, and design choices. Avoid 
        providing recommendations in your response. Always include citations for the code you reference in your 
        analysis. Use the format [file:filename] when citing code.""",
        llm_config=LLM_CONFIG
    )

    user_proxy = UserProxyAgent(
        name="Human",
        system_message="A human user interacting with the AI project analysis system.",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE_CONVERSATION" in msg["content"],
    )

    function_caller = FunctionCallingAgent()

    groupchat = autogen.GroupChat(
        agents=[user_proxy, assistant, function_caller],
        messages=[],
        max_round=5
    )

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=LLM_CONFIG)

    conversation_result = user_proxy.initiate_chat(
        manager,
        message=question,
        max_turns=3
    )

    return conversation_result


if __name__ == "__main__":
    # Sample questions
    # qs = "To what extent is the decision-making process of the AI system explainable and interpretable"
    # qs = "What data quality metrics and privacy considerations were used for the development of this project?"
    # qs = "What are the known limitations and constraints of the model that was chosen?"
    # qs = """Describe the steps taken to identify and mitigate potential biases in your AI system.
    qs = "Provide metrics and analysis from initial model training and validation"
    #          What fairness metrics were used?
    #          How were different subgroups evaluated?"""
    # qs = "What methods were used to provide insights into model predictions?"

    chat_result = main(question=qs)

    # Process and save results
    conversation = getattr(chat_result, 'chat_history', getattr(chat_result, 'messages', []))
    summary = next(
        (msg['content'] for msg in reversed(conversation) if msg['content'].startswith("CONVERSATION_SUMMARY:")), None)

    result_dict = {
        "chat_history": conversation,
        "summary": summary,
    }

    with open('conversation_result.json', 'w') as f:
        json.dump(result_dict, f, indent=2)

    print("Conversation result has been saved to 'conversation_result.json'")
