import autogen
from typing import List, Dict, Optional, Union, Any
import re
from config import EMPTY_MESSAGE_LIMIT, LLM_CONFIG
from rag_system import code_rag_search


class FunctionCallingAgent(autogen.AssistantAgent):
    def __init__(self):
        super().__init__(
            name="FunctionCaller",
            system_message="""You are responsible for calling functions when requested. 
            Available functions: code_rag_search(query: str, top_k: int = 5)
            When you receive a function call request, execute it and return the results.
            Always include the 'file' field from the search results in your response.
            Do not provide recommendations in your responses.""",
            llm_config=LLM_CONFIG
        )
        self.functions = {"code_rag_search": code_rag_search}
        self.last_message = None

    def generate_reply(self, messages=None, sender=None, config=None):
        if messages is None:
            messages = [self.last_message] if self.last_message else []

        if not messages:
            return "No messages to process."

        last_message = messages[-1] if isinstance(messages, list) else messages
        last_message_content = last_message.get('content', str(last_message)) if isinstance(last_message,
                                                                                            dict) else str(last_message)

        if "FUNCTION_CALL:" in last_message_content:
            function_call = last_message_content.split("FUNCTION_CALL:", 1)[1].strip()
            function_name, args_str = function_call.split("(", 1)
            function_name = function_name.strip()
            args_str = args_str.rsplit(")", 1)[0]

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
        return super().receive(message, sender, request_reply, silent)


class CodeAnalysisAgent(autogen.AssistantAgent):
    def __init__(self, name, system_message, llm_config):
        super().__init__(name=name, system_message=system_message, llm_config=llm_config)
        self.empty_message_count = 0
        self.conversation_summary = ""
        self.code_citations = []

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

        reply = super().generate_reply(messages, sender)

        if reply:
            self.conversation_summary += f"{reply}\n\n"
            self.extract_code_citations(reply)
        return reply

    def extract_code_citations(self, text):
        citations = re.findall(r'file: "(.*?:\d+)"', text)
        self.code_citations.extend(citations)

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


class UserProxyAgent(autogen.UserProxyAgent):
    def __init__(self, name, system_message, **kwargs):
        super().__init__(name=name, system_message=system_message, **kwargs)
