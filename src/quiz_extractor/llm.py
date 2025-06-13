import os
from langchain_deepseek import ChatDeepSeek


class DeepSeekModel:
    """
    DeepSeek model
    """
    def __init__(self):
        self.client = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=1024,
            timeout=60,
            max_retries=3,
        )


class ChatGPTModel:
    """
    ChatGPT model
    """
    pass


class ClaudeModel:
    """
    claude model
    """
    pass

