import os
from langchain_deepseek import ChatDeepSeek
from langchain_anthropic import ChatAnthropic


class DeepSeekModel:
    """
    DeepSeek model
    """
    def __init__(self):
        self.client = ChatDeepSeek(
            model="deepseek-chat",
            temperature=1.0,
            max_tokens=2048,
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
    def __init__(self):
        self.client = ChatAnthropic(
            model_name="claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens_to_sample=2048,
            timeout=60,
            max_retries=3,
            stop=None
        )
