import json
import os
import sys

from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from regex import split

from llm import DeepSeekModel
from typing import Optional
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict
from langchain.globals import set_verbose
from langchain.output_parsers import PydanticOutputParser
from pydantic import ValidationError
from qes_type import BaseQuestion

# 增强提示词
MULTI_TYPE_PROMPT = """
你是一位专业的日语试卷分析专家，请从以下试卷内容中提取所有试题，并按照要求分类整理。

试卷内容：
{text}

提取要求：
1. 识别每道题的题号(如：1, 2, 3)
2. 准确判断题型 (选择题, 完型填空, 阅读理解)
3. 根据题型提取相应的内容：
   - 选择题: 题目内容、所有选项、正确答案
   - 完型填空: 文章内容、相关问题、所有选项
   - 阅读理解: 文章内容、相关问题、所有选项
4. 保持原始题目的完整性

请严格按以下JSON格式返回结果（只需返回一个有效的JSON对象，不要包含其他说明）：

{{{{
  "questions": [
    {{
      "question_number": "题号",
      "question_type": "题型",
      "question_text": "题目内容",
      "passage": "文章内容（仅完型填空/阅读理解需要）",
      "options": ["选项1", "选项2", "选项3", "选项4"],
      "answer": "正确答案"
    }}
  ]
}}}}

示例模板（实际返回时不要包含注释）：
{{{{
  "questions": [
    # 选择题示例
    {{
      "question_number": "1",
      "question_type": "选择题",
      "question_text": "社会活動に参加することで、人脈を広げることができた。",
      "options": ["あつい", "あかい", "あまい", "あさい"],
      "answer": "1"
    }},
    # 完型填空示例
    {{
      "question_number": "41",
      "question_type": "完型填空",
      "passage": "次の文章を読んで、後の問いに答えなさい...",
      "question_text": "空欄に入る最も適切な語は？",
      "options": ["...", "...", "...", "..."],
      "answer": "3"
    }}
  ]
}}}}
"""


def extract_questions(file_path: str, output_path: str):
    # 加载试卷
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    text = documents[0].page_content

    # 实例化模型
    llm = DeepSeekModel()

    # 设置输出解析器
    paser = PydanticOutputParser(pydantic_object=BaseQuestion)
    print("开始分批次处理")
    question_batches = []
    # 创建处理链
    prompt = PromptTemplate(
        template=MULTI_TYPE_PROMPT,
        input_variables=['text'],
        partial_variables={
            "format_instructions": paser.get_format_instructions()
        }
    )

    chain = prompt | llm.client | paser

    try:
        result = chain.invoke({"text": text})

        # 处理试卷
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.dict(), f, ensure_ascii=False, indent=2)
        print(f"成功提取到{output_path}")

    except ValidationError as e:
        print(f"数据验证错误:{e}")
    except Exception as e:
        print(f"逻辑出错了:{e}")


if __name__ == '__main__':
    extract_questions("../../output/ocr_text.txt", "../../output/questions.json")
