import json
import os
import sys
import re
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from regex import split

from llm import DeepSeekModel
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict
from langchain.globals import set_verbose
from langchain.output_parsers import PydanticOutputParser, JsonOutputToolsParser
from pydantic import ValidationError
from qes_type import BaseQuestion
from typing import List, Dict, Any, Optional


def split_text(text: str, chunk_size: int = 2000, chunk_overlap: int = 50) -> list[str]:
    """将文本长度分割成适合处理的CHUNK"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)


def split_question(file_text: str) -> list[str]:
    """将试卷内容按照题目拆分成list"""
    pattern = re.compile(r"(問題\d+.*?)")
    file_content = re.split(pattern, file_text)

    text_list = []
    for i in range(1, len(file_content)-1, 2):
        title = file_content[i].strip()
        content = file_content[i+1].strip()
        text_list.append((title, content))
    return text_list


def process_in_batches(text: str, chain: LLMChain) -> list[str]:
    """分批次处理长文本"""
    chunks = split_question(text)
    for i, chunk in enumerate(chunks, start=1):
        print(f"正在处理第{i}/{len(chunks)} 块 (长度: {len(chunk)})")
        try:
            result = chain.invoke(chunk)
            print(f"{result}")
            if result is not None:
                # 提取消息内容
                if hasattr(result, 'content'):
                    content = result.content
                elif isinstance(result, dict) and 'text' in result:
                    content = result['text']
                else:
                    content = str(result)

                json_path = "../../output"
                os.makedirs(json_path, exist_ok=True)
                with open(os.path.join(json_path, f"{i}.json"), "w", encoding='utf-8') as f:
                    json.dump(content, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"处理第 {i} 块时出现了错误: {str(e)}")
            continue


def merge_questions(question_batches: list[str]) -> Dict:
    """合并分批处理的结果"""
    merged = {
        "exam_title": "",
        "questions": [],
        "metadata": {
            "total_questions": 0,
            "batch_count": len(question_batches)
        }
    }
    seen_numbers = set()

    for batch in question_batches:
        for q in batch.get("questions", []):
            if q["question_number"] not in seen_numbers:
                seen_numbers.add(q["question_number"])

    merged["metadata"]["total_questions"] = len(merged["questions"])
    return merged


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

示例模板(实际返回时不要包含注释):
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

# 初始化LLM
llm = DeepSeekModel()

prompt = PromptTemplate(
    input_variables=["text"],
    template=MULTI_TYPE_PROMPT,
)

chain = prompt | llm.client


def process_exam_in_batches(file_path: str, output_path: str) -> None:
    """主处理函数"""
    # 加载试卷
    # loader = TextLoader(file_path)
    # documents = loader.load()
    # full_text = documents[0].page_content

    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    # 分批次处理
    print(f"====== 开始分批次处理试卷 ======")

    process_in_batches(full_text, chain)

    print(f"====== 结果保存至{output_path} ======")


if __name__ == '__main__':
    process_exam_in_batches(
        file_path="../../output/ocr_text.txt",
        output_path=os.path.join("/out_put.json")
    )

    # 以下为测试代码
    # with open("../../output/ocr_text.txt", "r", encoding="utf-8") as f:
    #     text = f.read()
    #     print(type(text))
    #
    #     file_content = re.split(r"(問題\d+.*?)", text)
    #     for i, chunk in enumerate(file_content, start=1):
    #         print(f"正在处理{i}/{len(file_content)} 部分 (<文件长度>: {len(chunk)})个字符")
    #         print(chunk)


