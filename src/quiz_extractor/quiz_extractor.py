import json
import os
import sys
import re
from datetime import datetime
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from regex import split

from llm import DeepSeekModel, ClaudeModel
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict
from langchain.globals import set_verbose
from langchain.output_parsers import PydanticOutputParser, JsonOutputToolsParser
from pydantic import ValidationError
from qes_type import BaseQuestion
from typing import List, Dict, Any, Optional


class ExamExtractor:
    """日语试卷提取器"""

    def __init__(self):
        """
        初始化提取器
        :param model_name: 使用的模型名称
        """
        self.llm = DeepSeekModel()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "、"]
        )

        # 设置提示词模版
        self._setup_prompt()

    def load_exam_file(self, file_path: str) -> str:
        """
        加载试卷文件
        :param file_path: 文件路径
        :return: 试卷文本内容
        """
        if file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            return documents[0].page_content
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                documents = f.read()
                return documents

    def _setup_prompt(self):
        """设置提示词模版"""
        # 选择题提取提示
        self.multiple_choice_prompt = ChatPromptTemplate.from_template("""
        请从以下日语试卷内容中提取所有选择题，并按照指定的JSON格式输出。

        试卷内容：
        {content}

        提取要求：
        1. 识别题目编号、题目内容、选项
        2. 如果有答案, 请一并提取, 如果没有请忽略
        3. 如果有多个题目，请返回一个包含所有题目的JSON数组

        输出JSON格式：
        [
            {{
                "question_id": "题目编号",
                "question_type": "multiple_choice",
                "question_text": "题目内容",
                "options": ["选项A", "选项B", "选项C", "选项D"],
                "correct_answer": "正确答案（如果有）"
            }}
        ]

        请确保返回的是一个JSON数组，即使只有一个题目。
        """)

    def split_question(self, exam_text) -> list[str]:
        """
        将试卷内容按照题目拆分成list
        用于代替langchain自带的text_spliter
        不是通用方法
        """
        pattern = re.compile(r"(問題\d+.*?)")
        file_content = re.split(pattern, exam_text)

        text_list = []
        for i in range(1, len(file_content) - 1, 2):
            title = file_content[i].strip()
            content = file_content[i + 1].strip()
            # 将标题和内容合并，保留完整的题目信息
            text_list.append(f"{title}\n{content}")
        return text_list

    def extract_multiple_choice(self, content: str) -> List[Dict[str, Any]]:
        """
        提取选择题
        :param content: 试卷内容
        :return: 题目列表
        """
        try:
            # 使用改进的提示词模板
            prompt = self.multiple_choice_prompt.format(content=content)
            chain = self.llm.client
            result = chain.invoke(prompt)

            if result is None:
                print(f"LLM返回结果为空")
                return []

            # 提取JSON内容
            result_content = result.content
            print(f"LLM返回内容: {result_content[:200]}...")  # 只打印前200个字符

            # 尝试从返回内容中提取JSON
            json_content = self._extract_json_from_text(result_content)

            if json_content:
                parsed_result = json.loads(json_content)
                # 确保返回的是列表
                if isinstance(parsed_result, dict):
                    return [parsed_result]
                elif isinstance(parsed_result, list):
                    return parsed_result
                else:
                    print(f"解析结果类型异常: {type(parsed_result)}")
                    return []
            else:
                print(f"无法从返回内容中提取JSON")
                return [{"raw_result": result_content, "extraction_status": "failed"}]

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return [{"raw_result": result_content, "extraction_status": "failed", "error": str(e)}]
        except Exception as e:
            print(f"提取过程发生错误: {e}")
            return [{"extraction_status": "error", "error": str(e)}]

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        从文本中提取JSON内容
        :param text: 包含JSON的文本
        :return: JSON字符串或None
        """
        # 尝试找到JSON数组或对象的开始和结束
        patterns = [
            (r'\[[\s\S]*\]', 'array'),  # JSON数组
            (r'\{[\s\S]*\}', 'object')  # JSON对象
        ]

        for pattern, json_type in patterns:
            matches = re.findall(pattern, text)
            if matches:
                # 取最长的匹配（通常是完整的JSON）
                json_str = max(matches, key=len)
                try:
                    # 验证是否为有效JSON
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    continue

        return None

    def extract_all_questions(self, exam_content: str) -> Dict[str, Any]:
        """
        提取所有类型题目
        :param exam_content: 试卷内容
        :return: 提取结果字典
        """
        # 使用自定义问题分割器分割
        chunks = self.split_question(exam_content)

        all_questions = {
            "exam_info": {
                "extraction_date": datetime.now().isoformat(),
                "total_chunks": len(chunks),
                "content_length": len(exam_content),
            },
            "multiple_choice": [],
            "cloze_question": [],
            "reading_comprehension": [],
            "extraction_errors": []  # 添加错误记录
        }

        # 对每一块文本进行题目提取
        for i, chunk in enumerate(chunks):
            print(f"\n正在处理第 {i + 1} / {len(chunks)} 个文本块...")
            print(f"文本块内容预览: {chunk[:100]}...")  # 显示前100个字符

            # 提取选择题
            mc_questions = self.extract_multiple_choice(chunk)

            # 分离成功提取的题目和错误信息
            for question in mc_questions:
                if "extraction_status" in question and question["extraction_status"] in ["failed", "error"]:
                    question["chunk_index"] = i + 1
                    all_questions["extraction_errors"].append(question)
                else:
                    all_questions["multiple_choice"].append(question)

            print(f"本块提取到 {len(mc_questions)} 个题目")

        # 统计信息
        all_questions["exam_info"]["total_questions_extracted"] = len(all_questions["multiple_choice"])
        all_questions["exam_info"]["extraction_errors_count"] = len(all_questions["extraction_errors"])

        return all_questions

    def save_to_json(self, questions_data: Dict[str, Any], output_path: str):
        """
        保存提取结果到json文件
        :param questions_data: 题目数据
        :param output_path: 输出文件路径
        :return: None
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(questions_data, f, ensure_ascii=False, indent=4)

        print(f"提取结果已经保存到: {output_path}")

    def process_exam(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        处理完整的试卷提取流程
        :param input_file: 输入试卷文件路径
        :param output_file: 输出json文件路径
        :return: 提取结果
        """
        print(f"正在加载试卷文件: {input_file}")

        # 检查文件是否存在
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")

        exam_content = self.load_exam_file(input_file)
        print(f"文件加载成功，内容长度: {len(exam_content)} 字符")

        print(f"\n正在提取题目...")
        questions_data = self.extract_all_questions(exam_content)

        print(f"\n正在保存结果...")
        self.save_to_json(questions_data, output_file)

        # 打印统计信息
        print("\n=== 提取统计 ===")
        print(f"文本块总数: {questions_data['exam_info']['total_chunks']}")
        print(f"选择题数量: {len(questions_data['multiple_choice'])}")
        print(f"完型填空题数量: {len(questions_data['cloze_question'])}")
        print(f"阅读理解题数量: {len(questions_data['reading_comprehension'])}")
        print(f"提取错误数量: {len(questions_data['extraction_errors'])}")

        return questions_data


# 使用示例
def main():
    """
    主函数示例
    :return:
    """
    # 初始化提取器
    extractor = ExamExtractor()

    try:
        # 处理试卷
        extractor.process_exam(
            input_file="../../output/ocr_text_test.txt",
            output_file="../../output/ocr_result.json"
        )
        print("\n试卷处理完成！")
    except Exception as e:
        print(f"\n处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()