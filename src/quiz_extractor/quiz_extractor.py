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
from typing import List, Dict, Any, Optional, Tuple


class ExamExtractor:
    """日语试卷提取器"""

    def __init__(self):
        """
        初始化提取器
        :param model_name: 使用的模型名称
        """
        self.llm = ClaudeModel()

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
        # 题型识别提示词
        self.type_identification_prompt = ChatPromptTemplate.from_template("""
        请分析以下日语试卷内容，准确判断其题型。

        试卷内容：
        {content}

        题型识别规则：

        1. **选择题（multiple_choice）**特征：
           - 有明确的题号（如：問1、1、① 等）
           - 题目后跟着4个选项（通常标记为 1,2,3,4）
           - 每题相对独立，不依赖于文章
           - 示例格式：
             問1 （　）に入れるのに最もよいものを1·2·3·4から一つ選びなさい。
             私は昨日（　）へ行きました。
             1. 学校　2. 会社　3. 病院　4. 銀行

        2. **完型填空（cloze）**特征：
           - 有一篇连续的文章
           - 文章中有多个空格，通常用　＿41＿、＿42＿等标记
           - 空格后面集中列出所有选项
           - 示例格式：
             次の文章を読んで、文章全体の趣旨を踏まえて、＿41＿から＿45＿の中に入る最もよいものを1・2・3・4から一つ選びなさい。
             日本の＿41＿は四季がはっきりしています。春は＿42＿が咲き...
             41 1.天気 2.気候 3.気温 4.天候
             42 1.花 2.桜 3.梅 4.菊

        3. **阅读理解（reading）**特征：
           - 先有一篇较长的文章（通常超过3-4句）
           - 文章后有针对内容的问题
           - 问题通常问"何が"、"どうして"、"いつ"等
           - 示例格式：
             次の文章を読んで、後の問題に対する答えとして最もよいものを1・2・3・4から一つ選びさい。
             [长文章内容]
             46 筆者が一番伝えたいことは何ですか。
             1. ... 2. ... 3. ... 4. ...

        请仔细分析内容特征，并以json格式返回题型识别结果：
        输出json格式：
        [
            {{
                "question_type": "题型（multiple_choice/cloze/reading）",
                "confidence": "置信度（high/medium/low）",
                "reason": "判断理由（列出具体特征）",
                "key_features": ["识别到的关键特征1", "特征2", ...]
            }}
        ]
        """)

        # 选择题提取提示
        self.multiple_choice_prompt = ChatPromptTemplate.from_template("""
        请从以下日语试卷内容中提取所有选择题，并按照指定的JSON格式输出。

        试卷内容：
        {content}

        提取要求：
        1. 识别题目编号、题目内容、选项
        2. 如果有答案, 请一并提取, 如果没有请忽略, 只需要提取答案编号：(1,2,3,4)
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

        # 完型填空模版
        self.cloze_prompt = ChatPromptTemplate.from_template("""
        请从以下日语试卷内容中提取完型填空题，并按照指定的JSON格式输出。

        试卷内容：
        {content}

        提取要求：
        1. 识别文章内容
        2. 识别每个空格对应的题目编号和选项
        3. 如果有答案, 请一并提取, 如果没有请忽略, 只需要提取答案编号：(1,2,3,4)

        输出JSON格式：
        {{
            "question_type": "cloze",
            "reading_passage": "文章内容（保留空格标记）",
            "questions": [
                {{
                    "question_id": "题目编号",                
                    "options": ["选项A", "选项B", "选项C", "选项D"],
                    "correct_answer": "正确答案（如果有）"
                }},
                {{
                    "question_id": "题目编号",                
                    "options": ["选项A", "选项B", "选项C", "选项D"],
                    "correct_answer": "正确答案（如果有）"
                }}
            ]
        }}

        请确保返回的是一个JSON对象。        
        """)

        # 阅读理解模版
        self.reading_prompt = ChatPromptTemplate.from_template("""
        请从以下日语试卷内容中提取阅读理解题，并按照指定的JSON格式输出。

        试卷内容：
        {content}

        提取要求：
        1. 识别阅读文章内容
        2. 识别每个问题的编号、内容和选项
        3. 如果有答案, 请一并提取, 如果没有请忽略, 只需要提取答案编号：(1,2,3,4)

        输出JSON格式：
        {{
            "question_type": "reading",
            "reading_passage": "文章内容",
            "questions": [
                {{
                    "question_id": "题目编号",
                    "question_text": "题目内容",                
                    "options": ["选项A", "选项B", "选项C", "选项D"],
                    "correct_answer": "正确答案（如果有）"
                }},
                {{
                    "question_id": "题目编号",
                    "question_text": "题目内容",                
                    "options": ["选项A", "选项B", "选项C", "选项D"],
                    "correct_answer": "正确答案（如果有）"
                }}
            ]
        }}

        请确保返回的是一个JSON对象。        
        """)

    def identify_question_type(self, content: str) -> Tuple[str, str]:
        """
        识别题型
        :param content: 题目内容
        :return: (题型, 置信度)
        """
        try:
            prompt = self.type_identification_prompt.format(content=content)
            chain = self.llm.client
            result = chain.invoke(prompt)
            if result is None:
                return "unknown", "low"

            # 提取JSON内容
            json_content = self._extract_json_from_text(result.content)
            print("题目判断：" + json_content)
            if json_content:
                parsed_result = json.loads(json_content)[0]

                return parsed_result.get("question_type", "unknown"), parsed_result.get("confidence", "low")

            # 如果JSON提取失败，使用简单的规则判断
            return self._simple_type_identification(content)

        except Exception as e:
            print(f"题型识别失败: {e}")
            return self._simple_type_identification(content)

    def _simple_type_identification(self, content: str) -> Tuple[str, str]:
        """
        简单的题型识别规则
        :param content: 题目内容
        :return: (题型, 置信度)
        """
        # 检查是否包含文章和多个空格（完型填空特征）
        if re.search(r'[\(（]\s*\d+\s*[\)）]', content) and content.count('(') > 3:
            return "cloze", "medium"

        # 检查是否有明显的阅读文章特征
        if len(content) > 500 and re.search(r'問\d+', content):
            return "reading", "medium"

        # 默认为选择题
        return "multiple_choice", "low"

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

    def extract_questions_by_type(self, content: str, question_type: str) -> Dict[str, Any]:
        """
        根据题型提取题目
        :param content: 试卷内容
        :param question_type: 题型
        :return: 提取结果
        """
        try:
            # 根据题型选择相应的提示词
            if question_type == "multiple_choice":
                prompt = self.multiple_choice_prompt.format(content=content)
            elif question_type == "cloze":
                prompt = self.cloze_prompt.format(content=content)
            elif question_type == "reading":
                prompt = self.reading_prompt.format(content=content)
            else:
                print(f"未知题型: {question_type}")
                return {"extraction_status": "unknown_type", "question_type": question_type}

            chain = self.llm.client
            result = chain.invoke(prompt)

            if result is None:
                print(f"LLM返回结果为空")
                return {"extraction_status": "failed", "question_type": question_type}

            # 提取JSON内容
            result_content = result.content
            print(f"LLM返回内容: {result_content[:200]}...")  # 只打印前200个字符

            # 尝试从返回内容中提取JSON
            json_content = self._extract_json_from_text(result_content)

            if json_content:
                parsed_result = json.loads(json_content)

                # 对于选择题，确保返回列表格式
                if question_type == "multiple_choice":
                    if isinstance(parsed_result, dict):
                        return {"questions": [parsed_result], "question_type": question_type}
                    elif isinstance(parsed_result, list):
                        return {"questions": parsed_result, "question_type": question_type}
                else:
                    # 完型填空和阅读理解返回单个对象
                    return parsed_result
            else:
                print(f"无法从返回内容中提取JSON")
                return {"raw_result": result_content, "extraction_status": "failed", "question_type": question_type}

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return {"raw_result": result_content, "extraction_status": "failed", "error": str(e),
                    "question_type": question_type}
        except Exception as e:
            print(f"提取过程发生错误: {e}")
            return {"extraction_status": "error", "error": str(e), "question_type": question_type}

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
            "cloze": [],
            "reading": [],
            "extraction_errors": []  # 添加错误记录
        }

        # 对每一块文本进行题目提取
        for i, chunk in enumerate(chunks):
            print(f"\n正在处理第 {i + 1} / {len(chunks)} 个文本块...")
            print(f"文本块内容预览: {chunk[:100]}...")  # 显示前100个字符

            # 首先识别题型
            question_type, confidence = self.identify_question_type(chunk)
            print(f"识别题型: {question_type} (置信度: {confidence})")

            # 根据题型提取题目
            extracted_data = self.extract_questions_by_type(chunk, question_type)

            # 处理提取结果
            if "extraction_status" in extracted_data and extracted_data["extraction_status"] in ["failed", "error",
                                                                                                 "unknown_type"]:
                extracted_data["chunk_index"] = i + 1
                extracted_data["chunk_preview"] = chunk[:100]
                all_questions["extraction_errors"].append(extracted_data)
            else:
                if question_type == "multiple_choice":
                    if "questions" in extracted_data:
                        all_questions["multiple_choice"].extend(extracted_data["questions"])
                    elif isinstance(extracted_data, list):
                        all_questions["multiple_choice"].extend(extracted_data)
                    elif isinstance(extracted_data, dict) and "question_id" in extracted_data:
                        all_questions["multiple_choice"].append(extracted_data)
                elif question_type == "cloze":
                    if isinstance(extracted_data, dict) and (
                            "questions" in extracted_data or "reading_passage" in extracted_data):
                        all_questions["cloze"].append(extracted_data)
                    else:
                        all_questions["extraction_errors"].append({
                            "chunk_index": i + 1,
                            "error": "Invalid cloze format",
                            "data": extracted_data
                        })
                elif question_type == "reading":
                    if isinstance(extracted_data, dict) and (
                            "questions" in extracted_data or "reading_passage" in extracted_data):
                        all_questions["reading"].append(extracted_data)
                    else:
                        all_questions["extraction_errors"].append({
                            "chunk_index": i + 1,
                            "error": "Invalid reading format",
                            "data": extracted_data
                        })
                else:
                    # 未知格式，记录错误
                    all_questions["extraction_errors"].append({
                        "chunk_index": i + 1,
                        "error": "Unknown format",
                        "data": extracted_data
                    })

        # 统计信息
        total_questions = len(all_questions["multiple_choice"])

        # 统计完型填空题目数
        for q in all_questions["cloze"]:
            if isinstance(q, dict) and "questions" in q:
                total_questions += len(q["questions"])
            elif isinstance(q, list):
                total_questions += len(q)

        # 统计阅读理解题目数
        for q in all_questions["reading"]:
            if isinstance(q, dict) and "questions" in q:
                total_questions += len(q["questions"])
            elif isinstance(q, list):
                total_questions += len(q)

        all_questions["exam_info"]["total_questions_extracted"] = total_questions
        all_questions["exam_info"]["extraction_errors_count"] = len(all_questions["extraction_errors"])
        all_questions["exam_info"]["multiple_choice_count"] = len(all_questions["multiple_choice"])
        all_questions["exam_info"]["cloze_sections_count"] = len(all_questions["cloze"])
        all_questions["exam_info"]["reading_sections_count"] = len(all_questions["reading"])

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
        print(f"选择题数量: {questions_data['exam_info']['multiple_choice_count']}")
        print(f"完型填空部分数量: {questions_data['exam_info']['cloze_sections_count']}")
        print(f"阅读理解部分数量: {questions_data['exam_info']['reading_sections_count']}")
        print(f"题目总数: {questions_data['exam_info']['total_questions_extracted']}")
        print(f"提取错误数量: {questions_data['exam_info']['extraction_errors_count']}")

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
            input_file="../../output/ocr_text.txt",
            output_file="../../output/ocr_result.json"
        )
        print("\n试卷处理完成！")
    except Exception as e:
        print(f"\n处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
