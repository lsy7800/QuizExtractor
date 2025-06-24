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

    def preprocess_ocr_text(self, text: str) -> str:
        """
        预处理OCR文本，修复常见识别错误
        :param text: 原始OCR文本
        :return: 修复后的文本
        """
        # 常见OCR错误替换
        replacements = {
            '闘': '問',
            '周い': '問い',
            '见': '見',
            '话': '話',
            '决': '決',
            '间': '問',
            '润': '問',
            '阅': '問',
            '间题': '問題',
            '润题': '問題',
            '阅题': '問題',
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # 修复数字后的点号
        text = re.sub(r'(\d)\s*[.．。]\s*', r'\1. ', text)

        return text

    def _setup_prompt(self):
        """设置提示词模版"""
        # 题型识别提示词
        self.type_identification_prompt = ChatPromptTemplate.from_template("""
        请分析以下日语试卷内容，准确判断其题型。注意：这是OCR识别的文本，可能存在识别错误。

        试卷内容：
        {content}

        题型识别规则：

        1. **阅读理解（reading）**特征：
           - 通常包含以下标记之一：
             * "次の文章を読んで" 或 "次の(1)から(4)の文章を読んで"
             * "以下は...である"（如："以下は小説家が書いたエッセイである"）
             * "次の文を読んで"
             * 問題8、問題9、問題10、問題11、問題12、問題13 等标记
           - 包含较长的文章（通常超过100字）
           - 文章后有针对内容的问题，题号如：46、47、48 或 問1、問2
           - 问题通常包含：
             * "筆者の考えに合うのはどれか"
             * "～とあるが"
             * "～についてどのように述べているか"
             * "最も言いたいことは何か"

        2. **完型填空（cloze）**特征：
           - 通常有"文章全体の趣旨を踏まえて"这样的说明
           - 文章中有编号的空格：41、42、43 或 ＿41＿、＿42＿
           - 所有选项集中列在文章后面
           - 通常是問題7这种题目

        3. **选择题（multiple_choice）**特征：
           - 每题相对独立
           - 没有长篇文章
           - 题目简短，直接跟着选项

        请仔细分析，返回最可能的题型。

        输出json格式：
        [
            {{
                "question_type": "题型（reading/cloze/multiple_choice）",
                "confidence": "置信度（high/medium/low）",
                "reason": "判断理由",
                "key_features": ["识别到的关键特征"]
            }}
        ]
        """)

        # 选择题提取提示
        self.multiple_choice_prompt = ChatPromptTemplate.from_template("""
        请从以下日语试卷内容中提取所有选择题，并按照指定的JSON格式输出。
        注意：这是OCR文本，可能存在识别错误。

        试卷内容：
        {content}

        提取要求：
        1. 识别题目编号、题目内容、选项
        2. 如果有答案, 请一并提取, 如果没有请忽略, 只需要提取答案编号：(1,2,3,4)
        3. 如果有多个题目，请返回一个包含所有题目的JSON数组
        4. 注意OCR可能的错误，如：闘→問、见→見等

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
        注意：这是OCR文本，可能存在识别错误。

        试卷内容：
        {content}

        提取要求：
        1. 识别文章内容（保留空格标记）
        2. 识别每个空格对应的题目编号和选项
        3. 空格标记通常是数字，如 41、42、43 等
        4. 选项通常在文章后面集中列出

        输出JSON格式：
        {{
            "question_type": "cloze",
            "reading_passage": "文章内容（保留空格标记）",
            "questions": [
                {{
                    "question_id": "题目编号",                
                    "options": ["选项A", "选项B", "选项C", "选项D"],
                    "correct_answer": "正确答案（如果有）"
                }}
            ]
        }}

        请确保返回的是一个JSON对象。
        """)

        # 阅读理解模版 - 修改为支持返回列表
        self.reading_prompt = ChatPromptTemplate.from_template("""
        请从以下日语试卷内容中提取阅读理解题。这是OCR识别的文本，可能存在识别错误和格式问题。

        试卷内容：
        {content}

        **重要提取规则**：

        1. **文章识别**：
           - 阅读理解可能以以下方式开始：
             * "次の文章を読んで" 或 "次の(1)から(4)の文章を読んで"
             * "以下は...である"
             * 直接是(1)、(2)、(3)、(4)等标记后跟文章
             * 問題8、問題9等标记后的内容
           - 文章通常是较长的连续段落
           - 注意：一个问题块中可能包含多篇短文，如(1)、(2)、(3)、(4)

        2. **问题识别**：
           - 题号格式：46、47、48、49、50...或 問1、問2...
           - 每个问题都有4个选项（1. 2. 3. 4.）
           - 问题通常在文章之后
           - 常见问题模式：
             * "筆者の考えに合うのはどれか"
             * "～とあるが、～は何か"
             * "～について～はどのように述べているか"

        3. **提取策略**：
           - 如果有多篇短文（如(1)、(2)标记），分别提取每篇文章
           - 确保提取所有相关问题
           - 保持原文内容，包括OCR错误

        输出JSON格式：

        如果只有一篇文章，返回单个对象：
        {{
            "question_type": "reading",
            "reading_passage": "完整的文章内容",
            "questions": [
                {{
                    "question_id": "题目编号",
                    "question_text": "完整的问题内容",                
                    "options": ["选项1", "选项2", "选项3", "选项4"],
                    "correct_answer": "正确答案编号（如果有）"
                }}
            ]
        }}

        如果有多篇文章，返回数组：
        [
            {{
                "question_type": "reading",
                "reading_passage": "第一篇文章内容",
                "questions": [...]
            }},
            {{
                "question_type": "reading", 
                "reading_passage": "第二篇文章内容",
                "questions": [...]
            }}
        ]

        请根据实际情况返回合适的格式。
        """)

    def split_question_improved(self, exam_text: str) -> List[Tuple[str, str]]:
        """
        改进的试卷内容分割方法
        返回 (問題编号, 内容) 的列表
        """
        # 预处理文本
        exam_text = self.preprocess_ocr_text(exam_text)

        # 使用更精确的分割策略
        chunks = []

        # 首先尝试按問題X分割
        problem_pattern = r'問題(\d+)'
        problem_matches = list(re.finditer(problem_pattern, exam_text))

        if problem_matches:
            # 按問題分割
            for i in range(len(problem_matches)):
                problem_num = problem_matches[i].group(1)
                start = problem_matches[i].start()
                if i < len(problem_matches) - 1:
                    end = problem_matches[i + 1].start()
                else:
                    end = len(exam_text)

                chunk = exam_text[start:end].strip()
                if len(chunk) > 50:  # 过滤太短的块
                    chunks.append((f"問題{problem_num}", chunk))
        else:
            # 如果没有找到問題标记，将整个文本作为一个块
            chunks.append(("未知", exam_text))

        return chunks

    def identify_question_type(self, content: str) -> Tuple[str, str]:
        """
        识别题型 - 改进版
        """
        # 首先使用规则判断

        # 检查是否是完型填空（問題7特征）
        if '問題7' in content and re.search(r'[_＿]?\d{2}[_＿]?', content):
            return "cloze", "high"

        # 检查是否包含明显的阅读理解特征
        reading_indicators = [
            '問題8', '問題9', '問題10', '問題11', '問題12', '問題13',
            '次の文章を読んで', '次の(1)から', '以下は', '次の文を読んで'
        ]

        if any(indicator in content for indicator in reading_indicators):
            # 检查是否有问题编号（46, 47等）
            if re.search(r'\n\s*\d{2}\s+[^\d]', content):
                return "reading", "high"

        # 检查文章长度和问题格式
        if len(content) > 400 and re.search(r'\n\s*\d{2}\s+[^\d]', content):
            return "reading", "medium"

        # 尝试使用LLM判断
        try:
            prompt = self.type_identification_prompt.format(content=content)
            chain = self.llm.client
            result = chain.invoke(prompt)
            if result is None:
                return "unknown", "low"

            json_content = self._extract_json_from_text(result.content)
            if json_content:
                parsed_result = json.loads(json_content)[0]
                return parsed_result.get("question_type", "unknown"), parsed_result.get("confidence", "low")

        except Exception as e:
            print(f"题型识别失败: {e}")

        return "multiple_choice", "low"

    def extract_reading_questions_advanced(self, content: str) -> List[Dict[str, Any]]:
        """
        高级阅读理解提取方法 - 返回列表以支持多篇文章
        """
        results = []

        # 预处理内容
        content = self.preprocess_ocr_text(content)

        # 检查是否有(1)、(2)、(3)、(4)格式的多篇文章
        sub_article_pattern = r'\(\d+\)\s*\n([\s\S]+?)(?=\(\d+\)|$)'
        sub_articles = list(re.finditer(sub_article_pattern, content))

        if sub_articles:
            # 处理多篇短文的情况
            # 首先提取所有文章
            articles = []
            for match in sub_articles:
                article_text = match.group(1).strip()
                # 提取到问题开始为止的文本
                question_start = re.search(r'\n\s*\d{2}\s+[^\d]', article_text)
                if question_start:
                    article_text = article_text[:question_start.start()].strip()
                articles.append(article_text)

            # 提取所有问题
            all_questions = self._extract_questions_from_content(content)

            # 如果只有一组问题，创建一个包含所有文章的结果
            if all_questions:
                result = {
                    "question_type": "reading",
                    "reading_passage": "\n\n".join(f"({i + 1})\n{article}" for i, article in enumerate(articles)),
                    "questions": all_questions
                }
                results.append(result)
        else:
            # 处理单篇文章的情况
            result = self._extract_single_reading(content)
            if result["questions"]:
                results.append(result)

        return results

    def _extract_single_reading(self, content: str) -> Dict[str, Any]:
        """
        提取单篇阅读理解
        """
        result = {
            "question_type": "reading",
            "reading_passage": "",
            "questions": []
        }

        # 查找文章开始标记
        article_start_patterns = [
            r'次の文章を読んで[^。]*。?\s*\n([\s\S]+?)(?=\n\s*\d{2}\s+[^\d])',
            r'以下は[^。]+である[^。]*。?\s*\n([\s\S]+?)(?=\n\s*\d{2}\s+[^\d])',
            r'問題\d+[^\n]*\n([\s\S]+?)(?=\n\s*\d{2}\s+[^\d])'
        ]

        article_text = ""
        for pattern in article_start_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                article_text = match.group(1).strip()
                break

        if not article_text:
            # 尝试提取较长的连续文本
            lines = content.split('\n')
            article_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 如果遇到问题行，停止
                if re.match(r'^\d{2}\s+[^\d]', line):
                    break

                # 跳过标题行
                if re.match(r'^問題\d+', line) or re.match(r'^\(\d+\)$', line):
                    continue

                if len(line) > 10:
                    article_lines.append(line)

            article_text = '\n'.join(article_lines)

        result["reading_passage"] = article_text

        # 提取问题
        result["questions"] = self._extract_questions_from_content(content)

        return result

    def _extract_questions_from_content(self, content: str) -> List[Dict[str, Any]]:
        """
        从内容中提取所有问题
        """
        questions = []

        # 问题模式 - 更宽松的匹配
        question_pattern = r'(\d{2,})\s+([^\n]*(?:\n(?!\s*\d+\s*[.．])[^\n]*)*)\s*\n\s*1\s*[.．]\s*([^\n]+)\s*\n\s*2\s*[.．]\s*([^\n]+)\s*\n\s*3\s*[.．]\s*([^\n]+)\s*\n\s*4\s*[.．]\s*([^\n]+)'

        matches = re.finditer(question_pattern, content, re.MULTILINE)

        for match in matches:
            question = {
                "question_id": match.group(1),
                "question_text": match.group(2).strip().replace('\n', ' '),
                "options": [
                    match.group(3).strip(),
                    match.group(4).strip(),
                    match.group(5).strip(),
                    match.group(6).strip()
                ]
            }
            questions.append(question)

        return questions

    def extract_questions_by_type(self, content: str, question_type: str, problem_num: str) -> Dict[str, Any]:
        """
        改进的题目提取方法 - 添加问题编号参数
        """
        try:
            # 如果是阅读理解，先尝试高级提取方法
            if question_type == "reading":
                advanced_results = self.extract_reading_questions_advanced(content)
                if advanced_results:
                    # 如果有多个结果，返回特殊格式
                    if len(advanced_results) > 1:
                        return {
                            "problem_number": problem_num,
                            "question_type": "reading",
                            "multiple_passages": True,
                            "passages": advanced_results
                        }
                    else:
                        result = advanced_results[0]
                        result["problem_number"] = problem_num
                        return result

            # 根据题型选择相应的提示词
            if question_type == "multiple_choice":
                prompt = self.multiple_choice_prompt.format(content=content)
            elif question_type == "cloze":
                prompt = self.cloze_prompt.format(content=content)
            elif question_type == "reading":
                prompt = self.reading_prompt.format(content=content)
            else:
                print(f"未知题型: {question_type}")
                return {"extraction_status": "unknown_type", "question_type": question_type,
                        "problem_number": problem_num}

            chain = self.llm.client
            result = chain.invoke(prompt)

            if result is None:
                print(f"LLM返回结果为空")
                return {"extraction_status": "failed", "question_type": question_type, "error": "LLM返回为空",
                        "problem_number": problem_num}

            # 提取JSON内容
            result_content = result.content
            print(f"LLM返回内容预览: {result_content[:200]}...")

            # 尝试提取JSON
            json_content = self._extract_json_from_text(result_content)

            if not json_content:
                # 尝试修复常见错误
                json_content = self._extract_json_from_text(self._fix_common_json_errors(result_content))

            if json_content:
                try:
                    parsed_result = json.loads(json_content)

                    # 验证数据格式
                    is_valid, error_msg = self._validate_extracted_data(parsed_result, question_type)

                    if not is_valid:
                        print(f"数据格式验证失败: {error_msg}")
                        # 如果是阅读理解，尝试高级提取
                        if question_type == "reading":
                            advanced_results = self.extract_reading_questions_advanced(content)
                            if advanced_results:
                                # 如果有多个结果，返回特殊格式
                                if len(advanced_results) > 1:
                                    return {
                                        "problem_number": problem_num,
                                        "question_type": "reading",
                                        "multiple_passages": True,
                                        "passages": advanced_results
                                    }
                                else:
                                    result = advanced_results[0]
                                    result["problem_number"] = problem_num
                                    return result

                        return {
                            "extraction_status": "invalid_format",
                            "question_type": question_type,
                            "error": error_msg,
                            "raw_data": parsed_result,
                            "problem_number": problem_num
                        }

                    # 标准化返回格式，添加问题编号
                    if question_type == "multiple_choice":
                        if isinstance(parsed_result, dict):
                            parsed_result["problem_number"] = problem_num
                            return {"questions": [parsed_result], "question_type": question_type,
                                    "problem_number": problem_num}
                        elif isinstance(parsed_result, list):
                            for item in parsed_result:
                                item["problem_number"] = problem_num
                            return {"questions": parsed_result, "question_type": question_type,
                                    "problem_number": problem_num}
                    else:
                        # 完型填空和阅读理解
                        if question_type == "reading" and isinstance(parsed_result, list):
                            # 处理返回列表的情况（多篇文章）
                            return {
                                "problem_number": problem_num,
                                "question_type": "reading",
                                "multiple_passages": True,
                                "passages": parsed_result
                            }
                        else:
                            # 单篇文章或完型填空
                            parsed_result["question_type"] = question_type
                            parsed_result["problem_number"] = problem_num
                            return parsed_result

                except json.JSONDecodeError as e:
                    print(f"JSON解析失败: {e}")
                    # 如果是阅读理解，尝试高级提取
                    if question_type == "reading":
                        advanced_results = self.extract_reading_questions_advanced(content)
                        if advanced_results:
                            if len(advanced_results) > 1:
                                return {
                                    "problem_number": problem_num,
                                    "question_type": "reading",
                                    "multiple_passages": True,
                                    "passages": advanced_results
                                }
                            else:
                                result = advanced_results[0]
                                result["problem_number"] = problem_num
                                return result
                    return self._extract_partial_data(result_content, question_type)
            else:
                print(f"无法从返回内容中提取JSON")
                # 如果是阅读理解，尝试高级提取
                if question_type == "reading":
                    advanced_results = self.extract_reading_questions_advanced(content)
                    if advanced_results:
                        if len(advanced_results) > 1:
                            return {
                                "problem_number": problem_num,
                                "question_type": "reading",
                                "multiple_passages": True,
                                "passages": advanced_results
                            }
                        else:
                            result = advanced_results[0]
                            result["problem_number"] = problem_num
                            return result
                return self._extract_partial_data(result_content, question_type)

        except Exception as e:
            print(f"提取过程发生错误: {e}")
            import traceback
            traceback.print_exc()
            return {"extraction_status": "error", "error": str(e), "question_type": question_type,
                    "problem_number": problem_num}

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        改进的JSON提取方法
        :param text: 包含JSON的文本
        :return: JSON字符串或None
        """
        # 1. 首先尝试直接解析整个文本
        try:
            json.loads(text)
            return text
        except Exception as e:
            pass

        # 2. 查找代码块中的JSON
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        code_blocks = re.findall(code_block_pattern, text)
        for block in code_blocks:
            try:
                json.loads(block)
                return block
            except Exception as e:
                continue

        # 3. 使用改进的正则表达式提取JSON
        # 处理嵌套的大括号和方括号
        json_candidates = []

        # 查找以 [ 或 { 开始的位置
        for start_char, end_char in [('[', ']'), ('{', '}')]:
            pos = 0
            while True:
                start = text.find(start_char, pos)
                if start == -1:
                    break

                # 计算括号平衡
                bracket_count = 0
                end = start
                for i in range(start, len(text)):
                    if text[i] == start_char:
                        bracket_count += 1
                    elif text[i] == end_char:
                        bracket_count -= 1

                    if bracket_count == 0:
                        end = i
                        break

                if bracket_count == 0:
                    candidate = text[start:end + 1]
                    try:
                        json.loads(candidate)
                        json_candidates.append(candidate)
                    except Exception as e:
                        pass

                pos = start + 1

        # 4. 返回最长的有效JSON
        if json_candidates:
            return max(json_candidates, key=len)

        return None

    def _fix_common_json_errors(self, json_str: str) -> str:
        """
        修复常见的JSON错误
        :param json_str: JSON字符串
        :return: 修复后的JSON字符串
        """
        # 修复未闭合的引号
        json_str = re.sub(r':\s*"([^"]*?)(?=[,}])', r': "\1"', json_str)

        # 修复末尾多余的逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # 确保属性名有引号
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)

        # 修复已经有引号的情况
        json_str = re.sub(r'""(\w+)"":', r'"\1":', json_str)

        return json_str

    def _extract_partial_data(self, content: str, question_type: str) -> Dict[str, Any]:
        """
        当JSON解析失败时，尝试提取部分数据
        :param content: 原始内容
        :param question_type: 题型
        :return: 部分提取结果
        """
        print("尝试部分数据提取...")

        if question_type == "cloze" or question_type == "reading":
            # 尝试提取文章内容
            passage_match = re.search(r'"reading_passage"\s*:\s*"([^"]+)"', content)
            passage = passage_match.group(1) if passage_match else ""

            # 尝试提取题目
            questions = []
            question_pattern = r'"question_id"\s*:\s*"([^"]+)".*?"options"\s*:\s*\[(.*?)\]'
            for match in re.finditer(question_pattern, content, re.DOTALL):
                try:
                    question_id = match.group(1)
                    options_str = match.group(2)
                    # 提取选项
                    options = re.findall(r'"([^"]+)"', options_str)
                    if options:
                        questions.append({
                            "question_id": question_id,
                            "options": options[:4]  # 确保只有4个选项
                        })
                except Exception as e:
                    print(f"错误：{e}")
                    continue

            if passage or questions:
                return {
                    "question_type": question_type,
                    "reading_passage": passage,
                    "questions": questions,
                    "extraction_status": "partial"
                }

        return {
            "extraction_status": "failed",
            "question_type": question_type,
            "error": "无法提取有效数据"
        }

    def _validate_extracted_data(self, data: Any, question_type: str) -> Tuple[bool, str]:
        """
        验证提取的数据格式是否正确 - 修改为支持阅读理解的列表格式
        :param data: 提取的数据
        :param question_type: 题型
        :return: (是否有效, 错误信息)
        """
        try:
            if question_type == "multiple_choice":
                # 选择题应该是列表或包含questions的字典
                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            return False, "选择题项目应该是字典格式"
                        if "question_id" not in item or "options" not in item:
                            return False, "选择题缺少必要字段(question_id, options)"
                    return True, ""
                elif isinstance(data, dict) and "questions" in data:
                    return self._validate_extracted_data(data["questions"], question_type)
                else:
                    return False, "选择题格式不正确"

            elif question_type == "cloze":
                # 完型填空应该包含reading_passage和questions
                if not isinstance(data, dict):
                    return False, "完型填空应该是字典格式"
                if "reading_passage" not in data or "questions" not in data:
                    return False, "完型填空缺少必要字段(reading_passage, questions)"
                if not isinstance(data["questions"], list):
                    return False, "完型填空的questions应该是列表"
                for q in data["questions"]:
                    if "question_id" not in q or "options" not in q:
                        return False, "完型填空题目缺少必要字段"
                return True, ""

            elif question_type == "reading":
                # 阅读理解可以是单个字典或字典列表
                if isinstance(data, list):
                    # 多篇文章的情况
                    for item in data:
                        if not isinstance(item, dict):
                            return False, "阅读理解项目应该是字典格式"
                        if "reading_passage" not in item or "questions" not in item:
                            return False, "阅读理解缺少必要字段(reading_passage, questions)"
                        if not isinstance(item["questions"], list):
                            return False, "阅读理解的questions应该是列表"
                        for q in item["questions"]:
                            if "question_id" not in q or "options" not in q:
                                return False, "阅读理解题目缺少必要字段"
                    return True, ""
                elif isinstance(data, dict):
                    # 单篇文章的情况
                    if "reading_passage" not in data or "questions" not in data:
                        return False, "阅读理解缺少必要字段(reading_passage, questions)"
                    if not isinstance(data["questions"], list):
                        return False, "阅读理解的questions应该是列表"
                    for q in data["questions"]:
                        if "question_id" not in q or "options" not in q:
                            return False, "阅读理解题目缺少必要字段"
                    return True, ""
                else:
                    return False, "阅读理解格式不正确"

            return False, f"未知题型: {question_type}"

        except Exception as e:
            return False, f"验证过程出错: {str(e)}"

    def extract_all_questions(self, exam_content: str) -> Dict[str, Any]:
        """
        改进的提取所有题目方法 - 按問題编号组织
        """
        # 预处理OCR文本
        exam_content = self.preprocess_ocr_text(exam_content)

        # 使用改进的分割方法
        chunks = self.split_question_improved(exam_content)

        all_questions = {
            "exam_info": {
                "extraction_date": datetime.now().isoformat(),
                "total_chunks": len(chunks),
                "content_length": len(exam_content),
            },
            "problems": {},  # 改为按問題编号存储
            "extraction_errors": [],
            "partial_extractions": []
        }

        for i, (problem_num, chunk) in enumerate(chunks):
            print(f"\n正在处理第 {i + 1} / {len(chunks)} 个文本块...")
            print(f"問題编号: {problem_num}")
            print(f"文本块内容预览: {chunk[:100]}...")

            # 识别题型
            question_type, confidence = self.identify_question_type(chunk)
            print(f"识别题型: {question_type} (置信度: {confidence})")

            # 提取题目
            extracted_data = self.extract_questions_by_type(chunk, question_type, problem_num)

            # 处理提取结果
            if "extraction_status" in extracted_data:
                status = extracted_data["extraction_status"]
                extracted_data["chunk_index"] = i + 1
                extracted_data["chunk_preview"] = chunk[:100]
                extracted_data["identified_type"] = question_type
                extracted_data["confidence"] = confidence
                extracted_data["problem_number"] = problem_num

                if status == "partial":
                    # 部分提取成功
                    all_questions["partial_extractions"].append(extracted_data)
                    # 仍然尝试保存部分数据
                    if problem_num not in all_questions["problems"]:
                        all_questions["problems"][problem_num] = extracted_data
                else:
                    # 完全失败
                    all_questions["extraction_errors"].append(extracted_data)
            else:
                # 成功提取
                if problem_num not in all_questions["problems"]:
                    all_questions["problems"][problem_num] = {
                        "problem_number": problem_num,
                        "question_type": question_type,
                        "content": []
                    }

                # 根据题型处理数据
                if question_type == "multiple_choice":
                    if "questions" in extracted_data:
                        all_questions["problems"][problem_num]["content"] = extracted_data["questions"]
                    elif isinstance(extracted_data, list):
                        all_questions["problems"][problem_num]["content"] = extracted_data
                    else:
                        all_questions["problems"][problem_num]["content"] = [extracted_data]

                elif question_type in ["cloze", "reading"]:
                    # 处理阅读理解的特殊情况
                    if question_type == "reading" and extracted_data.get("multiple_passages"):
                        # 如果有多篇文章
                        all_questions["problems"][problem_num] = {
                            "problem_number": problem_num,
                            "question_type": "reading",
                            "passages": extracted_data.get("passages", [])
                        }
                    else:
                        # 单篇文章或完型填空
                        all_questions["problems"][problem_num] = extracted_data

        # 更新统计信息
        self._update_statistics_by_problem(all_questions)

        return all_questions

    def _update_statistics_by_problem(self, all_questions: Dict[str, Any]):
        """按問題编号更新统计信息"""
        total_questions = 0
        problem_stats = {}

        for problem_num, problem_data in all_questions["problems"].items():
            if isinstance(problem_data, dict):
                if "content" in problem_data and isinstance(problem_data["content"], list):
                    # 选择题类型
                    count = len(problem_data["content"])
                    total_questions += count
                    problem_stats[problem_num] = {
                        "type": problem_data.get("question_type", "unknown"),
                        "question_count": count
                    }
                elif "questions" in problem_data and isinstance(problem_data["questions"], list):
                    # 完型填空或阅读理解
                    count = len(problem_data["questions"])
                    total_questions += count
                    problem_stats[problem_num] = {
                        "type": problem_data.get("question_type", "unknown"),
                        "question_count": count
                    }
                elif "passages" in problem_data:
                    # 多篇阅读理解
                    count = sum(len(p.get("questions", [])) for p in problem_data["passages"])
                    total_questions += count
                    problem_stats[problem_num] = {
                        "type": "reading",
                        "question_count": count,
                        "passage_count": len(problem_data["passages"])
                    }

        all_questions["exam_info"].update({
            "total_questions_extracted": total_questions,
            "extraction_errors_count": len(all_questions["extraction_errors"]),
            "partial_extractions_count": len(all_questions["partial_extractions"]),
            "problem_count": len(all_questions["problems"]),
            "problem_statistics": problem_stats
        })

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
        print(f"問題总数: {questions_data['exam_info']['problem_count']}")
        print(f"题目总数: {questions_data['exam_info']['total_questions_extracted']}")
        print(f"提取错误数量: {questions_data['exam_info']['extraction_errors_count']}")
        print(f"部分提取数量: {questions_data['exam_info']['partial_extractions_count']}")

        # 打印每个問題的详细统计
        print("\n=== 問題详细统计 ===")
        for problem_num, stats in questions_data['exam_info']['problem_statistics'].items():
            print(f"{problem_num}:")
            print(f"  题型: {stats['type']}")
            print(f"  题目数: {stats['question_count']}")
            if 'passage_count' in stats:
                print(f"  文章数: {stats['passage_count']}")

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
        result = extractor.process_exam(
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
