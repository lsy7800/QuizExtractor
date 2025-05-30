from enum import Enum
from typing import Union, Optional
from pydantic import BaseModel


class QuestionType(str, Enum):
    """定义题型"""
    MULTIPLE_CHOICE = "选择题"
    CLOZE_TEST = "完型填空题"
    READING_COMPREHENSION = "阅读理解"


class BaseQuestion(BaseModel):
    """题目属性"""
    question_number: str  # 编号
    question_type: QuestionType  # 类型
    question_text: str  # 问题
    difficulty: int = 1  # 难度


class MultiChoiceQuestion(BaseQuestion):
    """
    单项选择
    """
    options: list[str]  # 选项
    correct_answer: str  # 答案


class CLOZEQuestion(BaseQuestion):
    """
    完型填空
    """
    passage: str  # 阅读材料
    sub_question: list[Union[MultiChoiceQuestion]]  # 子问题


class ReadingComprehensionQuestion(BaseQuestion):
    """阅读理解"""
    passage: str  # 阅读材料
    sub_questions: list[Union[MultiChoiceQuestion]]  # 子问题


Question = Union[MultiChoiceQuestion, CLOZEQuestion, ReadingComprehensionQuestion]
