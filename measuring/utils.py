from enum import Enum


class AnswerGroup(Enum):
    """Does the model answer agree with the original knowledge, context knowledge, or other?"""

    ORIGINAL = 0
    CONTEXT = 1
    OTHER = 2
