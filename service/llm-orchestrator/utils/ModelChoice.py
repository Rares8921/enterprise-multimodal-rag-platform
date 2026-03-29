from enum import Enum

class ModelChoice(str, Enum):
    GEMINI = "gemini"
    MISTRAL = "mistral"
    AUTO = "auto"
