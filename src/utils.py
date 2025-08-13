from typing import Any
from textwrap import dedent
import re

CODE_EXPRESSION = re.compile(r"```(?:python|py)?\s*([\s\S]*?)```", re.IGNORECASE)

def extract_code(text: str) -> str:
    if m := CODE_EXPRESSION.search(text):
        text = m.group(1)
    return text.strip()

def truncate_string(s: Any, length=200) -> str:
    s = str(s)
    if len(s) > length:
        return s[: length - 4] + "..." + s[-4:]
    return s

def s(a: str, *args) -> str:
    return dedent(a).strip().format(*args).strip()
