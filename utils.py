from typing import Any

def between_rope(string: str, rope_start: str, rope_end: str = None) -> str:
    rope_end = rope_end or rope_start
    start = string.index(rope_start) + len(rope_start)
    string = string[start:]
    end = string.index(rope_end)
    return string[:end]


def extract_code(code: str) -> str:
    code = between_rope(code, "```")
    return code.removeprefix("python").strip()

def truncate_string(s: Any, length=200) -> str:
    s = str(s)
    if len(s) > length:
        return s[: length - 4] + "..." + s[-4:]
    return s
