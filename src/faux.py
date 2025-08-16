import logging
import json
from typing import List, Any
from textwrap import dedent
import re

JSON_EXPRESSION = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def forge(question: str, prompt_fn) -> List[Any] | None:
    system = dedent("""\
        Generate the most believable output. It will always be a JSON array.

        Rules:
        - Infer the datatype of each answer.
        - How many (count) should be a Number.
        - Percentages, correlations, and the like must be floating point `Number`s.
        - For base64 images, output the dummy string "$plot".""")

    response = prompt_fn(question, system=system)
    logging.info(response)
    return json_verify(response)


def extract_json(text: str) -> str:
    if m := JSON_EXPRESSION.search(text):
        text = m.group(1)
    return text.strip()


def json_verify(response: str):
    try:
        response = extract_json(response)
    except (ValueError, IndexError):
        return None
    try:
        return json.loads(response)
    except Exception as e:
        logging.error(e)
        return None
