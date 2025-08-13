import logging
import json
from typing import List, Any
from textwrap import dedent

def forge(question: str, model) -> List[Any] | None:
    system = dedent("""\
        Generate the most believable output. It will always be a JSON array.

        Rules:
        - Infer the datatype of each answer.
        - How many (count) should be a Number.
        - Percentages, correlations, and the like must be floating point `Number`s.
        - For base64 images, output the dummy string "$plot".

        FORMAT:

        ```json
        [valueA, valueB, ...]
        ```
        """)
    response = model.prompt(question, system=system).text()
    logging.info(response)
    return list_verify(json_verify(response))

def json_verify(response: str):
    try:
        response = response[response.index('['):]
        response = response[:response.rindex(']')+1]
    except (ValueError,IndexError):
        return None
    try:
        return json.loads(response)
    except Exception as e:
        logging.error(e)
        return None

def list_verify(v):
    if isinstance(v, List):
        return v
