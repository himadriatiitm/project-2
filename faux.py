import logging
import json
from typing import List, Any

def _forge(question: str, model) -> List[Any] | None:
    system = """
Generate the most believable output. It will always be a JSON array.

```json
[valueA, valueB, ...]
```

Rules:
- Infer the datatype of each answer.
- How many (count) should be a Number.
- Percentages, correlations, and the like must be floating point `Number`s.
- For base64 images, output `$plot` without any quotes.
"""
    response = model.prompt(question, system=system).text()
    response = response[response.index('['):]
    response = response[:response.rindex(']')+1]
    logging.info(response)
    try:
        faux = json.loads(response)
    except Exception as e:
        logging.error(e)
        return None

    if isinstance(faux, List):
        return faux
    return None

def forge(question: str, model) -> List[Any]:
    while True:
        if faux := _forge(question, model):
            return faux
