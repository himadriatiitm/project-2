import sandbox
from fastapi import FastAPI, UploadFile, HTTPException, Response
from typing import List, Any, Dict, Callable
import llm
import os
import pandas as pd
import numpy as np
from func_timeout import func_set_timeout, FunctionTimedOut
import ssl
from utils import extract_code
import duckdb
from pathlib import Path
import json
import io
import faux
import agent
import contextlib
import logging

# basic monkey-patching
logging.basicConfig(level=logging.INFO)
ssl._create_default_https_context = ssl._create_unverified_context


def format_ns(ns) -> str:
    return " ".join(list(ns))

app = FastAPI()
model = llm.get_model("gpt-4o")
model.key = os.environ.get("OPENAI_API_KEY") or "electric-boogaloo"


@app.post("/api/")
async def upload_file(file: List[UploadFile]):
    """
    Accepts a text file on the /api/ route and returns its content.
    """
    question = None
    try:
        for handle in file:
            filename = Path(handle.filename).name
            if filename == "question.txt":
                question = (await handle.read()).decode()
                continue
            Path(filename).write_bytes(await handle.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error reading file: {str(e)}")

    if not question:
        return {"message": "that's a stupid request."}

    answer = faux.forge(question, model)
    tries = 3
    for _ in range(tries):
        try:
            answer = answer_attempt(question)
            break
        except FunctionTimedOut:
            continue

    return Response(content=answer, media_type="application/json")

imports = {
    "pd": pd,
    "np": np,
    "ssl": ssl,
    "duckdb": duckdb,
}


@func_set_timeout(90)
def answer_attempt(question):
    ns: Dict[str, Any] = {}

    code = must(breakdown, question)
    logging.info(code)
    result = sandbox.exec(code, imports, ns)
    print(result)

    while not faux.list_verify(faux.json_verify(result)):
        code = must(rectify, question, code, result)
        logging.info(code)
        result = sandbox.exec(code, imports, ns)
        print(result)

    return result


def rectify(objective: str, code: str, exc: Exception) -> str:
    system = (
        agent.description + "Rectify your previous code according to the errors."
    )
    prompt = fr"""
objective:
```
{objective}
```

```python
{code}
```

```output
{exc}
```

Rectify the code in the last block. In case of doubt, `print`. You will be given multiple chances to put it all together.
"""

    print(f"<to-llm>\n\n{system}\n\n{prompt}\n\n</to-llm>")

    response = model.prompt(prompt, system=system).text()
    return extract_code(response)


def breakdown(question: str) -> str:
    system = agent.description + """
Write python code to solve the following data science question.
Always `print` the final JSON serialized list.
"""

    response = model.prompt(question, system=system).text()
    return extract_code(response)

def must(f: Callable, *args, **kwargs):
    while True:
        if v := f(*args, **kwargs):
            return v

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# if all fails, send dummy json with the same schema
# keep a trace of all transactions
