import sandbox
from fastapi import FastAPI, Request, HTTPException
from typing import List, Any, Callable
import llm
import networkx as nx
import os
import pandas as pd
import numpy as np
from func_timeout import func_set_timeout, FunctionTimedOut
import ssl
from utils import extract_code, s
import duckdb
from pathlib import Path
import faux
import agent
import structlog


# top-level monkey-patching
logging = structlog.get_logger()
ssl._create_default_https_context = ssl._create_unverified_context

def format_ns(ns) -> str:
    return " ".join(list(ns))


app = FastAPI()
model = llm.get_model("gpt-4o")
model.key = os.environ.get("OPENAI_API_KEY")
tries = int(os.environ.get("MAX_TRIES", 3))


# TODO: keep a trace of all transactions
@app.post("/api/")
async def upload_file(request: Request):
    form_data = request.form()
    question = None
    try:
        for form_filename, in_file in form_data.items():
            name = Path(form_filename).name
            if name in ('questions.txt', 'question.txt'):
                question = await in_file.read().decode()
            save_to = Path(name)
            save_to.write_bytes(await in_file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error reading file: {str(e)}")

    if not question:
        return {"message": "next time, come up with a question I can answer."}

    answer = must(faux.forge, question, model)
    for _ in range(tries):
        try:
            answer = answer_attempt(question)
            break
        except FunctionTimedOut:
            continue

    return answer


class Platypus:
    def __init__(self, question: str):
        self.generated = False
        self.question = question
        self.code = None
        self.imports = {
            "pd": pd,
            "np": np,
            "ssl": ssl,
            "duckdb": duckdb,
            "nx": nx,
        }
        self.ns = dict()
        self.result = None

    def __call__(self):
        f = rectify if self.generated else generate
        self.generated = True
        self.code = must(f, self.question, self.code, self.result)
        logging.info(self.code)
        self.result = sandbox.exec(self.code, self.imports, self.ns)
        logging.info("llm generated code executed", result=self.result)
        return last_line_json_list(self.result)


@func_set_timeout(90)
def answer_attempt(question):
    return must(Platypus(question))


def last_line_json_list(result: str) -> List[Any]:
    lines = result.strip().splitlines()
    if len(lines) == 0:
        return None
    line = lines[-1]
    if any((badbad in line.lower() for badbad in ("nan", "not found"))):
        return None
    return faux.list_verify(faux.json_verify(line))


def rectify(objective: str, code: str, exc: str) -> str:
    system = agent.description + "Rectify your previous code according to the errors."
    prompt = s(
        """
        objective:
        ```
        {}
        ```

        ```python
        {}
        ```

        ```output
        {}
        ```

        Rectify the code in the last block.
        In case of doubt, `print`. You will be given multiple attempts.
        Always `print(json.dumps(...))` the final list. It may contain int, float and str.
        """,
        objective,
        code,
        exc,
    )

    logging.info("to llm", system=system, prompt=prompt)
    response = model.prompt(prompt, system=system).text()
    return extract_code(response)


def generate(question: str, *args) -> str:
    system = agent.description + s("""\
        Write python code to solve the following data science question.
        Always `print(json.dumps(...))` the final list. It may contain int, float and str.
        """)

    response = model.prompt(question, system=system).text()
    return extract_code(response)


def must(f: Callable, *args, **kwargs):
    while True:
        if v := f(*args, **kwargs):
            return v

def main():
    if not model.key:
        logging.error("OPENAI_API_KEY is unset")
        return

    jail_path = Path("./jail")
    jail_path.mkdir(exist_ok=True)
    os.chdir(jail_path)
    logging.info("changed current directory", path=jail_path)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
