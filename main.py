import sandbox
from fastapi import FastAPI, UploadFile, HTTPException, Response
from typing import List, Any, Dict
import llm
import os
import pandas as pd
import numpy as np
from func_timeout import func_set_timeout, FunctionTimedOut
import ssl
from utils import extract_code
import re
import duckdb
from pathlib import Path
import json
import io
import faux
import agent
import contextlib
from iochain import IOBlock, IOChain
from dataclasses import dataclass
import logging

# basic monkey-patching
logging.basicConfig(level=logging.INFO)
ssl._create_default_https_context = ssl._create_unverified_context


def format_ns(ns) -> str:
    return " ".join(list(ns))

@func_set_timeout(20)
def evaluate_code(code, ns):
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with (
        contextlib.redirect_stdout(stdout_buffer),
        contextlib.redirect_stderr(stderr_buffer),
    ):
        sandbox.exec(
            code,
            {
                "pd": pd,
                "np": np,
                "ssl": ssl,
                "duckdb": duckdb,
            },
            ns,
        )

    stderr = stderr_buffer.getvalue()
    if stderr and "and thus cannot be shown" not in stderr:
        stdout = stdout_buffer.getvalue()
        # Extract the result (assuming it's the last evaluated expression)
        raise ValueError("WARNING: " + stderr + "\n\n" + stdout)

    if stdout := stdout_buffer.getvalue():
        return stdout


app = FastAPI()
model = llm.get_model("gpt-4o")
model.key = os.environ["OPENAI_API_KEY"]


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

    answer = faux.forge(question)
    tries = 3
    for _ in range(tries):
        try:
            answer = answer_attempt(question)
            break
        except FunctionTimedOut:
            continue

        if not isinstance(json.loads(answer), List):
            continue

    return Response(content=answer, media_type="application/json")



@func_set_timeout(90)
def answer_attempt(question):
    ns: Dict[str, Any] = {}
    io_chain = IOChain([])
    steps = must_breakdown(question)
    logging.info(steps)

    i = 0
    for i, step in enumerate(steps):
        try:
            code = llm_generate(step, io_chain)
        except Exception as e:
            logging.warn(f"answer_attempt {e}")
            continue
        print(f"<LLM>\n{code}\n</LLM>")
        try:
            block = IOBlock(code, evaluate_code(code, ns))
            io_chain.chain.append(block)
        except Exception as e:
            print("answer_attempt::exception ", e)
            rectify_loop(step, io_chain, code, e, ns)

        if len(io_chain.chain) > 0:
            print(io_chain.chain[-1].result)
            print(format_ns(ns))

    return io_chain.chain[-1].result


def rectify_code(objective: str, io_chain: IOChain, erroneous_code: str, exc: Exception) -> str:
    system = (
        agent.description + "Rectify your previous code according to the errors."
    )
    prompt = fr"""

Code that executed successfully so far:
```python
{io_chain}
```

Erroneous code:
```python
{erroneous_code}
```

```error
{exc}
```

Write <= 6 lines of Python to rectify the code in the last block.
If some information like a column name is ambiguous, try to find that out in this cell.
You will be given another chance to put it all together. Just add a trailing comment "# partial step".

"""
    response = model.prompt(prompt, system=system).text()
    code = extract_code(response)
    if code.count("\n") > 6:
        raise ValueError("LLM produced slop")
    return code

# @param inout ic_chain
def rectify_loop(objective: str, io_chain: IOChain, code: str, exc: Exception, ns) -> Dict[str, Any]:
    while True:
        logging.info("rectify_loop::entering")
        try:
            code = rectify_code(objective, io_chain, code, exc)
            code = f"""
# {objective}
{code}
        """
            logging.info(f"rectify_loop::llm {code}")
        except Exception as e:
            logging.warn(f"rectify_loop::exception {e}")
            continue

        try:
            result = evaluate_code(code, ns)
        except FunctionTimedOut:
            continue
        except Exception as e:
            exc = e
            logging.info(e)
            continue
        io_chain.chain.append(IOBlock(code, result))
        return ns

def llm_generate(instr: str, io_chain: IOChain, full_question: str = "") -> str:
    system = agent.description + "Implement the described function and call it.\n"
    if io_chain:
        system += f"""
Code from the last {io_chain.limit} cells:
{io_chain}
"""

    response = model.prompt(
f"""
Write <= 6 line of Python to perform the logic in this pseudocode statement. In case of ambiguity use print. You will be given multiple chances.

```python
# {instr}
""",
        system=system,
    ).text()
    code = extract_code(response)
    if code.count("\n") > 6:
        raise ValueError("LLM produced slop")
    code = f"""
# {instr}
{code}
"""
    return code

@dataclass
class Step:
    identifier: str
    function_call: str

    def __init__(self, step: str):
        match = re.match(r'(\w+)\s*=\s*(.*)', step)
        self.identifier = match.group(1)
        self.function_call = match.group(2)

    def __repr__(self) -> str:
        return f"{self.identifier} = {self.function_call}"


def breakdown(question: str) -> List[Step]:
    system = """
Break the following question down into a sequence of function calls and assignments in pseudocode.
Use links and filenames as verbatim. If an idea does not fit in a function call, break it down further.

<output-format>
df = scrape_table("https://doc.e.foundation/devices")
devices_col = find_column_containing(df, "devices")
plot = perform_scatterplot(devices_col)
</output-format>

Always follow the format `identifier = function_call(...)`
No comments. No imports. No multiline expression.
"""

    response = model.prompt(question, system=system).text()
    print(response)
    actions = []
    for line in response.splitlines():
        if not line or line.startswith("```"):
            continue
        try:
            step = Step(line)
        except Exception as e: # noqa
            print(e)
            return None
        actions.append(step)
    return actions

def must_breakdown(question: str) -> List[Step]:
    steps = None
    while steps is None:
        steps = breakdown(question)
    return steps

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# if all fails, send dummy json with the same schema
# keep a trace of all transactions
