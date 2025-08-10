import sys
from pprint import pprint
import traceback
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
import tools
import io
import inspect
import contextlib
from iochain import IOBlock, IOChain
from dataclasses import dataclass

ssl._create_default_https_context = ssl._create_unverified_context


def format_ns(ns) -> str:
    return " ".join(list(ns))

# AVAILABLE_FUNCTIONS = ""

# for fn in tools.__all__:
#     doc = inspect.getdoc(fn)
#     sig = str(inspect.signature(fn))
#     name = fn.__name__
#     AVAILABLE_FUNCTIONS += f"""
# tools.{name}{sig}
# {doc}
# """

# AVAILABLE_FUNCTIONS = f"""
# <available-functions>
# {AVAILABLE_FUNCTIONS}
# </available-functions>
# """

# print(AVAILABLE_FUNCTIONS)

AGENT_SEED = (
    "You are a senior Python developer with a lot of experience in data science frameworks.\n"
    "You use succinct code that gets the job done without relying too much on extraneous libraries.\n"
    "Never write comments. There is only one correct way to solve a given problem. - The Zen of Python\n\n"
)

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
                "tools":tools,
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

    if result_id := next(iter(ns)):
        return ns[result_id]


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

    tries = 5
    answer = None
    for truh in range(tries):
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
    # I swear I hate this god awful snake language.
    # Take me back to C.
    io_chain = IOChain([])
    steps = must_breakdown(question)
    print(steps)

    i = 0
    for i, step in enumerate(steps):
        try:
            code = llm_generate(step, io_chain)
        except Exception as e:
            print("answer_attempt::warning ", e)
            continue
        print(f"<LLM>\n{code}\n</LLM>")
        try:
            block = IOBlock(code, evaluate_code(code, ns))
            io_chain.chain.append(block)
        except Exception as e:
            print("answer_attempt::exception ", e)
            ns = rectify_loop(step, io_chain, code, e)

        if len(io_chain.chain) > 0:
            print(io_chain.chain[-1].result)
            print(format_ns(ns))

    return io_chain.chain[-1].result


def rectify_code(objective: str, io_chain: IOChain, erroneous_code: str, exc: Exception) -> str:
    system = (
        AGENT_SEED + "Rectify your previous code according to the errors."
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

Write <= 6 lines of Python to rectify your previous code.
If some information like a column name is ambiguous, try to find that out in this cell.
You will be given another chance to put it all together. Just add a trailing comment "# partial step".

"""
    # + AVAILABLE_FUNCTIONS
    response = model.prompt(prompt, system=system).text()
    code = extract_code(response)
    if code.count("\n") > 6:
        raise ValueError("LLM produced slop")
    return code

# @param inout ic_chain
def rectify_loop(objective: str, io_chain: IOChain, erroneous_code: str, exc: Exception) -> Dict[str, Any]:
    while True:
        ns = {}
        try:
            code = rectify_code(objective, io_chain, erroneous_code, exc)
            code = f"""
        # {objective}
        {code}
        """
        except Exception as e:
            print("while asking llm to rectify code: ", e)
            continue
        try:
            result = evaluate_code(io_chain.dump_with(code), ns)
        except FunctionTimedOut:
            result = "<timeout />"
        except Exception as e:
            exc = e
            objective = code
            continue
        io_chain.chain.append(IOBlock(code, result))
        return ns

def llm_generate(instr: str, io_chain: IOChain, full_question: str = "") -> str:
    system = AGENT_SEED + "Implement the described function and call it.\n"
    if io_chain:
        system += f"""
Code from the last {io_chain.limit} cells:
{io_chain}
"""

    response = model.prompt(
f"""
Write <= 6 line of Python to perform the logic in this pseudocode statement.
If some information like a column name is ambiguous, try to find that out in this cell.
You will be given another chance to put it all together. Just add a trailing comment "# partial step".

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
        if not line:
            continue
        if line.startswith("```"):
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
