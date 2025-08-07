from fastapi import FastAPI, UploadFile, HTTPException, Response
from typing import List, Any, Dict
import llm
import os
import pandas as pd
import numpy as np
from func_timeout import func_set_timeout,FunctionTimedOut
import ssl
import duckdb
from pathlib import Path
import re

ssl._create_default_https_context = ssl._create_unverified_context

def format_ns(ns) -> str:
    return ' '.join(list(ns))

GOG = ""

def printer(*args):
    global GOG
    GOG += ' '.join(map(str, args))

@func_set_timeout(20)
def evaluate_code(code, local_namespace):
    global GOG
    GOG = ""
    exec(code, {'pd': pd, 'np': np, 'ssl': ssl, 'duckdb': duckdb, 'print': printer}, local_namespace)

    # Extract the result (assuming it's the last evaluated expression)
    if GOG:
        return GOG

    result_id = list(local_namespace)[-1]
    return local_namespace[result_id]

app = FastAPI()
model = llm.get_model("gpt-4o-mini")
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
            if filename == 'question.txt':
                question = (await handle.read()).decode()
                continue
            Path(filename).write_bytes(await handle.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error reading file: {str(e)}")

    if not question:
        return {"message":"that's a stupid request."}
    
    tries = 5
    answer = None
    for truh in range(tries):
        try:
            answer = answer_attempt(question)
            break
        except FunctionTimedOut:
            continue

    return Response(content=answer, media_type='application/json')
    return answer

@func_set_timeout(180)
def answer_attempt(question):
    local_namespace: Dict[str, Any] = {}
    # I swear I hate this god awful snake language.
    # Take me back to C.
    code: str | None = None
    last_known_good_code: str = ""
    result: Any | Exception = None
    steps = breakdown(question)
    i = 0
    while i < len(steps):
        instr = steps[i]
        ok = False
        while not ok:
            try:
                if isinstance(result, Exception):
                    code = rectify_code(instr, truncate_string(result), last_known_good_code, code)
                else:
                    code = broad_eval(question, instr, list(local_namespace), last_known_good_code, truncate_string(result))
            except Exception as e:
                print(e)
                continue
            print("== LLM says " + '=' * 68)
            print(code)
            print('=' * 80)
            result = None
            try:
                result = evaluate_code(code, local_namespace)
            except FunctionTimedOut:
                result = "<timeout />"
            except Exception as e:
                result = e
                continue

            if wanna_backtrack(result, local_namespace, code):
                # backtrack
                i -= 2

            ok = True
            last_known_good_code += "\n" + code

            if match := re.match(r'(\w+)\s*=\s*json.dumps', code):
                result_id = match.group(1)
                return local_namespace[result_id]

            print(truncate_string(str(result)))
            print(format_ns(local_namespace))
        i += 1

    return result
    

def between_rope(string: str, rope_start: str, rope_end: str = None) -> str:
    rope_end = rope_end or rope_start
    start = string.index(rope_start) + len(rope_start)
    string = string[start:]
    end = string.index(rope_end)
    return string[:end]
    

def between_tags(string: str, rope: str) -> str:
    rope_start = f"<{rope}>"
    rope_end = f"</{rope}>"
    return between_rope(string, rope_start, rope_end)

def extract_code(code: str) -> str:
    code = between_rope(code, "```")
    if code.startswith("python"):
        code = code[len("python"):]
    return code

def truncate_string(s: Any, length = 200) -> str:
    s = str(s)
    if len(s) > length:
        return s[:length-4] + "..." + s[-4:]
    return s

def rectify_code(objective: str, result: str, prev_cell: str, last_code: str) -> str:
    system = (
        "You are a senior Python developer with a lot of experience in data science frameworks.\n"
        "You use succinct code that gets the job done without relying too much on extraneous libraries.\n"
        "You do not write comments. Comments are for the weak. There is only one correct way to solve a given problem. - The Zen of Python\n\n"
        "Rectify your previous code according to the errors. Each step is exploratory so you decide to print and examine the outputs when needed."
    )
    prompt = f"""
Code that executed successfully so far:
```python
{prev_cell}
```

Code:
```python
# {objective}
{last_code}
```

Error:
```
{result}
```
"""
    response = model.prompt(prompt, system=system).text()
    return extract_code(response)

def wanna_backtrack(result: Any, ns: List[str], last_code) -> str:
    system = """
Make sure that the variables you created are actually present in the local namespace.
Respond with
- YES: to proceed to then ext step.
- BACKTRACK: to go back to a previous step.
"""
    result = truncate_string(str(result), 200)
    ns = format_ns(ns)
    proompt = f"""
Your code:
```python
{last_code}
```

<output>
{result}
</output>

<locals>{ns}</locals>
"""
    response = model.prompt(proompt, system=system).text()
    return "BACKTRACK" in response

def broad_eval(question:str, instr: str, ns: List[str], last_code, result) -> str:
    system = (
        "You are a senior Python developer with a lot of experience in data science frameworks.\n"
        "You use succinct code that gets the job done without relying too much on extraneous libraries.\n"
        "You do not write comments. Comments are for the weak. You do not use magic numbers or magic literals.\n"
        "You write and examine code one step at a time, starkly contrasing the haste of junior developers who try to solve a problem in one go.\n"
        "There is only one correct way to solve a given problem. - The Zen of Python\n\n"
        "Write Python code for the described step. Each step is exploratory so you decide to print and examine the outputs when needed."
    )
    if ns:
        ns = format_ns(ns)
        system += (
            "These local variables exist from your previous run.\n"
            f"<locals>{ns}</locals>"
        )
    if last_code:
        system += f"""Code executed so far:
```python
{last_code}
```

output:
```
{result}
```
"""
    response = model.prompt("```python\n# " + instr, system=system).text()
    return extract_code(response)


def breakdown(question: str) -> List[str]:
    system = """Break the following question down into a list of steps to be taken in Python.
Use links and filenames as verbatim. Describe each step in a single line.
If the step does not fit in a line, break it down further. Answer only the list of steps so that our API can ingest them with ease.

Example:

1. Scrape the table from https://doc.e.foundation/devices
2. Read the image `phone.png` into variable `img` with PIL
"""

    response = model.prompt(question, system=system).text()
    return response.splitlines()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# if all fails, send dummy json with the same schema
# keep a trace of all transactions
