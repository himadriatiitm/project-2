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
from copy import deepcopy as copy
from pydantic import BaseModel

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

def copycopy(a):
    b = {}
    for a_i in a:
        try:
            b[a_i] = copy(a_i)
        except TypeError:
            b[a_i] = a[a_i]
    return b

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
                question += "\n"
                question += "Finally, store the output list in the identifier `result_list`."
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

class LastKnownGood(BaseModel):
    namespace: Dict[str, Any]
    code: str

@func_set_timeout(180)
def answer_attempt(question):
    local_namespace: Dict[str, Any] = {}
    # I swear I hate this god awful snake language.
    # Take me back to C.
    last_known_good = LastKnownGood(namespace={}, code = "")
    code: str = ""
    result: Any | Exception = None
    while 'result_list' not in local_namespace:
        try:
            code = broad_eval(question, list(local_namespace), last_known_good.code, truncate_string(result))
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
            print(e)
            # we messed up, reset namespace
            local_namespace = last_known_good.namespace
            last_known_good.namespace = copycopy(local_namespace)
            continue
        last_known_good.code += "\n" + code
        last_known_good.namespace = copycopy(local_namespace)

        if match := re.match(r'(\w+)\s*=\s*json.dumps', code):
            result_id = match.group(1)
            return local_namespace[result_id]

        print(truncate_string(str(result)))
        print(format_ns(local_namespace))

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
Code that executed successfully so far, not necessarily correct:
```python
{prev_cell}
```

Code you executed this time:
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

def broad_eval(question:str, ns: List[str], last_code, result) -> str:
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
    response = model.prompt(f"{question}\n```python\n# Just a single step. Preferably < 5 lines of code. Always inspect output with print in the end.", system=system).text()
    return extract_code(response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# if all fails, send dummy json with the same schema
# keep a trace of all transactions
