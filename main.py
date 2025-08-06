from fastapi import FastAPI, UploadFile, HTTPException
from typing import List, Any
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

@func_set_timeout(20)
def evaluate_code(code, local_namespace):
    """
    Evaluates a given Python code string in a separate thread and stores the state.

    Args:
        code_string: The Python code string to execute.

    Returns:
        A dictionary containing the result and state of the evaluation.
        Returns None if an error occurs.
    """
    try:
        # Execute the code in a local namespace
        exec(code, {'pd': pd, 'np': np, 'ssl': ssl, 'duckdb': duckdb}, local_namespace)

        # Extract the result (assuming it's the last evaluated expression)
        result_id = list(local_namespace.keys())[-1]
        # Store the result and state
        return local_namespace[result_id]

    except Exception as e:
        return e

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

    print(answer)
    return answer

@func_set_timeout(180)
def answer_attempt(question):
    local_namespace = {}
    # I swear I hate this god awful snake language.
    # Take me back to C.
    code = None
    for instr in breakdown(question):
        ok = False
        while not ok:
            try:
                code = broad_eval(question, instr, list(local_namespace.keys()), code)
            except Exception as e:
                print(e)
                continue
            print("== LLM says " + '=' * 68)
            print(code)
            print('=' * 80)
            result = None
            ok = True
            try:
                result = evaluate_code(code, local_namespace)
            except FunctionTimedOut:
                result = "<timeout />"
            except Exception as e:
                result = f"""<error>
{e}
</error>"""

            ok = proceed_consent(result, list(local_namespace.keys()), code)

            if match := re.match(r'(\w+)\s*=\s*json.dumps', code):
                result_id = match.group(1)
                return local_namespace[result_id]

            print(truncate_string(str(result)))
            print(truncate_string(str(local_namespace.keys())))

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

def truncate_string(s: str) -> str:
    if len(s) > 80:
        return s[:76] + "..." + s[-4:]
    return s

def proceed_consent(result: Any, ns: List[str], last_code) -> str:
    system = "Respond with a YES if you wish to proceed with the current state of the code."
    result = truncate_string(str(result))
    ns = truncate_string(str(ns))
    proompt = f"""
Your code:
```python
{last_code}
```

<output>
{result}
</output>

<locals>
{ns}
</locals>
"""
    response = model.prompt(proompt, system=system).text()
    return "YES" in response

def broad_eval(question:str, instr: str, ns: List[str], last_code) -> str:
    system = f"""
<question>
{question}
</question>

Write Python code for the described step. DO NOT USE PLACEHOLDERS LIKE "YOUR CODE HERE". I am not coding, YOU are coding.
You will receive the output of the code in <output></output> tags or a <timeout /> tag."""
    if ns:
        system += f"""These local variables exist from your previous run.
<locals>{ns}</locals>
"""
    if last_code:
        system += f"""Code you had executed in the last cell:
```python
{last_code}
```
"""
    response = model.prompt("ONLY THE CODE FOR THIS STEP\n\n\n" +  '\n'.join([instr] * 16), system=system).text()
    return extract_code(response)


def breakdown(question: str) -> List[str]:
    system = """Break the following question down into a list of steps to be taken in Python.
Use links and filenames as verbatim. Describe each step in a single line.
If the step does not fit in a line, break it down further. Answer only the list of steps so that our API can ingest them with ease."""

    response = model.prompt(question, system=system).text()
    return response.splitlines()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# if all fails, send dummy json with the same schema
# keep a trace of all transactions
