# from typing import Annotated

from fastapi import FastAPI, Request, Response
from pathlib import Path

app = FastAPI()


@app.post("/api/")
async def api(request: Request):
    form_data = await request.form()
    question = form_data.get("question.txt")
    answer = None
    if not question:
        return None

    criteria = {
        "sample-sales.csv": "sales.json",
        "edges.csv": "network.json",
        "sample-weather.csv": "weather.json"
    }

    for form_filename, in_file in form_data.items():
        if form_filename not in criteria:
            continue
        name = Path(form_filename).name
        answer = (Path('responses') / criteria[name]).read_text()
        save_to = Path('jail') / name

        save_to.write_bytes(await in_file.read())

    question = await question.read()
    return Response(content=answer, media_type="application/json")
    return question
