from fastapi import FastAPI, Request
from NLPManager import NLPManager
import time

app = FastAPI()

nlp_manager = NLPManager()

@app.middleware('http')
async def log_request_duration(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    print(f"request took: {process_time} ms")
    return response

@app.get("/health")
def health():
    return {"message": "health ok"}


@app.post("/extract")
async def extract(instance: Request):
    """
    Performs QA extraction given a context string

    returns a dictionary with fields:

    {
        "heading": str,
        "target": str,
        "tool": str,
    }
    """
    # get transcription, and pass to NLP model
    request_dict = await instance.json()

    predictions = []
    for instance in request_dict["instances"]:
        # each is a dict with one key "transcript" and the transcription as a string
        answers = nlp_manager.qa(instance["transcript"])
        predictions.append(answers)

    return {"predictions": predictions}
