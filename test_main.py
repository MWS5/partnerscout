from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/ping")
def ping():
    return {"status": "pong", "service": "partnerscout", "port": os.environ.get("PORT", "not_set")}
