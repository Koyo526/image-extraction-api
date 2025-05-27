from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
import random
import base64
import json
import urllib

from utils.openai_client import client, GPT_MODEL


router = APIRouter()

@router.get("/check-gpt")
async def checkGPT():
    completion = client.chat.completions.create(
        model = GPT_MODEL,
        messages=[{
            "role": "user",
            "content": "Write a one-sentence bedtime story about a unicorn."
        }]
    )
    return {"result": completion.choices[0].message.content}
