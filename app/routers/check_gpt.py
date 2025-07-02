from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
import random
import base64
import json
import urllib

from utils.bedrock_client import invoke_nova_lite_text


router = APIRouter()

@router.get("/check-gpt")
async def checkGPT():
    prompt = "Write a one-sentence bedtime story about a unicorn."
    result = invoke_nova_lite_text(prompt)
    return {"result": result}
