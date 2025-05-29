from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
import random
import base64
import json
import urllib
import requests

from utils.openai_client import client, GPT_MODEL


# Openapi の API に対するリクエストとレスポンスの定義
class ImageRequest(BaseModel):
    image_base64: str
    outing_purpose_id: int   # 0:職場, 1:デート, 2:買い物, 3:大学, 4:カフェ, 5:飲み会（同性のみ）, 6:飲み会, 7:運動, 9:特になし
class ImageResponse(BaseModel):
    coordinate_review: str
    coordinate_item01: str
    recommend_item01: str
    coordinate_item02: str
    recommend_item02: str
    coordinate_item03: str
    recommend_item03: str
    gender: str


class CoordinateResponse(BaseModel):
    id: int
    coordinate_review: str
    coordinate_item01: str
    recommend_item01: str
    recommend_item01_url: str
    coordinate_item02: str
    recommend_item02: str
    recommend_item02_url: str
    coordinate_item03: str
    recommend_item03: str
    recommend_item03_url: str


router = APIRouter()

@router.post("/coordinate-review", response_model = CoordinateResponse)
def coordinateReview(request: ImageRequest):
    gender = "men"   # TODO: 動的化

    system_prompt = """
    あなたはプロのファッションコーディネータです。
    豊富な経験と鋭い観察力で、与えられたコーデ画像をもとに、ユーザーの服装を分析し、着こなし方や色の組み合わせ、シルエットやサイズ感など、的確で具体的なアドバイスを提供します。
    また、トレンド感のあるアイテムやスタイル提案にも精通しており、コーデ画像から季節感を考慮し、ユーザーの魅力を引き出す最適なコーディネートを提案できます。

    ### 制約条件
    ・口調は親近感を持つようなアドバイスしたりする口調にしてください。
    ・基本的にはポジティブな印象を与え、coordinate_reviewの最後はワンポイントアドバイスをお願いします。
    ・出力形式は必ず守ってください。
    ・出力に <> は含めないでください
    """

    if request.outing_purpose_id == 0:
        prompt_filename = "business.txt"
    elif request.outing_purpose_id == 1:
        prompt_filename = "couple.txt"
    elif request.outing_purpose_id == 2:
        prompt_filename = "shopping.txt"
    elif request.outing_purpose_id == 3:
        prompt_filename = "school.txt"
    elif request.outing_purpose_id == 4:
        prompt_filename = "cafe.txt"
    elif request.outing_purpose_id == 5:
        prompt_filename = "dining.txt"
    elif request.outing_purpose_id == 6:
        prompt_filename = "diningWithOppositeSex.txt"
    elif request.outing_purpose_id == 7:
        prompt_filename = "excercise.txt"
    elif request.outing_purpose_id == 8:
        prompt_filename = "nothing.txt"
    else:
        prompt_filename = "nothing.txt"

    with open(f"prompt/{prompt_filename}", "r", encoding="utf-8") as file:
        prompt = file.read()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{request.image_base64}"
                        }
                    }
                ]
            }
        ],
        response_format = {"type": "json_object"},
        max_tokens = 2048,
    )
    print(response.choices[0].message.content)

    openAIResponse = response.choices[0].message.content
    openAIResponseJSON = json.loads(openAIResponse)
    imageResponse = ImageResponse(**openAIResponseJSON)
    # gender が men or women 出ない場合は men にする
    if imageResponse.gender == "men" or imageResponse.gender == "women":
        imageResponse.gender = "men"


    coordinate_item02: str
    recommend_item02: str
    recommend_item02_url: str
    coordinate_item03: str
    recommend_item03: str
    recommend_item03_url: str

    coordinateResponse = CoordinateResponse(
        id = random.randrange(10**10),

        coordinate_review = imageResponse.coordinate_review,   # coordinate_review

        coordinate_item01 = imageResponse.coordinate_item01,   # coordinate_item01
        recommend_item01 = imageResponse.recommend_item01,   # recommend_item01
        recommend_item01_url = f"https://zozo.jp/search/?sex={imageResponse.gender}&p_keyv={urllib.parse.quote(imageResponse.recommend_item01, encoding='shift_jis')}",   # recommend_item01_url

        coordinate_item02 = imageResponse.coordinate_item02,
        recommend_item02 = imageResponse.recommend_item02,
        recommend_item02_url = f"https://zozo.jp/search/?sex={imageResponse.gender}&p_keyv={urllib.parse.quote(imageResponse.recommend_item02, encoding='shift_jis')}",

        coordinate_item03 = imageResponse.coordinate_item03,
        recommend_item03 = imageResponse.recommend_item03,
        recommend_item03_url = f"https://zozo.jp/search/?sex={imageResponse.gender}&p_keyv={urllib.parse.quote(imageResponse.recommend_item03, encoding='shift_jis')}"
    )
    print(coordinateResponse)

    return coordinateResponse
