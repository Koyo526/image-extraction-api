from pydantic import BaseModel
from typing import List

class ReviewResponse(BaseModel):
    createdAt: str
    tops_image_url: str
    bottoms_image_url: str
    coordinate: dict
    graph_image: str
    recommendations: List[dict]

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


# -------------------------------
# 入出力モデル
# -------------------------------
class QueryInput(BaseModel):
    image_base64: str  # 入力画像のbase64文字列
    query_width: int = None    # （任意）中央画像の希望横幅（ピクセル）
    query_height: int = None   # （任意）中央画像の希望縦幅（ピクセル）

class SimilarWearItem(BaseModel):
    username: str
    post_url: str   # 追加：投稿へのURLを含める
    image_url: str

class PredictResponse(BaseModel):
    graph_image: str
    similar_wear: List[SimilarWearItem]


