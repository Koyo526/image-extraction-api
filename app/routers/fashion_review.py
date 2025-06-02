# routers/bff.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from datetime import datetime, timezone, timedelta
from services.upload_service import upload_to_s3
from services.segment_image import segment_tops_bottoms_images
from services.review_coordinate import review_coordinate
from services.search_fashion_items import search_fashion_items
import requests
from io import BytesIO
from schemas.review import ReviewResponse,ImageRequest,QueryInput
import base64

router = APIRouter()

@router.post("/fashion-review", response_model=ReviewResponse)
def fashion_review(
    image: UploadFile = File(..., description="全身画像"),
    user_token: str = Form(..., description="ユーザートークン"),
    outing_purpose_id: int = Form(9, description="0:職場, 1:デート, 2:買い物, 3:大学, 4:カフェ, 5:飲み会（同性のみ）, 6:飲み会, 7:運動, 9:特になし"),
    query_width: int = Form(None, description="中央画像の希望横幅（ピクセル）"),
    query_height: int = Form(None, description="中央画像の希望縦幅（ピクセル）")
):
    try:
        # 1. 全身画像をS3にアップロード
        timestamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d%H%M%S')
        fileName = f"input/{user_token}-{timestamp}.jpg"
        original_url = upload_to_s3(image, fileName)

        response = requests.get(original_url)
        if response.status_code != 200:
            raise Exception("画像のダウンロードに失敗しました")
        
        image_data = BytesIO(response.content)

        image_base64 = base64.b64encode(image_data.getvalue()).decode('utf-8')

        # 2. セグメンテーション実行（ローカルファイル or URLで呼ぶ）
        segment_result = segment_tops_bottoms_images(image_base64, user_token,timestamp)

        # 3. コーディネートレビュー
        # imageをbase64に変換
        image_request = ImageRequest(
            image_base64=image_base64,
            outing_purpose_id=outing_purpose_id
        )
        coordinate_result = review_coordinate(image_request)

        # 4. 類似アイテム検索
        queryInput = QueryInput(
            image_base64=image_base64,
            query_width=query_width,
            query_height=query_height
        )

        fashion_results = search_fashion_items(queryInput, user_token,timestamp)

        print("=== OpenAI Response Content ===")
        print(repr(fashion_results))
        # 5. 結果をまとめて返す
        return {
            "createdAt": timestamp,
            "tops_image_url": segment_result.tops_img_url,
            "bottoms_image_url": segment_result.bottoms_img_url,
            "coordinate": {
                "coordinate_review": coordinate_result.coordinate_review,
                "coordinate_item01": coordinate_result.coordinate_item01,
                "recommend_item01": coordinate_result.recommend_item01,
                "coordinate_item02": coordinate_result.coordinate_item02,
                "recommend_item02": coordinate_result.recommend_item02,
                "coordinate_item03": coordinate_result.coordinate_item03,
                "recommend_item03": coordinate_result.recommend_item03
            },
            "graph_image": fashion_results.graph_image,
            "recommendations": [
                {
                    "username": item.username,
                    "image_base64": item.image_base64,
                    "post_url": item.post_url,
                    "image_url": item.image_url
                } for item in fashion_results.similar_wear
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
