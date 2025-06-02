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
import time
import logging


# ログ設定（必要に応じてカスタマイズ）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        overall_start = time.time()

        # === Step 1: S3アップロード ===
        start = time.time()
        timestamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d%H%M%S')
        fileName = f"input/{user_token}-{timestamp}.jpg"
        original_url = upload_to_s3(image, fileName)
        logger.info(f"[Step 1] S3 upload completed in {time.time() - start:.2f} seconds")

        # === Step 2: Base64エンコード ===
        start = time.time()
        response = requests.get(original_url)
        if response.status_code != 200:
            raise Exception("画像のダウンロードに失敗しました")
        image_data = BytesIO(response.content)
        image_base64 = base64.b64encode(image_data.getvalue()).decode('utf-8')
        logger.info(f"[Step 2] Image base64 encoding completed in {time.time() - start:.2f} seconds")

        # === Step 3: セグメンテーション ===
        start = time.time()
        segment_result = segment_tops_bottoms_images(image_base64, user_token, timestamp)
        logger.info(f"[Step 3] Segmentation completed in {time.time() - start:.2f} seconds")

        # === Step 4: コーディネートレビュー ===
        start = time.time()
        image_request = ImageRequest(
            image_base64=image_base64,
            outing_purpose_id=outing_purpose_id
        )
        coordinate_result = review_coordinate(image_request)
        logger.info(f"[Step 4] Coordinate review completed in {time.time() - start:.2f} seconds")

        # === Step 5: 類似アイテム検索 ===
        start = time.time()
        queryInput = QueryInput(
            image_base64=image_base64,
            query_width=query_width,
            query_height=query_height
        )
        fashion_results = search_fashion_items(queryInput, user_token, timestamp)
        logger.info(f"[Step 5] Fashion item search completed in {time.time() - start:.2f} seconds")

        logger.info(f"[All Steps] Total execution time: {time.time() - overall_start:.2f} seconds")

        # === Step 6: 結果整形・返却 ===
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
                    "post_url": item.post_url,
                    "image_url": item.image_url
                } for item in fashion_results.similar_wear
            ]
        }

    except Exception as e:
        logger.exception("ファッションレビュー処理中に例外が発生しました")
        raise HTTPException(status_code=500, detail=str(e))
