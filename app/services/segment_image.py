import requests
from io import BytesIO
from utils.segmentation import load_model, run_batch_segmentation
from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# アプリ起動時にモデルをロード
device, processor, model = load_model(use_cpu=False)

class SegmentRequest(BaseModel):
    img_base64: str
    use_cpu: bool = Field(False, description="CPUを強制使用する場合は True")



def segment_image(original_url: str, user_token: str) -> tuple[str, str]:
    # 画像を取得
    response = requests.get(original_url)
    if response.status_code != 200:
        raise Exception("画像のダウンロードに失敗しました")
    
    image_data = BytesIO(response.content)

    # セグメンテーションモデルで画像を分割（ファイル名は仮）
    segment_result = run_batch_segmentation(image_data, processor, model, user_token=user_token)

    return segment_result
