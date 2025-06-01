import requests
from io import BytesIO
from utils.segmentation import load_model, run_batch_segmentation
from pydantic import BaseModel, Field
import logging
from utils.models import SegmentationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# アプリ起動時にモデルをロード
device, processor, model = load_model(use_cpu=False)

class SegmentRequest(BaseModel):
    img_base64: str
    use_cpu: bool = Field(False, description="CPUを強制使用する場合は True")



def segment_tops_bottoms_images(image_base64: str, user_token: str) -> SegmentationResult:

    # セグメンテーションモデルで画像を分割（ファイル名は仮）
    segment_result = run_batch_segmentation(image_base64, processor, model, user_token=user_token)

    return segment_result
