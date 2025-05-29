from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import logging

from utils.segmentation import load_model, run_batch_segmentation
from utils.file_io import save_results
from utils.models import Item, SegmentationResult

# アプリ起動時にモデルをロード
device, processor, model = load_model(use_cpu=False)

class SegmentRequest(BaseModel):
    """
    セグメンテーション要求データモデル。
    """
    img_base64: str
    use_cpu: bool = Field(False, description="CPUを強制使用する場合は True")

router = APIRouter()


"""
- Request: SegmentRequest
    セグメンテーション要求情報を表すデータクラス。
    Attributes:
        img_base64 (str): 全身画像のBase64エンコード文字列。
        use_cpu (bool): CPUを強制使用する場合はTrue。デフォルトはFalse。


- Response : SegmentationResult:
    セグメンテーション処理結果を表すデータクラス。
    Attributes:
        tops_img_url (Optional[str]): トップスの画像URL。検出されなかった場合はNone。
        bottoms_img_url (Optional[str]): ボトムスの画像URL。検出されなかった場合はNone。
        runtime_sec (float): 推論に要した時間（秒）。
"""
@router.post("/segment", response_model=SegmentationResult)
def segment(request: SegmentRequest) -> SegmentationResult:
    """
    画像リストに対してトップス/ボトムスのセグメンテーションを実行するエンドポイント。

    Args:
        request (SegmentRequest): セグメンテーション要求情報。

    Returns:
        List[Result]: セグメンテーション結果のリスト。
    """
    # セグメンテーション実行
    segmentationResult = run_batch_segmentation(request.img_base64, processor, model, request.out_dir)
    
    return segmentationResult
