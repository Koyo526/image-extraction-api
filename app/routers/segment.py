from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import logging

from utils.segmentation import load_model, run_batch_segmentation
from utils.file_io import save_results
from models.models import Item, Result

# アプリ起動時にモデルをロード
device, processor, model = load_model(use_cpu=False)

class SegmentRequest(BaseModel):
    """
    セグメンテーション要求データモデル。
    """
    img_base64: str
    use_cpu: bool = Field(False, description="CPUを強制使用する場合は True")

router = APIRouter()

@router.post("/segment", response_model=List[Result])
def segment(request: SegmentRequest) -> List[Result]:
    """
    画像リストに対してトップス/ボトムスのセグメンテーションを実行するエンドポイント。

    Args:
        request (SegmentRequest): セグメンテーション要求情報。

    Returns:
        List[Result]: セグメンテーション結果のリスト。
    """
    # セグメンテーション実行
    results = run_batch_segmentation(request.img_base64, processor, model, request.out_dir)
    # 結果をファイルに保存
    save_results(results, Path("outputs/api_results.json"))
    return results
