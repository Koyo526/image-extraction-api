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
    img_paths: List[Path] = Field(..., description="解析対象の画像ファイルパスリスト")
    out_dir: Optional[Path] = Field(Path("outputs/segmented"), description="透過PNG保存先ディレクトリ")
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
    logging.info("Received segment request for %d images", len(request.img_paths))
    # Pydanticモデル -> dataclass へ変換
    items = [Item(img_path=p, filename=None) for p in request.img_paths]
    # セグメンテーション実行
    results = run_batch_segmentation(items, processor, model, request.out_dir)
    # 結果をファイルに保存
    save_results(results, Path("outputs/api_results.json"))
    return results
