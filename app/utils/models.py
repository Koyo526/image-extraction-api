from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
import io

@dataclass
class Item:
    """
    セグメンテーション対象の画像アイテムを表すデータクラス。

    Attributes:
        img_base64 (str): 画像データをBase64エンコードした文字列
        filename (Optional[str]): 保存時に用いるファイル名。未指定の場合は元ファイル名を使用。
    """
    img_base64: str
    filename: Optional[str] = None

@dataclass
class PartResult:
    """
    各パーツ（トップス/ボトムス）の検出結果を表すデータクラス。

    Attributes:
        detected (bool): パーツが検出されたかどうか。
    """
    detected: bool

@dataclass
class SegmentationResult:
    """
    セグメンテーション処理結果を表すデータクラス。

    Attributes:
        tops_img_url (Optional[str]): トップスの画像URL。検出されなかった場合はNone。
        bottoms_img_url (Optional[str]): ボトムスの画像URL。検出されなかった場合はNone。
        runtime_sec (float): 推論に要した時間（秒）。
    """
    tops_img_url: Optional[str] = None
    bottoms_img_url: Optional[str] = None
    runtime_sec: float = 0.0

    def to_dict(self) -> dict:
        """
        Resultオブジェクトを辞書形式に変換する。

        Returns:
            dict: JSONシリアライズ可能な辞書。
        """
        return {
            "status": self.status,
            "parts": {k: v.__dict__ for k, v in self.parts.items()},
            "runtime_sec": self.runtime_sec,
        }


class SimpleUploadFile:
    def __init__(self, filename: str, file: io.BytesIO):
        self.filename = filename
        self.file = file
