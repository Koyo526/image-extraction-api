from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

@dataclass
class Item:
    """
    セグメンテーション対象の画像アイテムを表すデータクラス。

    Attributes:
        img_path (Path): 処理対象の画像ファイルパス。
        filename (Optional[str]): 保存時に用いるファイル名。未指定の場合は元ファイル名を使用。
    """
    img_path: Path
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
class Result:
    """
    セグメンテーション処理結果を表すデータクラス。

    Attributes:
        filename (str): 結果保存時に用いるファイル名。
        img_path (str): 入力画像のパス。
        status (str): 処理ステータス ('success' or 'skipped')。
        parts (Dict[str, PartResult]): パーツ名をキーとする PartResult の辞書。
        runtime_sec (float): 推論に要した時間（秒）。
    """
    filename: str
    img_path: str
    status: str
    parts: Dict[str, PartResult]
    runtime_sec: float

    def to_dict(self) -> dict:
        """
        Resultオブジェクトを辞書形式に変換する。

        Returns:
            dict: JSONシリアライズ可能な辞書。
        """
        return {
            "filename": self.filename,
            "img_path": self.img_path,
            "status": self.status,
            "parts": {k: v.__dict__ for k, v in self.parts.items()},
            "runtime_sec": self.runtime_sec,
        }
