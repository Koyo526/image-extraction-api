from pathlib import Path
import json
from typing import List
import boto3
import uuid
import os
from datetime import timedelta
# from utils.models import Item, Result
from utils.models import Item




def load_items(path: Path) -> List[Item]:
    """
    入力JSONからItem一覧を生成する。

    Args:
        path (Path): 入力JSONファイルへのパス。

    Returns:
        List[Item]: JSONの内容をもとに生成したItemオブジェクトのリスト。
    """
    # JSONファイルを読み込み
    data = json.loads(path.read_text(encoding="utf-8"))
    items: List[Item] = []
    for d in data:
        img_path = Path(d["img_path"])
        filename = d.get("filename")
        # Itemオブジェクトを作成
        items.append(Item(img_path=img_path, filename=filename))
    return items


# def save_results(results: List[Result], path: Path) -> None:
#     """
#     セグメンテーション結果(Result)のリストをJSON形式で保存する。

#     Args:
#         results (List[Result]): 保存対象のResultオブジェクトのリスト。
#         path (Path): 保存先JSONファイルへのパス。
#     """
#     # 保存先ディレクトリを作成
#     path.parent.mkdir(parents=True, exist_ok=True)
#     # 辞書形式に変換してJSONを出力
#     with path.open("w", encoding="utf-8") as f:
#         json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
