import time
from typing import List, Tuple, Dict
import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

from utils.models import Item, Result, PartResult

# 定数: セグメンテーション対象のラベルID
TOP_IDS: List[int] = [4, 7]
BOTTOM_IDS: List[int] = [5, 6, 7]

# 出力画像をこのサイズの正方形にリサイズ
OUTPUT_SIZE: int = 256

def get_device(use_cpu: bool) -> torch.device:
    """
    セグメンテーション処理で使用するデバイスを選択する。

    Args:
        use_cpu (bool): CPU を強制使用する場合は True。False の場合、CUDA が利用可能なら GPU を使用。

    Returns:
        torch.device: 選択されたデバイスオブジェクト。
    """
    if not use_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_binary_mask(mask: np.ndarray, id_list: List[int]) -> np.ndarray:
    """
    ラベルマスクから指定した ID の部分を検出し、バイナリアルファマスクを生成する。

    Args:
        mask (np.ndarray): セグメンテーションの予測ラベルマップ (H×W)。
        id_list (List[int]): 抽出したいラベル ID のリスト。

    Returns:
        np.ndarray: 0-255 の範囲でアルファチャンネル用マスクを表す二値配列。
    """
    binary = np.isin(mask, id_list).astype("uint8")
    return binary * 255


# 全身画像からアイテムのピクセルだけをくり抜き、正方形の PNG 画像を生成する
def create_png_with_alpha(
    img: Image.Image,
    alpha_mask: np.ndarray
) -> Image.Image:
    """
    ① 元画像とアルファマスクを合成して RGBA 画像を作成
    ② アルファマスク領域のバウンディングボックスでクロップ
    ③ 正方形キャンバスを透過背景で作成し、クロップ画像を中央に配置
    ④ OUTPUT_SIZE×OUTPUT_SIZE に高品質リサイズ

    Args:
        img (Image.Image):
            元の RGB 画像。
        alpha_mask (np.ndarray):
            アルファチャンネル用マスク（0〜255 の二値配列）。

    Returns:
        square (Image.Image):
            OUTPUT_SIZE×OUTPUT_SIZE の正方形画像。
    """
    # ① RGBA 合成
    rgba = img.convert("RGBA")
    alpha_img = Image.fromarray(alpha_mask)
    rgba.putalpha(alpha_img)

    # ② バウンディングボックス取得
    ys, xs = np.where(alpha_mask > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # ③ クロップ
    cropped = rgba.crop((x0, y0, x1 + 1, y1 + 1))
    w, h = cropped.size

    # ④ 正方形キャンバス作成＋中央配置
    new_size = max(w, h)
    square = Image.new("RGBA", (new_size, new_size), (0, 0, 0, 0))
    pad_x = (new_size - w) // 2
    pad_y = (new_size - h) // 2
    square.paste(cropped, (pad_x, pad_y), cropped)

    # ⑤ 高品質リサンプルで指定サイズにリサイズ
    square = square.resize((OUTPUT_SIZE, OUTPUT_SIZE), resample=Image.LANCZOS)

    return square


def load_model(use_cpu: bool) -> Tuple[torch.device, SegformerImageProcessor, AutoModelForSemanticSegmentation]:
    """
    セグメンテーション用モデルとプロセッサをロードし、推論モードに設定する。

    Args:
        use_cpu (bool): CPU を強制使用するかどうか。

    Returns:
        Tuple[torch.device, SegformerImageProcessor, AutoModelForSemanticSegmentation]:
            - device: 推論に利用するデバイス
            - processor: 画像前処理用オブジェクト
            - model: セグメンテーションモデル (推論モード)
    """
    device = get_device(use_cpu)
    processor = SegformerImageProcessor.from_pretrained(
        "mattmdjaga/segformer_b2_clothes"
    )
    model = AutoModelForSemanticSegmentation.from_pretrained(
        "mattmdjaga/segformer_b2_clothes"
    )
    model.to(device).eval()
    return device, processor, model


def run_batch_segmentation(
    img_base64: str
    processor: SegformerImageProcessor,
    model: AutoModelForSemanticSegmentation,
) -> SegmentationResult:
    """
    画像リストに対してトップスとボトムスのセグメンテーションを一括実行し、結果を返す。

    Args:
        img_base64 (str): 全身画像のBase64エンコード文字列。
        processor (SegformerImageProcessor): 前処理プロセッサ。
        model (AutoModelForSemanticSegmentation): セグメンテーション用モデル。

    Returns:
        SegmentationResult: トップスとボトムスのセグメンテーション結果を含むデータクラス
    """
    start_time = time.time()

    # 画像を読み込み RGB に変換
    decoded_data = base64.b64decode(img_base64)
    image_stream = io.BytesIO(decoded_data)
    img = Image.open(image_stream).convert("RGB")

    # モデル入力の準備
    inputs = processor(images=img, return_tensors="pt").to(model.device)

    # 推論実行
    with torch.inference_mode():
        logits = model(**inputs).logits

    # 出力サイズを元画像サイズに合わせて補間
    seg = F.interpolate(
        logits,
        size=img.size[::-1],
        mode="bilinear",
        align_corners=False
    )

    # ラベルごとの最大値を持つインデックスを取得
    mask = seg.argmax(dim=1)[0].cpu().numpy()

    # 各パートごとにマスクを生成し、検出有無を判定
    # [tops]
    tops_alpha = create_binary_mask(mask, TOP_IDS)
    tops_detected = bool(tops_alpha.max() > 0)
    # 検出できたなら結果を格納
    if tops_detected:
        tops_img = create_png_with_alpha(img, tops_alpha)
        # TODO: S3へ画像をアップロードし、トップス画像のURLを生成する。tops_image_url へURLを格納する。
        # 例: tops_image_url = upload_to_s3(tops_img, filename="tops.png")
    # 検出に失敗したならNoneを格納
    else:
        tops_image_url = None

    # [bottoms]
    bottoms_alpha = create_binary_mask(mask, BOTTOM_IDS)
    bottoms_detected = bool(bottoms_alpha.max() > 0)
    if bottoms_detected:
        bottoms_img = create_png_with_alpha(img, bottoms_alpha)
        # TODO: S3へ画像をアップロードし、ボトムス画像のURLを生成する。bottoms_image_url へURLを格納する。
        # 例: bottoms_image_url = upload_to_s3(bottoms_img, filename="tops.png")
    else:
        bottoms_image_url = None

    runtime_sec = round(time.time() - start_time, 3)

    segmentationResult = SegmentationResult(
        tops_image_url = tops_image_url,
        bottoms_image_url = bottoms_image_url,
        runtime_sec = runtime_sec
    )

    return segmentationResult
