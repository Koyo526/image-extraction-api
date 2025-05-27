from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import base64
import io
import json
import os
import sqlite3
import requests
import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # 非表示バックエンドを指定
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams["font.family"] = "Noto Sans CJK JP"

# 追加: torchvisionを用いた人物検出用モジュール
import torchvision
from torchvision import transforms as T
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# -------------------------------
# グローバル設定・初期化
# -------------------------------
DATA_CSV_PATH = "data/merged_fashion.csv"         # CSVファイルのパス
EMBEDDINGS_DB_PATH = "db/embeddings.db"               # SQLite DBのパス
SIMILARITY_THRESHOLD = 0.7                         # グラフ作成用の閾値（必要に応じて調整）
TOP_K = 30                                       # 出力グラフのノード数（上位30件）

# グローバル変数（データセット情報）
dataset_image_urls = []
dataset_ids = []
dataset_post_urls = []   # 追加：投稿URLを格納するリスト
dataset_embeddings = []  # 各画像のembedding（numpy配列）
prefix_to_color = {}

# CLIPモデルとプロセッサの読み込み（アプリ起動時に一度だけ読み込み）
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 追加: 人物検出モデルの読み込み（Faster R-CNN、COCOのpersonカテゴリを使用）
person_detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
person_detector.eval()  # 推論モードへ

# FastAPIアプリケーションの初期化
router = FastAPI()

# -------------------------------
# ヘルパー関数
# -------------------------------
def detect_person(image: Image.Image) -> Image.Image:
    """
    入力画像から人物（person）を検出し、最も大きな領域をクロップして返す。
    検出できなかった場合は元の画像をそのまま返す。
    """
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        predictions = person_detector(img_tensor)[0]
    
    # COCOのpersonカテゴリはラベル1。信頼度0.8以上の候補を採用
    person_boxes = []
    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if label == 1 and score > 0.8:
            person_boxes.append((box, score))
    
    if person_boxes:
        # 複数ある場合は、面積が最大の領域を選択
        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])
        best_box, _ = max(person_boxes, key=lambda x: box_area(x[0]))
        best_box = best_box.tolist()
        # 画像をクロップ（[x_min, y_min, x_max, y_max]）
        cropped_image = image.crop(best_box)
        return cropped_image
    else:
        # 人物が検出できなかった場合は元の画像を返す
        return image

def compute_embedding(image: Image.Image) -> np.ndarray:
    """
    PIL ImageからCLIPのembeddingを計算する。
    画像全体ではなく、まず人物領域を検出してからembeddingを計算する。
    """
    # 人物領域の検出とクロップ
    person_image = detect_person(image)
    inputs = processor(images=person_image, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    vector = outputs.detach().numpy().flatten()
    return vector

def load_dataset_and_embeddings():
    """
    CSVからデータ読み込み、SQLite DBを用いて各画像のembeddingを取得・計算、
    外れ値除去、色の割当を行う。
    """
    global dataset_image_urls, dataset_ids, dataset_post_urls, dataset_embeddings, prefix_to_color

    # CSVから画像URL, ID, 投稿URLを読み込み
    df = pd.read_csv(DATA_CSV_PATH)
    dataset_image_urls = df["image_url"].tolist()
    dataset_ids = df["id"].tolist()
    dataset_post_urls = df["post_url"].tolist()  # 追加：投稿URLの読み込み

    # SQLite DB接続（なければ作成）
    conn = sqlite3.connect(EMBEDDINGS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            image_url TEXT PRIMARY KEY,
            embedding TEXT
        )
    """)
    conn.commit()

    embeddings_list = []
    for url in dataset_image_urls:
        cursor.execute("SELECT embedding FROM embeddings WHERE image_url = ?", (url,))
        row = cursor.fetchone()
        if row is not None:
            # DBから読み込み
            vector = np.array(json.loads(row[0]))
            print(f"{url} のembeddingをDBから読み込みました。")
        else:
            try:
                response = requests.get(url, stream=True, timeout=10)
                image = Image.open(response.raw).convert("RGB")
            except Exception as e:
                print(f"{url} の画像取得エラー: {e}")
                vector = np.zeros(512)  # エラー時はゼロベクトル
            else:
                vector = compute_embedding(image)
                cursor.execute("INSERT INTO embeddings (image_url, embedding) VALUES (?, ?)",
                               (url, json.dumps(vector.tolist())))
                conn.commit()
                print(f"{url} のembeddingを計算し、DBに保存しました。")
        embeddings_list.append(vector)
    conn.close()

    # 全画像間の類似度を計算し、各画像の平均類似度が平均-2σより低いものを外れ値として除去
    similarity_matrix = cosine_similarity(embeddings_list)
    avg_similarities = np.mean(similarity_matrix, axis=1)
    mean_avg = np.mean(avg_similarities)
    std_avg = np.std(avg_similarities)
    threshold_outlier = mean_avg - 2 * std_avg
    print(f"全画像数: {len(dataset_image_urls)}, 除外する外れ値の画像数: {len(dataset_image_urls) - np.count_nonzero(avg_similarities >= threshold_outlier)}")

    non_outlier_indices = np.where(avg_similarities >= threshold_outlier)[0]

    # 外れ値除去後のリスト更新
    dataset_image_urls = [dataset_image_urls[i] for i in non_outlier_indices]
    dataset_ids = [dataset_ids[i] for i in non_outlier_indices]
    dataset_post_urls = [dataset_post_urls[i] for i in non_outlier_indices]  # 追加
    dataset_embeddings = [embeddings_list[i] for i in non_outlier_indices]

    # idのプレフィックスから色を割り当て（wearのユーザー名とみなす）
    prefixes = [id_str.split("_")[0] for id_str in dataset_ids]
    unique_prefixes = sorted(set(prefixes))
    prefix_to_color = {prefix: plt.cm.tab20(i % 20) for i, prefix in enumerate(unique_prefixes)}

