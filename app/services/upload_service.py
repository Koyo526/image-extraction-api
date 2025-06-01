import boto3
from fastapi import UploadFile
from datetime import datetime, timezone, timedelta
from pytz import timezone
import os

# 設定（実際は環境変数やconfigファイルから読み込むことを推奨）
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-1") 
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "irodori-s3")
# boto3クライアント生成
s3 = boto3.client("s3", region_name=AWS_REGION)

def upload_to_s3(file: UploadFile, fileName: str) -> str:
    """画像ファイルをS3にアップロードし、CloudFront URLを返す"""

    # ファイル拡張子取得
    ext = file.filename.split(".")[-1]
    if ext not in ["jpg", "jpeg", "png"]:
        raise ValueError("Unsupported file type")

    # ユーザーIDとタイムスタンプを元にファイル名を生成
    timestamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d%H%M%S')

    # S3アップロード
    s3.upload_fileobj(
        Fileobj=file.file,
        Bucket=BUCKET_NAME,
        Key=fileName,
        ExtraArgs={"ContentType": file.content_type}
    )

    # Presigned URL の生成（1時間有効）
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET_NAME, "Key": fileName},
        ExpiresIn=3600  # 1時間
    )

    return url
