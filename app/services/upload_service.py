import boto3
from fastapi import UploadFile
from pytz import timezone
import os

AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-1") 
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "irodori-s3")
# boto3クライアント生成
s3 = boto3.client("s3", 
                region_name=AWS_REGION,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

def upload_to_s3(file: UploadFile, fileName: str) -> str:
    # ファイル拡張子取得
    ext = file.filename.split(".")[-1].lower()
    if ext not in ["jpg", "jpeg", "png"]:
        raise ValueError("Unsupported file type")

    # ContentType の補完（Noneなら "image/jpeg" を仮定）
    content_type = file.content_type or "image/jpeg"

    # S3アップロード
    s3.upload_fileobj(
        Fileobj=file.file,
        Bucket=BUCKET_NAME,
        Key=fileName,
        ExtraArgs={"ContentType": content_type}
    )

    # Presigned URL の生成（1時間有効）
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET_NAME, "Key": fileName},
        ExpiresIn=3600  # 1時間
    )

    return url
