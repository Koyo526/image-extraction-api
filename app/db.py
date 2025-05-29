import os
import json
import logging
import boto3
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("🔍 起動環境: %s", os.getenv("ENV", "local"))

def get_db_url_from_secrets(secret_id: str, region_name: str = "ap-northeast-1") -> str:
    try:
        client = boto3.client("secretsmanager", region_name=region_name)
        response = client.get_secret_value(SecretId=secret_id)
        secret = json.loads(response["SecretString"])
        logger.info("🔐 SecretsManagerから秘密情報を正常に取得しました。")

        return (
            f"mysql+pymysql://{secret['username']}:{secret['password']}"
            f"@{secret['host']}:{secret['port']}/{secret['dbname']}?charset=utf8mb4"
        )
    except Exception as e:
        logger.error(f"SecretsManagerからの取得に失敗しました: {e}")
        raise

def get_db_url_from_env() -> str:
    user = os.getenv("DB_USER", "user")
    password = os.getenv("DB_PASSWORD", "password")
    host = os.getenv("DB_HOST", "db")
    port = os.getenv("DB_PORT", "3306")
    dbname = os.getenv("DB_NAME", "irodori")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}?charset=utf8mb4"

# 本番 or ローカル環境判定
ENV = os.getenv("ENV", "local")

if ENV == "production":
    secret_name = os.getenv("SECRET_NAME")
    if not secret_name:
        raise RuntimeError("SECRET_NAME environment variable must be set in production mode.")
    DB_URL = get_db_url_from_secrets(secret_name)
else:
    DB_URL = get_db_url_from_env()

# SQLAlchemyエンジン設定
ENGINE = create_engine(DB_URL, echo=True)

session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=ENGINE))
Base = declarative_base()
Base.query = session.query_property()

def get_db():
    db = session()
    try:
        yield db
    finally:
        db.close()
