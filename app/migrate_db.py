import time
from sqlalchemy.orm import sessionmaker
from db import Base, ENGINE
import logging

# SQLAlchemyセッションの準備
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def reset_database():
    # テーブルの初期化（全削除 → 作成）
    Base.metadata.drop_all(bind=ENGINE)
    Base.metadata.create_all(bind=ENGINE)

    db = SessionLocal()
    try:
        # ユーザー初期データ

        db.commit()
        logger.info("✅ 初期データの挿入が完了しました。")

    except Exception as e:
        db.rollback()
        logger.error(f"❌ 初期データ挿入中にエラー: {e}")

    finally:
        db.close()

if __name__ == "__main__":
    for i in range(10):
        try:
            logger.info(f"⏳ DB接続試行中... ({i+1}/10)")
            reset_database()
            break
        except Exception as e:
            logger.error(f"DB接続失敗、再試行({i+1}/10)...: {e}")
            time.sleep(2)
