from fastapi import FastAPI
from contextlib import asynccontextmanager

from routers import (
    check_gpt,
    check_vision_gpt,
    coordinate_response,
    segment,
    search_my_fashion_api,
    upload,
    fashion_review
)
from services.search_fashion_items import load_dataset_and_embeddings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動前処理
    load_dataset_and_embeddings()
    print("✅ Datasetとembeddingsのロード完了")
    yield
    # 終了時処理（あれば）

app = FastAPI(lifespan=lifespan)

API_VERSION = "v1"

# router登録

@app.get("/")
def root():
    return {"message": "Hello, FastAPI on Render!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

app.include_router(upload.router, prefix=f"/{API_VERSION}", tags=["upload"])
app.include_router(check_gpt.router, prefix=f"/{API_VERSION}", tags=["gpt"])
app.include_router(check_vision_gpt.router, prefix=f"/{API_VERSION}", tags=["vision"])
app.include_router(coordinate_response.router, prefix=f"/{API_VERSION}", tags=["coordinate"])
app.include_router(segment.router, prefix=f"/{API_VERSION}", tags=["segment"])
app.include_router(search_my_fashion_api.router, prefix=f"/{API_VERSION}", tags=["search-my-fashion-api"])
app.include_router(fashion_review.router, prefix=f"/{API_VERSION}", tags=["fashion-review"])

