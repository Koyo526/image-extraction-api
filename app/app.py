from fastapi import FastAPI
from contextlib import asynccontextmanager

from routers import (
    check_gpt,
    check_vision_gpt,
    coordinate_response,
    segment,
    search_my_fashion_api,
    upload
)
from routers.search_my_fashion_api import load_dataset_and_embeddings  # あなたの実装に合わせて変更

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動前処理
    load_dataset_and_embeddings()
    print("✅ Datasetとembeddingsのロード完了")
    yield
    # 終了時処理（あれば）

app = FastAPI(lifespan=lifespan)

# router登録

@app.get("/")
def root():
    return {"message": "Hello, FastAPI on Render!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

app.include_router(upload.router, tags=["upload"])
app.include_router(check_gpt.router, tags=["gpt"])
app.include_router(check_vision_gpt.router, tags=["vision"])
app.include_router(coordinate_response.router, tags=["coordinate"])
app.include_router(segment.router, tags=["segment"])
app.include_router(search_my_fashion_api.router, tags=["search-my-fashion-api"])

