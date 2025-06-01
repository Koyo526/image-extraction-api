from fastapi import APIRouter, UploadFile, File, Form
from utils.file_io import upload_image_and_get_presigned_url
router = APIRouter()

@router.post("/upload")
def upload_item_image(user_token: str = Form(...), image: UploadFile = File(...)):
    image_bytes = image.file.read()
    presigned_url = upload_image_and_get_presigned_url(image_bytes, "item_images")

    #ここでBFFのように、他のマイクロサービスのAPIを呼び出して、画像URLを取得する処理を追加できます。
    tops_image, bottoms_image = bff(image_bytes)

    tops_image_url = tops_image.get("tops_image_url")
    bottoms_image_url = bottoms_image.get("bottoms_image_url")
    return {
        "tops_image_url": tops_image_url,
        "bottoms_image_url": bottoms_image_url,
    }
