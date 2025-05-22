from fastapi import APIRouter, UploadFile, File, Form

router = APIRouter()

@router.post("/upload")
async def upload_item_image(
    session_id: str = Form(...),
    user_token: str = Form(...),
    image: UploadFile = File(...)
):
    return {
        "tops_image_url": "https://example.com/tops.jpg",
        "bottoms_image_url": "https://example.com/bottoms.jpg",
        "shoes_image_url": "https://example.com/shoes.jpg"
    }
