from pydantic import BaseModel


class ImageUploadResponse(BaseModel):
    filename: str
    width: int
    height: int
    message: str