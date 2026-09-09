from pydantic import BaseModel


class SegmentationResponse(BaseModel):
    result_id: str
    filename: str
    width: int
    height: int
    image_url: str
    message: str
    
    
