from pydantic import BaseModel


class OCRBase(BaseModel):
    filename: str
    docetID: str
    yolo_text: str
    surya_text: str
    exec_time: str
    entities: str


class OCRCreate(OCRBase):
    pass


class OCR(OCRBase):
    id: int

    class Config:
        orm_mode = True
