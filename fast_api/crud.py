from sqlalchemy.orm import Session
import models, schemas


def create_ocr(db: Session, ocr: schemas.OCRCreate):
    db_ocr = models.OCR(
        filename=ocr.filename,
        yolo_text=ocr.yolo_text,
        surya_text=ocr.surya_text,
        docetID=ocr.docetID,
        exec_time=ocr.exec_time,
        entities=ocr.entities,
    )
    db.add(db_ocr)
    db.commit()
    db.refresh(db_ocr)
    return db_ocr
