from sqlalchemy import or_
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


def get_db_all_ocr(db: Session):
    return db.query(models.OCR).all()


def get_string_matches(db: Session, search_text: str):
    output_list = (
        db.query(models.OCR)
        .filter(
            or_(
                models.OCR.surya_text.contains(search_text),
                models.OCR.yolo_text.contains(search_text),
                models.OCR.entities.contains(search_text),
            )
        )
        .all()
    )

    return [ocr.docetID for ocr in output_list]


def get_substring_matches(db: Session, search_text: str):
    if len(search_text.split()) < 2:
        return []
    output_list = []
    for key in search_text.split():
        if len(key) > 2:
            output_list.extend(
                db.query(models.OCR)
                .filter(
                    or_(
                        models.OCR.surya_text.contains(key),
                        models.OCR.yolo_text.contains(key),
                        models.OCR.entities.contains(key),
                    )
                )
                .all()
            )
    return [ocr.docetID for ocr in output_list]
