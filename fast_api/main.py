from fastapi import FastAPI, HTTPException, UploadFile, status, Depends
import aiofiles

from yoloOCR import yoloTesseract
from suryaOCR import suryaOCR
from pdf2image import convert_from_bytes
from PIL import Image
from io import BytesIO
import numpy as np
from entity import extract_entity
import time

import models
from database import engine, SessionLocal
from jsonpickle import encode
from sqlalchemy.orm import Session
import crud, models

models.Base.metadata.create_all(bind=engine)


app = FastAPI()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/ocr")
async def ocr(file: UploadFile, docetID: str, db: Session = Depends(get_db)):
    yolo_text = ""
    surya_text = ""
    entities = {
        "FROM": set([]),
        "FROM_ADD": set([]),
        "SUBJECT": set([]),
        "DATE": set([]),
        "TO": set([]),
        "TO_ADD": set([]),
        "PHONE_NUM": set([]),
    }
    start_time = time.time()
    try:
        contents = await file.read()
        async with aiofiles.open(f"data/{file.filename}", "wb") as f:
            await f.write(contents)

        if file.content_type == "application/pdf":
            images = convert_from_bytes(contents)
            for i in range(len(images)):
                yolo_text_1 = yoloTesseract(np.array(images[i])) + "\n"
                entities = extract_entity(yolo_text_1, entities)
                yolo_text += yolo_text_1
                surya_text_1 = suryaOCR(images[i]) + "\n"
                entities = extract_entity(surya_text_1, entities)
                surya_text += surya_text_1
        else:
            image = Image.open(BytesIO(contents))
            yolo_text = yoloTesseract(np.array(image))
            entities = extract_entity(yolo_text, entities)
            surya_text = suryaOCR(image)
            entities = extract_entity(surya_text, entities)

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="There was an error uploading the file",
        )
    finally:
        await file.close()

    database_response = crud.create_ocr(
        db=db,
        ocr=models.OCR(
            filename=file.filename,
            docetID=docetID,
            yolo_text=yolo_text,
            surya_text=surya_text,
            exec_time=str(time.time() - start_time),
            entities=str(entities),
        ),
    )

    return database_response
