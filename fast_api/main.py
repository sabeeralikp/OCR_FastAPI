from fastapi import FastAPI, HTTPException, UploadFile, status, Depends, BackgroundTasks
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

import chroma_utils

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
async def ocr(
    file: UploadFile,
    docetID: str,
    background_task: BackgroundTasks,
    db: Session = Depends(get_db),
):
    doc_texts = {
        "yolo_text": [],
        "surya_text": [],
        "filename": "",
        "docetID": docetID,
        "entities": {
            "FROM": set([]),
            "FROM_ADD": set([]),
            "SUBJECT": set([]),
            "DATE": set([]),
            "TO": set([]),
            "TO_ADD": set([]),
            "PHONE_NUM": set([]),
        },
    }
    start_time = time.time()
    try:
        contents = await file.read()
        async with aiofiles.open(f"data/{file.filename}", "wb") as f:
            await f.write(contents)
        doc_texts["filename"] = file.filename
        if file.content_type == "application/pdf":
            images = convert_from_bytes(contents)
            for i in range(len(images)):
                yolo_text_1 = yoloTesseract(np.array(images[i])) + "\n"
                doc_texts["entities"] = extract_entity(
                    yolo_text_1, doc_texts["entities"]
                )
                doc_texts["yolo_text"].append(yolo_text_1)
                surya_text_1 = suryaOCR(images[i]) + "\n"
                doc_texts["entities"] = extract_entity(
                    surya_text_1, doc_texts["entities"]
                )
                doc_texts["surya_text"].append(surya_text_1)
        else:
            image = Image.open(BytesIO(contents))
            doc_texts["yolo_text"].append(yoloTesseract(np.array(image)))
            doc_texts["entities"] = extract_entity(
                doc_texts["yolo_text"][0], doc_texts["entities"]
            )
            doc_texts["surya_text"].append(suryaOCR(image))
            doc_texts["entities"] = extract_entity(
                doc_texts["surya_text"][0], doc_texts["entities"]
            )

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="There was an error uploading the file",
        )
    finally:
        await file.close()

    background_task.add_task(
        crud.create_ocr,
        db=db,
        ocr=models.OCR(
            filename=file.filename,
            docetID=docetID,
            yolo_text="\n".join(doc_texts["yolo_text"]),
            surya_text="\n".join(doc_texts["surya_text"]),
            exec_time=str(time.time() - start_time),
            entities=str(doc_texts["entities"]),
        ),
    )

    background_task.add_task(
        chroma_utils.add_collections,
        doc_texts,
    )

    return {
        "filename": file.filename,
        "docetID": docetID,
        "yolo_text": "\n".join(doc_texts["yolo_text"]),
        "surya_text": "\n".join(doc_texts["surya_text"]),
        "exec_time": str(time.time() - start_time),
        "entities": doc_texts["entities"],
    }
