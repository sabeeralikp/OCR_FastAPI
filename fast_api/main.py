from fastapi import FastAPI, HTTPException, UploadFile, status, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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

from chroma_utils import ChromaUtils

from fastapi import FastAPI, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.status import HTTP_403_FORBIDDEN

models.Base.metadata.create_all(bind=engine)


app = FastAPI()

chroma_utils = ChromaUtils()

allowed_IPs = [
    "117.193.73.30",
    "117.193.73.30",
    "app.cmo.kerala.gov.in",
    "117.193.73.44",
    "117.193.73.44",
    "192.168.16.54",
]

allowed_ports = [80, 443, 8080, 8000]

origins = [
    "http://117.193.73.30",
    "https://117.193.73.30",
    "https://app.cmo.kerala.gov.in",
    "http://117.193.73.44",
    "https://117.193.73.44",
    "http://192.168.16.54",
]


# Middleware to restrict access based on IP address and ports
class IPRestrictionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, allowed_ips, allowed_ports):
        super().__init__(app)
        self.allowed_ips = allowed_ips
        self.allowed_ports = allowed_ports

    async def dispatch(self, request, call_next):
        if request.client:
            remote_ip = request.client.host
            remote_port = request.url.port

            if (
                remote_ip not in self.allowed_ips
                or remote_port not in self.allowed_ports
            ):
                return JSONResponse(
                    content="Access denied", status_code=HTTP_403_FORBIDDEN
                )

        response = await call_next(request)
        return response


# Add the middleware to the app
app.add_middleware(
    IPRestrictionMiddleware,
    allowed_ips=allowed_IPs,
    allowed_ports=allowed_ports,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
            "DATE": set([]),
            "DOC_ID": set([]),
            "FROM": set([]),
            "FROM_ADD": set([]),
            "INDICATION": set([]),
            "PERSONAL_NAME": set([]),
            "PHONE": set([]),
            "PLACE": set([]),
            "SUBJECT": set([]),
            "TO": set([]),
            "TO_ADD": set([]),
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
        "ocr_text1": "\n".join(doc_texts["yolo_text"]),
        "ocr_text2": "\n".join(doc_texts["surya_text"]),
        # "exec_time": str(time.time() - start_time),
        "entities": doc_texts["entities"],
    }


@app.get("/search")
def vector_search(query_str: str):
    return chroma_utils.vector_search(query_str)
