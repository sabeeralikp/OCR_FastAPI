from sqlalchemy import Column, Integer, String


from database import Base


class OCR(Base):
    __tablename__ = "ocr"

    id = Column(Integer, primary_key=True)
    filename = Column(String)
    docetID = Column(String)
    yolo_text = Column(String)
    surya_text = Column(String)
    exec_time = Column(String)
    entities = Column(String)
