from surya.ocr import run_ocr
from surya.model.detection import segformer
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor
import torch

langs = ["ml", "en"]  # Replace with your languages

det_processor, det_model = segformer.load_processor(), segformer.load_model(
    device="cpu", dtype=torch.float32
)

rec_model, rec_processor = (
    load_model(device="cpu", dtype=torch.float32),
    load_processor(),
)


def suryaOCR(image):
    torch.cuda.empty_cache()
    # image = Image.open("test.jpeg")
    predictions = run_ocr(
        [image], [langs], det_model, det_processor, rec_model, rec_processor
    )
    output = "\n".join([p.text for p in predictions[0].text_lines])
    predictions = None
    return output
