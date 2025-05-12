from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import paddleocr
from PIL import Image
import io

app = FastAPI()
ocr = paddleocr.OCR()

@app.post("/ocr/")
async def read_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    result = ocr.ocr(image)

    texts = []
    for line in result:
        for box in line:
            texts.append(box[1][0])  # box[1][0] is the detected text

    return JSONResponse(content={"text": texts})
