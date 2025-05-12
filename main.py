from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import io
from PIL import Image
import numpy as np

app = FastAPI()
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # You can customize this

@app.post("/ocr/")
async def ocr_api(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    result = ocr.ocr(image_np, cls=True)
    text_results = []
    for line in result:
        for word_info in line:
            text_results.append(word_info[1][0])  # The recognized text

    return JSONResponse(content={"text": text_results})
