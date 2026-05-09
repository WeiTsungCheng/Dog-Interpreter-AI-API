
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import io

app = FastAPI()

processor: Optional[AutoProcessor] = None
model: Optional[AutoModelForImageTextToText] = None

@app.on_event("startup")
def load_model():
    global processor, model

    LOCAL_MODEL_PATH = "./models/blip-base"

    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(LOCAL_MODEL_PATH)
    model.eval()

    print("Model loaded!")

@app.get("/")
def root():
    return {"message": "Dog Caption API is running"}

@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):

    global processor, model
    if processor is None or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5
        )

    caption = processor.decode(output[0], skip_special_tokens=True)

    return {
        "caption": caption
    }