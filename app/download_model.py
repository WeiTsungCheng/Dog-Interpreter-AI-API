
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_NAME = "Salesforce/blip-image-captioning-base"
LOCAL_MODEL_PATH = "./models/blip-base"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME)
processor.save_pretrained(LOCAL_MODEL_PATH)
model.save_pretrained(LOCAL_MODEL_PATH)

print("Model saved to", LOCAL_MODEL_PATH)