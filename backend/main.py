from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import os
from typing import List, Dict
import tensorflow as tf
from model.flower_classes import get_flower_name, get_total_classes

app = FastAPI(title="HSSAN Flower Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "model/hssan_rgb_model.h5")
model = None


def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={'SqueezeExcitation': None},
                compile=False
            )
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
            print("Starting with placeholder model. Train and upload a model to enable predictions.")
            from model.hssan import build_hssan_rgb_model
            model = build_hssan_rgb_model(num_classes=get_total_classes())
            print("Placeholder model created.")
    except Exception as e:
        print(f"Error loading model: {e}")
        from model.hssan import build_hssan_rgb_model
        model = build_hssan_rgb_model(num_classes=get_total_classes())


@app.on_event("startup")
async def startup_event():
    load_model()


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.get("/")
async def root():
    return {
        "message": "HSSAN Flower Classification API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "total_classes": get_total_classes()
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image, verbose=0)
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]

        results = []
        for idx in top_3_indices:
            results.append({
                "class_index": int(idx),
                "class_name": get_flower_name(int(idx)),
                "confidence": float(predictions[0][idx])
            })

        return {
            "success": True,
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/classes")
async def get_classes():
    from model.flower_classes import FLOWER_CLASSES
    return {
        "total": len(FLOWER_CLASSES),
        "classes": FLOWER_CLASSES
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
