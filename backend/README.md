# HSSAN Flower Classification Backend

This is the FastAPI backend for the HSSAN flower classification system.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model (optional):
```bash
python model/hssan.py
```

This will create an untrained model architecture. To train the model, you'll need to:
- Download a flower dataset (e.g., Oxford 102 Flowers)
- Create a training script
- Train the model and save the weights

3. Place your trained model at `model/hssan_rgb_model.h5`

## Running the API

```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Upload an image and get flower predictions
- `GET /classes` - Get all flower class names

## Model Architecture

The HSSAN (Hybrid Spectral-Spatial Attention Network) model consists of:

1. **3D CNN Module** (for hyperspectral data) - Extracts spectral-spatial features
2. **Inception Blocks** - Multi-scale spatial feature extraction
3. **Squeeze-and-Excitation Blocks** - Channel-wise attention mechanism
4. **Global Average Pooling** - Lightweight classification head

For RGB images, a simplified version is used that replaces the 3D CNN with 2D convolutions.

## Note

The current implementation uses RGB images. To use hyperspectral data:
1. Use the `build_hssan_model()` function in `model/hssan.py`
2. Modify the preprocessing to handle hyperspectral image cubes
3. Retrain with hyperspectral data
