# Flower Classification Training Platform

A full-stack application for training custom flower classification models using Hugging Face datasets and PyTorch, with an interactive UI for managing datasets and monitoring training jobs.

## Features

- **Interactive UI**: Upload flower images and get instant predictions
- **Training Manager**: Add multiple Hugging Face dataset URLs and train custom models
- **Real-time Monitoring**: Track training progress with live updates
- **Database Integration**: Supabase for secure data storage and authentication
- **PyTorch Backend**: Flexible training pipeline with MobileNetV2 or custom CNNs

## Architecture

### Frontend (React + TypeScript)
- **Upload View**: Classify flower images with AI
- **History View**: View past classifications with real-time updates
- **Training View**: Manage datasets and start training jobs

### Backend (Python + FastAPI + PyTorch)
- **Inference API**: Serves trained models for classification
- **Training Pipeline**: Loads Hugging Face datasets and trains models
- **Flexible Configuration**: Support for multiple datasets and hyperparameters

### Database (Supabase PostgreSQL)
- **Users & Authentication**: Secure user management
- **Training Datasets**: Store Hugging Face dataset URLs
- **Training Jobs**: Track training progress and metrics
- **Trained Models**: Store model metadata and performance

## Quick Start

### 1. Install Dependencies

```bash
# Frontend
npm install

# Backend
cd backend
pip install -r requirements.txt
cd ..
```

### 2. Configure Environment

The `.env` file is already configured with Supabase credentials.

### 3. Start the Application

**Terminal 1 - Frontend:**
```bash
npm run dev
```

**Terminal 2 - Backend (Inference API):**
```bash
cd backend
python main.py
```

Open http://localhost:5173 in your browser.

## Training Custom Models

### Add Hugging Face Datasets

1. Navigate to the **Training** tab
2. Click **Add Dataset**
3. Enter:
   - **Dataset Name**: e.g., "Flowers 102"
   - **Dataset URL**: Hugging Face parquet URL
   - **Description**: Optional notes

Example URLs:
```
hf://datasets/huggan/flowers-102-categories/data/train-00000-of-00001.parquet
```

### Configure and Start Training

1. Click **Start Training**
2. Configure:
   - **Job Name**: e.g., "Flower Model v1"
   - **Batch Size**: 32 (default)
   - **Learning Rate**: 0.0001 (default)
   - **Epochs**: 10 (default)
   - **Use Pre-trained**: Enable for MobileNetV2 (recommended)
3. Click **Start Training**

### Run Training Script

The UI creates training job records in the database. To actually train:

**Option 1: Using Training Configuration**
```bash
cd backend
python train.py
```

**Option 2: Custom Configuration**

Create `backend/training_configs/custom_config.json`:
```json
{
  "dataset_urls": [
    "hf://datasets/huggan/flowers-102-categories/data/train-00000-of-00001.parquet",
    "hf://datasets/another-dataset/data/train.parquet"
  ],
  "batch_size": 32,
  "learning_rate": 0.0001,
  "num_epochs": 10,
  "use_pretrained": true,
  "model_name": "my_flower_model.pth"
}
```

Then run:
```bash
python -c "from train import train_from_config; train_from_config('backend/training_configs/custom_config.json')"
```

## Training Pipeline Details

### Supported Dataset Formats

The training pipeline expects parquet files with:
- **image** column: Image data (bytes, path, or URL)
- **label/category/class** column: Flower class name

### Model Architecture

**Pre-trained (Recommended):**
- Base: MobileNetV2
- Custom head: Dropout(0.4) + Dense(num_classes)
- Fast training, high accuracy

**Custom CNN:**
- 3 Conv2D layers (32, 64, 128 filters)
- MaxPooling, GlobalAvgPooling
- Dense layers with Dropout
- Lightweight, trains from scratch

### Training Process

1. **Data Loading**: Downloads and parses parquet files
2. **Preprocessing**: Resizes to 128x128, normalizes to [-1, 1]
3. **Training**: Adam optimizer, CrossEntropyLoss
4. **Validation**: Tracks accuracy and loss
5. **Checkpointing**: Saves best model automatically

### Outputs

Trained models are saved to `backend/models/`:
- `<model_name>.pth`: Model weights and metadata
- `class_names.json`: Class mapping for inference

## API Endpoints

### Inference API (FastAPI)

**POST /predict**
- Upload image for classification
- Returns top 3 predictions with confidence scores

**GET /classes**
- Returns all supported flower classes

**GET /health**
- Health check and model status

## Database Schema

### training_datasets
- Stores Hugging Face dataset URLs
- User can activate/deactivate datasets
- Multiple datasets can be combined for training

### training_jobs
- Tracks training job status (pending, running, completed, failed)
- Stores configuration and progress
- Real-time updates to UI

### trained_models
- Links to training jobs
- Stores model metadata (accuracy, classes, parameters)
- Marks active model for inference

## Adding More Datasets

To train on additional datasets:

1. **Find Dataset on Hugging Face**:
   - Visit https://huggingface.co/datasets
   - Search for flower or plant datasets
   - Look for datasets with parquet files

2. **Get Parquet URL**:
   - Navigate to "Files and versions"
   - Find `.parquet` files in the data directory
   - Right-click and copy link
   - Format: `hf://datasets/USERNAME/DATASET/data/FILENAME.parquet`

3. **Add to Training Manager**:
   - Open Training tab in UI
   - Click "Add Dataset"
   - Paste URL and configure

4. **Activate and Train**:
   - Toggle dataset to "Active"
   - Multiple active datasets will be combined during training

## Environment Variables

**Frontend (.env):**
```
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_API_URL=http://localhost:8000
```

**Note**: API keys for training are configured through the Training Manager UI, not environment variables.

## GPU Support

The training pipeline automatically uses GPU if available:

```python
# Check GPU availability
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

For GPU training:
- Install CUDA toolkit
- Install PyTorch with CUDA support:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

## Deployment

### Frontend
- Build: `npm run build`
- Deploy `dist/` to Vercel, Netlify, or similar

### Backend
- Deploy to Railway, Render, or AWS
- Ensure GPU instance for faster training
- Store trained models in cloud storage (S3, GCS)

## Troubleshooting

### White Screen / Blank Page
- Check browser console (F12) for errors
- Verify `.env` file has correct Supabase credentials
- Clear browser cache and reload

### Training Not Starting
- Training jobs are created in database but require manual script execution
- Run `python backend/train.py` to process pending jobs

### Out of Memory Errors
- Reduce batch size in training configuration
- Use smaller image size (128x128 instead of 224x224)
- Enable gradient accumulation for larger effective batch sizes

### Dataset Loading Fails
- Verify parquet URL is correct and accessible
- Check dataset has required columns (image, label)
- Some datasets may require Hugging Face authentication token

## Next Steps

1. **Add More Datasets**: Search Hugging Face for flower datasets
2. **Experiment with Hyperparameters**: Try different learning rates and epochs
3. **Deploy Models**: Serve trained models in production
4. **Fine-tune**: Use pre-trained models on domain-specific data

## API Keys Required

**At the end of setup, you'll need to configure:**

1. **Supabase API Keys** (Already in .env)
   - Project URL
   - Anon key

2. **Hugging Face Token** (Optional, for private datasets)
   - Get from: https://huggingface.co/settings/tokens
   - Set as environment variable: `export HF_TOKEN=your_token`

That's it! No other API keys needed. The system is ready to use.

## Support

For issues or questions:
- Check browser console for frontend errors
- Check terminal output for backend errors
- Verify database tables exist in Supabase dashboard
- Ensure all dependencies are installed

Happy training!
