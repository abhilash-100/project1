# HSSAN Flower Classification Platform

A full-stack web application for AI-powered flower species classification using the Hybrid Spectral-Spatial Attention Network (HSSAN) model.

## Overview

This application enables users to upload flower images and receive accurate species predictions. The system features:

- User authentication with Supabase
- Secure image storage
- AI-powered classification with confidence scores
- Classification history tracking
- Beautiful, responsive UI with Tailwind CSS

## Architecture

### Frontend
- React 18 with TypeScript
- Vite for fast development and optimized builds
- Tailwind CSS for styling
- Supabase client for authentication and data management
- Lucide React for icons

### Backend
- FastAPI (Python) for the API server
- TensorFlow/Keras for the HSSAN model
- Supports 102 flower species (Oxford 102 Flowers dataset)

### Database & Storage
- Supabase PostgreSQL database
- Row Level Security (RLS) for data privacy
- Secure file storage with user-scoped access

## Getting Started

### Frontend Setup

1. The project is already configured with all dependencies installed
2. Start the development server:
```bash
npm run dev
```

3. Build for production:
```bash
npm run build
```

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## Database Schema

### `uploads` Table
Stores information about uploaded images:
- `id`: Unique identifier
- `user_id`: Links to authenticated user
- `image_path`: Storage path for the image
- `status`: Processing status (processing/completed/failed)
- `created_at`: Upload timestamp

### `predictions` Table
Stores classification results:
- `id`: Unique identifier
- `upload_id`: Links to the upload
- `predicted_class`: Flower species name
- `confidence_score`: Prediction confidence (0-1)
- `is_top_prediction`: Marks the highest confidence prediction
- `created_at`: Prediction timestamp

### Storage Bucket
- `flower_images`: Secure bucket for uploaded images
- User-scoped access with path structure: `{user_id}/{filename}`

## HSSAN Model Architecture

The Hybrid Spectral-Spatial Attention Network consists of:

### 1. Spectral-Spatial Feature Extractor (3D CNN)
- Processes hyperspectral data across spatial and spectral dimensions
- Two 3D convolutional layers with max pooling
- Captures subtle features invisible in RGB images

### 2. Spatial Feature Refiner (Inception + Attention)
- Multi-scale feature extraction using Inception blocks
- Processes features at multiple receptive field sizes
- Squeeze-and-Excitation (SE) blocks for channel attention
- Focuses on the most discriminative features

### 3. Lightweight Classifier
- Global Average Pooling to reduce parameters
- Dropout (0.4) for regularization
- Softmax output for 102 flower classes

### Current Implementation
The deployed model uses an RGB-adapted version (`build_hssan_rgb_model`) that:
- Accepts standard RGB images (224x224x3)
- Replaces 3D CNN with 2D convolutions
- Maintains the Inception-SE architecture for spatial features
- Can be upgraded to full hyperspectral when HSI data is available

## Training the Model

To train the HSSAN model:

1. Download a flower dataset (e.g., Oxford 102 Flowers)
2. Create a training script using the model architecture in `backend/model/hssan.py`
3. Train the model and save weights:
```python
from model.hssan import build_hssan_rgb_model

model = build_hssan_rgb_model(num_classes=102)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train with your dataset
model.fit(train_data, epochs=50, validation_data=val_data)

# Save the trained model
model.save('backend/model/hssan_rgb_model.h5')
```

4. The API will automatically load the trained model on startup

## API Endpoints

### `GET /`
Returns API information and status

### `GET /health`
Health check endpoint

### `POST /predict`
Upload an image for classification
- Accepts: multipart/form-data with image file
- Returns: Top 3 predictions with confidence scores

### `GET /classes`
Returns list of all 102 flower species

## Features

### Authentication
- Email/password authentication via Supabase
- Secure session management
- Protected routes and API endpoints

### Image Upload & Classification
- Drag-and-drop or click to upload
- Real-time preview
- Progress indicators
- Top 3 predictions with confidence percentages
- Visual confidence bars

### Classification History
- Grid view of all past classifications
- Image thumbnails
- Top prediction display
- Confidence scores
- Timestamps

### Security
- Row Level Security (RLS) ensures users can only access their own data
- Storage policies restrict file access to owners
- Authentication required for all operations
- Secure API communication

## Environment Variables

### Frontend (.env)
```
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_API_URL=http://localhost:8000
```

### Backend
No configuration needed - runs on localhost:8000 by default

## Deployment

### Frontend
Deploy to Vercel, Netlify, or similar platforms:
1. Build the project: `npm run build`
2. Deploy the `dist` folder
3. Update `VITE_API_URL` to point to your deployed API

### Backend
Deploy to Render, Fly.io, or similar Python hosting:
1. Ensure `requirements.txt` includes all dependencies
2. Upload trained model weights
3. Set appropriate environment variables
4. Deploy with Python 3.10+

### Model Weights
Store trained model weights in cloud storage (S3, GCS) and download on startup, or include them in your Docker image.

## Future Enhancements

1. **Hyperspectral Support**: Implement full HSI data processing for maximum accuracy
2. **Class Activation Maps**: Visualize which regions influenced predictions
3. **Batch Upload**: Process multiple images simultaneously
4. **Export Results**: Download classification reports
5. **Model Versioning**: Track and compare different model versions
6. **Social Features**: Share classifications with other users
7. **Mobile App**: Native iOS/Android applications

## Flower Species (102 Classes)

The model recognizes 102 flower species including:
- Pink primrose, Hard-leaved pocket orchid, Canterbury bells
- Sweet pea, English marigold, Tiger lily, Moon orchid
- Bird of paradise, Sunflower, Rose, Lotus, Hibiscus
- And 90 more species...

See `backend/model/flower_classes.py` for the complete list.

## Technical Highlights

- **Attention Mechanisms**: SE blocks improve feature representation
- **Multi-scale Processing**: Inception architecture captures diverse patterns
- **Lightweight Design**: Global Average Pooling reduces overfitting
- **Scalable Architecture**: Easy to extend to more classes
- **Production Ready**: Full error handling, loading states, and user feedback

## Support

For issues or questions, please refer to:
- Frontend documentation in component files
- Backend API documentation at `/docs` endpoint (FastAPI auto-generated)
- Model architecture in `backend/model/hssan.py`
