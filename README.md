# HSSAN Flower Classification Application

A sophisticated AI-powered flower classification system that combines deep learning with intelligent knowledge management using Google's Gemini API and Supabase.

## 🌟 Overview

This application uses a Hybrid Spatial-Spectral Attention Network (HSSAN) to classify flower species from images, and intelligently builds a comprehensive knowledge base that grows smarter with every user interaction.

## 🏗️ Architecture

### Frontend (React + TypeScript + Vite)

#### `src/main.tsx`
- Application entry point
- Renders the root React component with StrictMode

#### `src/App.tsx`
- Main application component
- Manages view routing between Upload and History pages
- Wraps application in AuthProvider for authentication context
- Displays loading states and conditional rendering based on auth status

#### `src/index.css`
- Global styles and Tailwind CSS imports
- Defines application-wide styling

#### `src/contexts/AuthContext.tsx`
- Provides authentication context throughout the application
- Manages user session state with Supabase Auth
- Implements signUp, signIn, and signOut methods
- Handles auth state changes properly using async blocks to avoid deadlocks

#### `src/lib/supabase.ts`
- Initializes and exports the Supabase client
- Configured with environment variables for URL and anonymous key

#### `src/components/Navbar.tsx`
- Navigation bar component
- Allows switching between Upload and History views
- Displays user email and sign-out button
- Responsive design with gradient styling

#### `src/components/AuthForm.tsx`
- Authentication form component
- Handles both sign-up and sign-in functionality
- Includes form validation and error handling
- Beautiful gradient design with loading states

#### `src/components/UploadView.tsx`
- **Core classification interface**
- File upload with image preview
- Calls Python backend for AI classification
- Calls Supabase Edge Function for intelligent flower information retrieval
- Displays comprehensive flower information including:
  - Scientific and common names
  - Description and botanical properties
  - Common uses and care instructions
  - Visual states and toxicity information
- **Interactive Q&A Feature**: Users can ask questions about the flower
- Real-time status updates (processing, completed, failed)

#### `src/components/HistoryView.tsx`
- Displays user's classification history
- Real-time updates using Supabase subscriptions
- Automatically refreshes when new uploads are completed
- Shows images, predictions, confidence scores, and timestamps
- Fetches images from Supabase storage

### Backend (Python + FastAPI)

#### `backend/main.py`
- FastAPI server for AI model inference
- CORS enabled for cross-origin requests
- Endpoints:
  - `GET /` - API information and model status
  - `GET /health` - Health check
  - `POST /predict` - Classify flower image
  - `GET /classes` - List all flower classes
- Image preprocessing (resize to 224x224, normalize to 0-1)
- Returns top 3 predictions with confidence scores

#### `backend/model/hssan.py`
- **CONTAINS THE HSSAN ATTENTION MECHANISM**
- Implements the Hybrid Spatial-Spectral Attention Network architecture
- **Key Components**:

  **1. SqueezeExcitation (Attention Mechanism) - Lines 5-31**:
  - Implements channel attention mechanism
  - Applies global average pooling to capture channel statistics
  - Uses two dense layers for channel recalibration
  - Multiplies input features with learned attention weights
  - Ratio parameter (default 16) controls dimensionality reduction

  **2. Inception Block - Lines 34-48**:
  - Multi-scale feature extraction using parallel convolutions (1x1, 3x3, 5x5)
  - Includes max pooling branch for spatial information
  - Concatenates all branches for rich feature representation

  **3. HSSAN Model Architectures**:
  - `build_hssan_model()`: Full hyperspectral version (224x224x150 input)
  - `build_hssan_rgb_model()`: RGB version (224x224x3 input) - Currently used

  **Model Pipeline**:
  - Convolutional layers for feature extraction
  - Inception blocks for multi-scale processing
  - **Squeeze-Excitation blocks for attention-based feature recalibration**
  - Global average pooling for spatial aggregation
  - Dropout (0.4) for regularization
  - Softmax output for class probabilities

#### `backend/model/flower_classes.py`
- Defines all 102 flower species classes
- Provides utility functions for class name mapping
- Maps class indices to readable flower names

#### `backend/requirements.txt`
- Python dependencies:
  - fastapi: Web framework
  - uvicorn: ASGI server
  - tensorflow: Deep learning framework
  - pillow: Image processing
  - numpy: Numerical operations

### Supabase Edge Functions (Deno + TypeScript)

#### `supabase/functions/classify-flower/index.ts`
- **Intelligent classification workflow**
- Receives uploadId and scientificName from frontend
- **Step 1**: Checks Supabase `flowers` table for existing information
- **Step 2a**: If found, returns cached data immediately (no API call)
- **Step 2b**: If new flower:
  - Calls Google Gemini API with detailed prompt
  - Requests comprehensive information:
    - Common names
    - Description
    - Botanical properties (family, genus, native region, bloom season, growth habit)
    - Common uses
    - Visual states (healthy, wilted, damaged, diseased)
    - Care instructions
    - Toxicity information (pets and humans)
  - Parses JSON response from Gemini
  - Stores in Supabase `flowers` table for future use
- **Step 3**: Updates upload status to 'completed'
- Returns flower information to frontend
- Includes comprehensive error handling

#### `supabase/functions/ask-question/index.ts`
- **Interactive Q&A system**
- Receives scientificName and question from user
- **Step 1**: Retrieves flower from database
- **Step 2**: Checks existing Q&A array for cached answer
- **Step 3a**: If question was asked before, returns cached answer
- **Step 3b**: If new question:
  - Calls Google Gemini API with the question
  - Gets accurate answer
  - Appends Q&A pair to flower's q_and_a JSON field
  - Updates the database
- Returns answer to user
- Database becomes smarter with every question

### Database (Supabase PostgreSQL)

#### `supabase/migrations/20251016060915_apply_existing_schema.sql`
- Creates `uploads` table:
  - Tracks all user image uploads
  - Fields: id, user_id, created_at, image_path, status
  - Status: 'processing', 'completed', or 'failed'
  - RLS policies: Users can only access their own uploads
- Creates `predictions` table:
  - Stores classification results
  - Fields: id, upload_id, predicted_class, confidence_score, is_top_prediction, created_at
  - Links to uploads via foreign key
  - RLS policies: Users can only see predictions for their own uploads
- Creates indexes for optimal query performance
- Implements Row Level Security for data privacy

#### `supabase/migrations/20251016060946_create_flowers_knowledge_base.sql`
- Creates `flowers` table (intelligent knowledge base):
  - Fields:
    - scientific_name (unique): Primary lookup key
    - common_names (text[]): Array of common names
    - description (text): Full flower description
    - botanical_properties (jsonb): Structured botanical data
    - common_uses (text[]): Uses array
    - visual_states (jsonb): Appearance in different conditions
    - care_instructions (text): Care guide
    - toxicity_info (jsonb): Safety information
    - q_and_a (jsonb): Array of user questions and answers
    - created_at, updated_at: Timestamps
    - source: Data source (gemini_api, manual)
  - RLS policies: All authenticated users can read, system can write
  - Grows smarter with every classification and question

#### Storage Bucket: `flower_images`
- Stores user-uploaded flower images
- Organized by user_id folders
- File size limit: 10MB
- Allowed formats: JPEG, JPG, PNG, WEBP
- RLS policies: Users can upload, view, and delete their own images

## 🔄 Intelligent Workflow

### Classification Process

1. **User Uploads Image**
   - Image is validated and previewed
   - Uploaded to Supabase storage at `{user_id}/{timestamp}.{ext}`
   - Upload record created with status 'processing'

2. **AI Classification**
   - Image sent to Python backend (`/predict` endpoint)
   - HSSAN model processes image
   - Returns top 3 predictions with confidence scores
   - Predictions stored in database

3. **Smart Information Retrieval**
   - Frontend calls `classify-flower` Edge Function
   - **Fast Path**: If flower exists in database, return immediately
   - **Slow Path**: If new flower:
     - Call Gemini API for comprehensive research
     - Parse and structure the information
     - Store in database for future use
   - Upload status updated to 'completed'
   - Rich information displayed to user

4. **Q&A Enhancement**
   - User asks question about the flower
   - Frontend calls `ask-question` Edge Function
   - **Fast Path**: If question was asked before, return cached answer
   - **Slow Path**: If new question:
     - Call Gemini API for answer
     - Store Q&A pair in database
   - Answer displayed immediately
   - Database becomes smarter

### Why This Approach is Efficient

- **First classification is slow** (calls Gemini API)
- **All subsequent classifications are instant** (cached in database)
- **Common questions get answered instantly** (cached Q&A)
- **Database grows organically** with user interactions
- **Reduces API costs** significantly over time
- **Improves response time** as the system learns

## 🔑 Configuration

### Required Environment Variables

Create `.env` file:
```env
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_API_URL=http://localhost:8000
```

### Gemini API Key Setup

The Edge Functions require a Gemini API key:

1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add to Supabase:
   - Go to Supabase Dashboard
   - Project Settings > Edge Functions > Secrets
   - Add secret: `GEMINI_API_KEY` = your_key
3. The application will automatically use this key

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- Python 3.8+
- Supabase account

### Frontend Setup
```bash
npm install
npm run dev
```

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```

The backend runs on `http://localhost:8000`

### Database Setup
All migrations are already applied. The database includes:
- Authentication tables (built-in Supabase)
- `uploads` table
- `predictions` table
- `flowers` knowledge base table
- `flower_images` storage bucket

## 🎯 Key Features

1. **AI-Powered Classification**: HSSAN model with attention mechanism
2. **Intelligent Caching**: Database-first approach minimizes API calls
3. **Real-Time Updates**: History page updates automatically
4. **Interactive Q&A**: Ask questions about any classified flower
5. **Growing Knowledge Base**: System becomes smarter with use
6. **Secure Authentication**: Row-level security on all data
7. **Comprehensive Information**: Detailed flower descriptions, care, toxicity
8. **Beautiful UI**: Gradient design with smooth transitions

## 🔧 Troubleshooting

### Blank Preview Issue
If you see a blank page:
1. Check browser console for errors
2. Verify Supabase environment variables are set
3. Ensure you're signed in (auth required)
4. Clear browser cache and reload

### "Failed to fetch" Error
This is now fixed by:
- Proper error handling in Edge Functions
- Status updates throughout the process
- Fallback error messages

### History Not Updating
Now fixed with:
- Real-time Supabase subscriptions
- Automatic refresh on upload completion
- Proper cleanup of subscriptions

## 📊 The HSSAN Attention Mechanism

Located in `backend/model/hssan.py`, the **Squeeze-Excitation (SE) block** implements the attention mechanism:

**How it works**:
1. **Global Average Pooling**: Captures channel-wise statistics
2. **Dimensionality Reduction**: Dense layer reduces channels by ratio (16)
3. **Activation**: ReLU introduces non-linearity
4. **Dimensionality Expansion**: Dense layer expands back to original channels
5. **Sigmoid Activation**: Creates attention weights (0-1)
6. **Feature Recalibration**: Multiplies input with learned weights

This allows the model to focus on the most informative features for classification.

## 📁 Project Structure Summary

```
├── src/
│   ├── main.tsx                 # Entry point
│   ├── App.tsx                  # Main app component with routing
│   ├── index.css                # Global styles
│   ├── components/
│   │   ├── AuthForm.tsx         # Sign in/up interface
│   │   ├── Navbar.tsx           # Navigation
│   │   ├── UploadView.tsx       # Classification interface (main feature)
│   │   └── HistoryView.tsx      # Classification history with real-time updates
│   ├── contexts/
│   │   └── AuthContext.tsx      # Authentication state management
│   └── lib/
│       └── supabase.ts          # Supabase client initialization
├── backend/
│   ├── main.py                  # FastAPI server with prediction endpoint
│   ├── requirements.txt         # Python dependencies
│   └── model/
│       ├── hssan.py             # 🎯 ATTENTION MECHANISM HERE
│       └── flower_classes.py    # Flower species definitions
├── supabase/
│   ├── migrations/
│   │   ├── ...apply_existing_schema.sql      # Core tables
│   │   └── ...create_flowers_knowledge_base.sql  # Smart cache
│   └── functions/
│       ├── classify-flower/     # Intelligent classification workflow
│       └── ask-question/        # Q&A with caching
└── SETUP.md                     # Configuration guide
```

## 🔐 Security

- Row Level Security enabled on all tables
- Users can only access their own data
- Edge Functions validate JWT tokens
- API keys stored securely in Supabase secrets
- No sensitive data exposed to frontend

## 🎓 Credits

This application uses the HSSAN (Hybrid Spatial-Spectral Attention Network) architecture for flower classification, combining inception modules with squeeze-excitation attention mechanisms for superior accuracy.

---

Built with ❤️ using React, TypeScript, FastAPI, TensorFlow, Supabase, and Google Gemini API
