# Training Guide - Adding Hugging Face Datasets

## Quick Start

### Step 1: Find Datasets on Hugging Face

Visit https://huggingface.co/datasets and search for:
- "flower classification"
- "plant recognition"
- "botanical images"

### Step 2: Get Parquet URL Format

For any Hugging Face dataset, the parquet URL follows this pattern:

```
hf://datasets/USERNAME/DATASET-NAME/data/FILENAME.parquet
```

## Example Datasets

### 1. Flowers 102 Categories
```
hf://datasets/huggan/flowers-102-categories/data/train-00000-of-00001.parquet
```
- 102 flower species
- High-quality images
- Well-labeled data

### 2. Additional Datasets (Examples)

You can find more datasets by:
1. Go to https://huggingface.co/datasets
2. Search for "flowers" or "plants"
3. Click on a dataset
4. Go to "Files and versions" tab
5. Navigate to `data/` folder
6. Copy the path to `.parquet` files

## Adding Datasets to Your Training

### Via UI (Recommended)

1. Open the application
2. Navigate to **Training** tab
3. Click **Add Dataset**
4. Fill in:
   - **Name**: Descriptive name (e.g., "Flowers 102")
   - **Dataset URL**: The full `hf://` URL
   - **Description**: Optional notes
5. Click **Add Dataset**
6. Toggle the dataset to **Active**

### Via Database (Advanced)

```sql
INSERT INTO training_datasets (user_id, name, dataset_url, description, is_active)
VALUES (
  'your-user-id',
  'Flowers 102',
  'hf://datasets/huggan/flowers-102-categories/data/train-00000-of-00001.parquet',
  'Oxford 102 Flowers dataset',
  true
);
```

## Training with Multiple Datasets

You can combine multiple datasets:

1. Add all datasets via UI
2. Set all to **Active**
3. Start a training job
4. The system will automatically combine all active datasets

Example: Train on 3 different flower datasets simultaneously for better generalization.

## Dataset Requirements

Your dataset parquet file must have:

### Required Columns
- **image**: Image data in one of these formats:
  - Bytes: Raw image bytes
  - Path: Local file path
  - URL: HTTP/HTTPS image URL

- **label** (or **category** or **class**): String label for the flower class

### Example Dataset Structure

```python
import pandas as pd

# Valid dataset format
df = pd.DataFrame({
    'image': [image_bytes_1, image_bytes_2, ...],
    'label': ['rose', 'tulip', 'sunflower', ...]
})
```

## Training Configuration

When you start a training job, you can configure:

### Hyperparameters

- **Batch Size** (default: 32)
  - Larger = faster but more memory
  - Smaller = slower but less memory
  - Try: 16, 32, 64

- **Learning Rate** (default: 0.0001)
  - Higher = faster learning but may be unstable
  - Lower = stable but slower
  - Try: 0.001, 0.0001, 0.00001

- **Epochs** (default: 10)
  - More epochs = better fit but may overfit
  - Fewer epochs = faster training
  - Try: 10, 20, 50

- **Use Pre-trained** (default: true)
  - ✅ Enabled: Uses MobileNetV2 (faster, better accuracy)
  - ❌ Disabled: Trains from scratch (slower, needs more data)

## Running Training

### Method 1: Simple Training
```bash
cd backend
python train.py
```

This uses the default configuration.

### Method 2: Custom Configuration
```bash
cd backend
python -c "from train import train_from_config; train_from_config('training_configs/example_config.json')"
```

### Method 3: Interactive Python
```python
from backend.train import FlowerTrainer

config = {
    'batch_size': 32,
    'learning_rate': 0.0001,
    'num_epochs': 10,
    'use_pretrained': True
}

trainer = FlowerTrainer(config)

# Load single dataset
num_classes = trainer.load_dataset('hf://datasets/huggan/flowers-102-categories/data/train-00000-of-00001.parquet')

# Or load multiple datasets
num_classes = trainer.load_multiple_datasets([
    'hf://datasets/huggan/flowers-102-categories/data/train-00000-of-00001.parquet',
    'hf://datasets/another-dataset/data/train.parquet'
])

# Build and train model
trainer.build_model(num_classes)
history = trainer.train(num_epochs=10)

# Save model
trainer.save_model('my_custom_model.pth')
```

## Monitoring Training Progress

### In Terminal
Watch the output:
```
Epoch [1/10] Train Loss: 2.3451 | Train Acc: 45.23% | Val Loss: 1.9876 | Val Acc: 52.10%
Epoch [2/10] Train Loss: 1.8765 | Train Acc: 58.91% | Val Loss: 1.6543 | Val Acc: 63.45%
...
```

### In UI (Coming Soon)
The Training tab will show real-time updates once the training script is connected to the database.

## Output Files

After training completes:

```
backend/
├── models/
│   ├── flower_model.pth          # Trained model weights
│   ├── best_flower_model.pth     # Best checkpoint
│   └── class_names.json          # Class name mapping
└── training_configs/
    └── your_config.json
```

## Tips for Best Results

1. **Use Pre-trained Models**: Always enable "Use Pre-trained" unless you have 10,000+ images
2. **Combine Datasets**: More diverse data = better generalization
3. **Monitor Validation Accuracy**: If it stops improving, stop training (early stopping)
4. **GPU Recommended**: Training on CPU is slow. Use Google Colab or cloud GPU
5. **Start Small**: Train for 5 epochs first, then increase if needed

## Troubleshooting

### Error: "Dataset URL not accessible"
- Check the URL is correct
- Ensure you have internet connection
- Some datasets may require Hugging Face authentication token

### Error: "Out of Memory"
- Reduce batch size (try 16 or 8)
- Use CPU if GPU memory is insufficient
- Close other applications

### Error: "Column 'label' not found"
- Check your dataset has 'label', 'category', or 'class' column
- The script auto-detects these names

### Training is Very Slow
- Enable GPU if available
- Increase batch size
- Use pre-trained model (much faster)

## Adding Private Datasets

If you have a private Hugging Face dataset:

1. Get your token from https://huggingface.co/settings/tokens
2. Set environment variable:
   ```bash
   export HF_TOKEN=your_token_here
   ```
3. The training script will automatically use it

## Example: Complete Workflow

```bash
# 1. Start the application
npm run dev

# 2. Add datasets via UI
# - Open Training tab
# - Add Flowers 102 dataset
# - Add any other datasets
# - Set all to Active

# 3. Create training job via UI
# - Click "Start Training"
# - Configure parameters
# - Click "Start Training"

# 4. Run training script
cd backend
python train.py

# 5. Wait for completion
# Watch terminal for progress

# 6. Model saved automatically
# Find it in backend/models/
```

## Next Steps

After training:
1. Update the inference API to use your new model
2. Test predictions with new images
3. Deploy to production
4. Collect more data and retrain

Happy training!
