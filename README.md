# VL-JEPA Implementation

A PyTorch implementation of Vision-Language Joint Embedding Predictive Architecture (VL-JEPA) inspired by Meta's research. This project demonstrates how to build a multimodal learning system that predicts text embeddings from visual inputs.

## Overview

This implementation combines vision and language models to create a unified embedding space. The architecture uses:

- **Vision Encoder**: CLIP ViT-B/16 model for extracting visual features
- **Text Encoder**: Sentence Transformers (all-MiniLM-L6-v2) for text embeddings
- **Predictor Network**: A lightweight transformer-based MLP to predict text embeddings from visual embeddings

The system learns to predict corresponding text embeddings from image embeddings using cosine similarity loss, enabling cross-modal retrieval and zero-shot classification tasks.

## Architecture

```
Image/Video Input
    ↓
Vision Encoder (CLIP ViT-B/16)
    ↓
Visual Embedding (512-dim)
    ↓
Predictor Network (Small Transformer MLP)
    ↓
Predicted Text Embedding (384-dim)
    ↓
Compare with Ground Truth Text Embedding
    ↓
(Optional) Decode with LLM
```

## Project Structure

```
vl-jepa-implementation/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── architecture.txt          # Architecture overview
├── train.py                  # Training script
├── eval.py                   # Evaluation/inference script
├── test.ipynb               # Jupyter notebook for testing
├── caption.csv              # Full dataset with image-caption pairs
├── unique_caption.csv       # Deduplicated captions for training
├── models/
│   ├── __init__.py
│   ├── vision.py            # Vision encoder implementation
│   ├── text.py              # Text encoder implementation
│   ├── predictor.py         # Predictor network implementation
│   └── dataset.py           # Custom dataset loader
└── my_image_archive/        # Image files for training/evaluation
```

## Installation

### Prerequisites
- Python 3.7+
- CUDA 11.0+ (recommended for GPU acceleration)
- Git

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd vl-jepa-implementation
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- **torch**: Deep learning framework
- **pandas**: Data manipulation and CSV handling
- **Pillow**: Image processing
- **sentence-transformers**: Text encoding models
- **clip**: Vision-language model from OpenAI

## Usage

### Training

Run the training script to train the predictor network:

```bash
python train.py
```

**Training Details:**
- **Dataset**: Uses `unique_caption.csv` with image paths and captions
- **Batch Size**: 16
- **Epochs**: 10
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW
- **Loss Function**: 1 - Cosine Similarity (encourages predicted embeddings to align with ground truth)
- **Device**: Automatically uses CUDA if available, otherwise CPU

The script:
1. Loads vision and text encoders
2. Creates a dataset from CSV with image-caption pairs
3. Trains the predictor network to predict text embeddings from vision embeddings
4. Prints loss for each epoch

### Evaluation/Inference

Run the evaluation script to perform inference on a test image:

```bash
python eval.py
```

**Example Usage:**
The script:
1. Loads pre-trained encoders and the predictor network
2. Takes an image from `my_image_archive/10002456.jpg`
3. Encodes the image to get visual embedding
4. Predicts the corresponding text embedding
5. Compares against candidate labels using cosine similarity
6. Returns the most similar label and similarity scores

**Output Example:**
```
Prediction: persons hanging bridge and cable
Scores: [0.85, 0.45, 0.32]
```

### Interactive Testing

Use the Jupyter notebook `test.ipynb` for interactive exploration:

```bash
jupyter notebook test.ipynb
```

## Model Components

### VisionEncoder (`models/vision.py`)
- Uses OpenAI's CLIP ViT-B/16 model
- Outputs normalized 512-dimensional embeddings
- Pre-trained and frozen during training
- Includes image preprocessing pipeline

**Class Methods:**
- `__init__(device)`: Initialize with specified device
- `encode(images)`: Encode batch of images to embeddings

### TextEncoder (`models/text.py`)
- Uses Sentence Transformers all-MiniLM-L6-v2 model
- Outputs normalized 384-dimensional embeddings
- Pre-trained and frozen during training
- Supports batch text encoding

**Class Methods:**
- `__init__(device)`: Initialize with specified device
- `encode(texts)`: Encode batch of text strings to embeddings

### Predictor (`models/predictor.py`)
- Lightweight 2-layer MLP with GELU activation
- Maps 512-dim visual embeddings to 384-dim text embeddings
- Trainable component of the architecture
- L2 normalization on output

**Architecture:**
```
Input (512-dim) → Linear → GELU → Linear → L2 Norm → Output (384-dim)
```

### VisionTextDataset (`models/dataset.py`)
- PyTorch Dataset for loading image-caption pairs from CSV
- Automatically applies vision preprocessing to images
- Handles image loading and format conversion

**CSV Format Required:**
```
image_name,comment
10002456.jpg,persons hanging bridge and cable
```

## Dataset Format

Your dataset should be organized as:
- **CSV File**: Contains columns `image_name` and `comment`
- **Images**: Located in `my_image_archive/` directory
- **Path Format**: Image filenames match the `image_name` column in CSV

Example CSV structure:
```csv
image_name,comment
image1.jpg,A person walking in the park
image2.jpg,Two cats playing together
image3.jpg,Sunset over the ocean
```

## Training Tips

1. **Data Preparation**: Ensure all image files exist and CSV paths are correct
2. **GPU Memory**: Adjust batch size if encountering OOM errors
3. **Learning Rate**: Default (1e-4) works well; adjust if loss isn't decreasing
4. **Epochs**: Increase for more training (default is 10)
5. **Evaluation**: Periodically save checkpoints and evaluate on validation set

## Advanced Usage

### Custom Training Loop

```python
from models.vision import VisionEncoder
from models.text import TextEncoder
from models.predictor import Predictor
from models.dataset import VisionTextDataset
from torch.utils.data import DataLoader
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize models
vision = VisionEncoder(device=device)
text = TextEncoder(device=device)
predictor = Predictor().to(device)

# Load dataset
dataset = VisionTextDataset("unique_caption.csv", vision.preprocess)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training loop
optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-4)

for epoch in range(10):
    for images, captions in dataloader:
        images = images.to(device)
        
        with torch.no_grad():
            vision_emb = vision.encode(images)
            text_emb = text.encode(captions).to(device)
        
        pred = predictor(vision_emb)
        loss = 1 - torch.cosine_similarity(pred, text_emb, dim=-1).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Zero-Shot Classification

```python
# For classification without seeing those specific labels during training
labels = ["object1", "object2", "object3"]
images = [load_image("path/to/image.jpg")]

vision_emb = vision.encode(images)
pred_emb = predictor(vision_emb)

text_embs = text.encode(labels)
scores = torch.cosine_similarity(pred_emb, text_embs, dim=-1)

predicted_label = labels[scores.argmax().item()]
```

## Performance Considerations

- **Vision Encoder**: ~500MB (ViT-B/16 model)
- **Text Encoder**: ~100MB (MiniLM model)
- **Predictor Network**: Minimal (~5MB)
- **Training Time**: ~2-3 minutes per epoch on NVIDIA GPU

## Troubleshooting

**Issue: CUDA out of memory**
- Reduce batch size in training script
- Use CPU mode (automatically fallback)

**Issue: Missing image files**
- Verify image paths in CSV match actual files in `my_image_archive/`
- Check file extensions match exactly

**Issue: Poor predictions**
- Ensure sufficient training data
- Increase number of training epochs
- Check that image-caption pairs are semantically aligned

## Future Improvements

- [ ] Add validation/test split evaluation
- [ ] Implement model checkpointing and resuming
- [ ] Add learning rate scheduling
- [ ] Support for additional vision/text encoders
- [ ] Attention visualization for interpretability
- [ ] Support for video inputs
- [ ] Multi-GPU training support
- [ ] Model quantization for deployment

## References

- CLIP: https://github.com/openai/CLIP
- Sentence Transformers: https://www.sbert.net/
- VL-JEPA: Meta's Vision-Language Joint Embedding Predictive Architecture
- PyTorch: https://pytorch.org/

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or issues.

## Contact

For questions or suggestions, please open an issue on the repository.
