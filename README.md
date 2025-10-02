# AI Image Classifier with ViT and ELA

A web application for image classification using Vision Transformer (ViT) with Error Level Analysis (ELA) preprocessing. This system is designed to detect ai-generated image or classify images based on your custom dataset.

## 🌟 Key Features

- **Custom Model Training**: Train ViT model with your own dataset
- **ELA Preprocessing**: Detect image manipulation using Error Level Analysis
- **Real-time Progress**: Monitor training and evaluation progress in real-time
- **Model Management**: Load and unload models easily
- **Comprehensive Evaluation**: Complete confusion matrix and classification report
- **Fast Inference**: Classify images with confidence scores

## 🏗️ Architecture

- **Model**: Vision Transformer (ViT) with custom configuration
  - Image size: 224x224
  - Patch size: 16x16
  - Hidden size: 384
  - 6 attention heads
  - 6 transformer layers
- **Preprocessing**: Error Level Analysis (ELA) for manipulation detection
- **Framework**: PyTorch + Transformers (Hugging Face)
- **Backend**: Flask
- **Frontend**: HTML + JavaScript (AJAX)

## 🚀 Installation

1. **Clone repository**
```bash
git clone https://github.com/Yuu18id/vit-ela-ai-image-classifier
cd vit-ela-ai-image-classifier
```

2. **Python Dependencies**

For CUDA-enabled systems (GPU):
```
pip install -r requirements-gpu.txt
```

For CPU-only systems:
```
pip install -r requirements-cpu.txt
```

3. **Run the application**
```bash
python app.py
```

The application will run at `http://localhost:5000`

## 📁 Dataset Structure

The dataset must be organized in the following folder structure:

```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── class3/
    ├── image1.jpg
    └── ...
```

## 💻 Usage Guide

### 1. Training Model

1. Open the **Train** page from the navigation menu
2. Select dataset folder using the "Choose Folder" button
3. Configure hyperparameters:
   - **Learning Rate**: Default 1e-6 (recommended 1e-6 to 1e-4)
   - **Epochs**: Number of training epochs (default: 10)
   - **Batch Size**: 
     - GPU: 32 (default), can go higher with more VRAM
     - CPU: 8-16 recommended (lower to avoid memory issues)
   - **Train Split**: Validation data proportion (default: 0.2 = 20%)
4. Click **Start Training**
5. Monitor real-time progress in the log panel

**Training Tips**:
- **GPU users**: 
  - Use batch size 32-64 for optimal speed
  - Enable mixed precision (FP16) - already enabled in code
- **CPU users**: 
  - Use smaller batch size (8-16) to avoid OOM
  - Disable FP16 by editing `app.py`: set `fp16=False` in TrainingArguments
  - Consider using smaller datasets or pre-trained models

### 2. Load Model

After training completes or to use a previously trained model:
1. Click the **Load Model** button in the navbar
2. Model will be loaded from `models/final_model/` folder
3. Class labels will be automatically loaded

### 3. Evaluate Model

1. Open the **Evaluate** page
2. Select evaluation dataset folder (same structure as training dataset)
3. Click **Start Evaluation**
4. View evaluation results:
   - Confusion Matrix
   - Precision, Recall, F1-Score per class
   - Overall Accuracy

### 4. Classify Image

1. Ensure model is loaded
2. Open the **Home** page
3. Upload the image you want to classify
4. Click **Classify**
5. View results:
   - Predicted class
   - Confidence score
   - ELA visualization
   - Probabilities for all classes

## 🔧 API Endpoints

### Training
- `POST /train` - Start training model
- `GET /api/training_progress` - Get training progress

### Evaluation
- `POST /evaluate` - Start evaluation
- `GET /api/evaluation_progress` - Get evaluation progress

### Inference
- `POST /classify` - Classify single image
- `GET /load_model` - Load trained model
- `GET /unload_model` - Unload model from memory

## 🔒 Security

- Max upload size: 2GB
- File validation for image formats
- Automatic temporary file cleanup
- Session-based processing

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions or issues, please open an issue in this repository.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- Error Level Analysis for digital forensics

---

## 👥 Authors & Contributors

This project was developed as an undergraduate thesis by:

| Name |GitHub |
|------|--------|
| Muhammad Reza Mahendra Laiya |[@Kyovens](https://github.com/Kyovens) |
| Bayu Arma Praja |[@Yuu18id](https://github.com/Yuu18id) |
| Yusra Budiman Hasibuan |[@yusrabudiman](https://github.com/yusrabudiman) |


**Disclaimer:** This is an academic project developed as part of an undergraduate thesis requirement. The software is provided "as-is" without warranty of any kind. The authors and Universitas Mikroskil are not liable for any damages or issues arising from the use of this software.








