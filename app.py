from io import BytesIO
import subprocess
from flask import Flask, request, jsonify, render_template
import os, time, threading, json, shutil
import torch
from PIL import Image, ImageChops, ImageEnhance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import ViTForImageClassification, Trainer, TrainingArguments, TrainerCallback, ViTConfig
from datasets import Dataset, Image as PILImage, ClassLabel
from huggingface_hub import hf_hub_download

app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024
app.secret_key = 'ai_image_classifier_secret'

# Constants
RESULTS_FOLDER = 'static/results'
MODEL_FOLDER = 'models'
UPLOAD_FOLDER = 'uploads'

# Create necessary directories
for folder in [RESULTS_FOLDER, MODEL_FOLDER, UPLOAD_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global variables
model = None
label_names = []
training_status = {
    'progress': 0,
    'log': [],
    'is_complete': False
}

# Add this global variable for evaluation status
evaluation_status_data = {
    'progress': 0,
    'log': [],
    'is_complete': False
}

def get_ela_image(image, quality=90):
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved_image = Image.open(buffer)
    ela_image = ImageChops.difference(image, resaved_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

@app.route('/')
def index():
    global model, label_names
    model_loaded = model is not None
    return render_template('index.html', model_loaded=model_loaded, labels=label_names)

# Callback untuk mengupdate progres training
class FixedProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.completed_epochs = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        global training_status
        self.start_time = time.time()
        self.total_epochs = int(args.num_train_epochs)
        training_status['log'].append(f"Training start - {self.total_epochs} epochs")
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        global training_status
        self.epoch_start_time = time.time()
        # Increment epoch counter saat epoch dimulai
        self.current_epoch += 1
        training_status['log'].append(f"Epoch {self.current_epoch}/{self.total_epochs} start at {time.strftime('%H:%M:%S')}")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        global training_status
        
        # Hitung waktu yang dibutuhkan untuk epoch ini
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        # Update completed epochs
        self.completed_epochs = self.current_epoch
        
        # Hitung progress
        progress = int((self.completed_epochs / self.total_epochs) * 100)
        training_status['progress'] = progress
        
        training_status['log'].append(
            f"Epoch {self.completed_epochs}/{self.total_epochs} done at {epoch_time:.1f}s at {time.strftime('%H:%M:%S')}"
        )
        
        # Estimasi waktu tersisa (hanya jika belum selesai)
        if self.completed_epochs < self.total_epochs:
            avg_time_per_epoch = (time.time() - self.start_time) / self.completed_epochs
            remaining_epochs = self.total_epochs - self.completed_epochs
            estimated_remaining = avg_time_per_epoch * remaining_epochs
            hours, remainder = divmod(estimated_remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            training_status['log'].append(
                f"ETA: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            )
        return control
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        global training_status
        if logs:
            log_messages = []
            if 'train_loss' in logs:
                log_messages.append(f"Loss: {logs['train_loss']:.4f}")
            if 'eval_loss' in logs:
                log_messages.append(f"Val Loss: {logs['eval_loss']:.4f}")
            if 'eval_accuracy' in logs:
                log_messages.append(f"Val Acc: {logs['eval_accuracy']:.3f}")
                
            if log_messages:
                training_status['log'].append(" | ".join(log_messages))

def train_model(dataset_path, learning_rate=1e-6, epochs=10, batch_size=32, train_split=0.2):
    global model, label_names, training_status

    training_status['progress'] = 0
    training_status['log'] = ["Training started at " + time.strftime('%H:%M:%S')]
    training_status['is_complete'] = False

    try:
        # Kumpulkan path gambar dan label dari struktur folder
        file_names = []
        labels = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    label = os.path.basename(root)
                    file_path = os.path.join(root, file)
                    file_names.append(file_path)
                    labels.append(label)
        
        if not file_names:
            training_status['log'].append("Error: No valid images were found in the dataset.")
            training_status['is_complete'] = True
            return

        training_status['log'].append(f"Ditemukan {len(file_names)} image from dataset")

        # Step 2: Buat DataFrame dan Dataset HuggingFace
        df = pd.DataFrame({"image": file_names, "labels": labels})
        dataset = Dataset.from_pandas(df).cast_column("image", PILImage())
        
        unique_labels = list(df["labels"].unique())
        label_names = unique_labels
        dataset = dataset.cast_column("labels", ClassLabel(names=unique_labels))
        
        training_status['log'].append(f"Labels found: {', '.join(unique_labels)}")

        dataset = dataset.train_test_split(test_size=train_split, shuffle=True, stratify_by_column="labels")
        train_data = dataset["train"]
        test_data = dataset["test"]

        training_status['log'].append(f"Train set: {len(train_data)} image, Test set: {len(test_data)} image")

        # Step 3: Preprocessing dengan ELA
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        size = 224
        _train_transforms = Compose([Resize((size, size)), ToTensor(), normalize])
        _test_transforms = Compose([Resize((size, size)), ToTensor(), normalize])

        def get_ela_image(image, quality=90):
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            resaved_image = Image.open(buffer)
            ela_image = ImageChops.difference(image, resaved_image)
            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            scale = 255.0 / max_diff if max_diff != 0 else 1
            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
            return ela_image

        def train_transforms_with_ela(examples):
            examples["pixel_values"] = [
                _train_transforms(get_ela_image(image.convert("RGB")))
                for image in examples["image"]
            ]
            del examples["image"]
            return examples

        def test_transforms_with_ela(examples):
            examples["pixel_values"] = [
                _test_transforms(get_ela_image(image.convert("RGB")))
                for image in examples["image"]
            ]
            del examples["image"]
            return examples

        train_data.set_transform(train_transforms_with_ela)
        test_data.set_transform(test_transforms_with_ela)

        # Step 4: Konfigurasi dan Model
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1536,
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.3,
            num_labels=len(unique_labels)
        )
        model = ViTForImageClassification(config)

        # Step 5: TrainingArguments dan Trainer
        args = TrainingArguments(
            output_dir=MODEL_FOLDER,
            logging_dir='./logs',
            report_to="none",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            fp16=True,
            num_train_epochs=epochs,
            weight_decay=0.01,
            remove_unused_columns=False,
            save_strategy="epoch",
            logging_strategy="epoch",
            max_steps=-1,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_data,
            callbacks= [FixedProgressCallback()]
        )

        training_status['log'].append("Starting the training process...")
        trainer.train()

        training_status['log'].append("Training complete, saving model...")
        model_save_path = os.path.join(MODEL_FOLDER, 'final_model')
        trainer.save_model(model_save_path)

        training_status['progress'] = 100
        training_status['log'].append("The training ended at " + time.strftime('%H:%M:%S'))
        training_status['is_complete'] = True

        # Cleanup
        try:
            cleanup_path = os.path.join(UPLOAD_FOLDER, "extracted")
            if os.path.exists(cleanup_path):
                shutil.rmtree(cleanup_path)
        except Exception as cleanup_error:
            training_status['log'].append(f"Warning: Failed to delete temporary files: {str(cleanup_error)}")

    except Exception as e:
        training_status['log'].append(f"Error during the training process: {str(e)}")
        training_status['log'].append("Training failed - please check the log above")
        training_status['is_complete'] = True

@app.route('/api/training_progress')
def training_progress():
    return jsonify(training_status)

@app.route('/train', methods=['GET', 'POST'])
def train_route():
    if request.method == 'GET':
        return render_template('train.html')
    
    file_list = request.files.getlist('dataset')
    if not file_list or len(file_list) == 0:
        return jsonify({'status': 'error', 'message': 'No files selected'})
    
    timestamp = str(int(time.time()))
    extract_path = os.path.join(UPLOAD_FOLDER, 'extracted', timestamp)
    os.makedirs(extract_path, exist_ok=True)
    
    for file in file_list:
        if file.filename:
            # Preserve directory structure from webkitdirectory
            filename = file.filename
            destination = os.path.join(extract_path, filename)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            file.save(destination)
    
    learning_rate = float(request.form.get('learning_rate', 1e-6))
    epochs = int(request.form.get('epochs', 10))
    batch_size = int(request.form.get('batch_size', 32))
    train_split = float(request.form.get('train_split', 0.2))
    
    def background_training():
        try:
            train_model(extract_path, learning_rate, epochs, batch_size, train_split)
        except Exception as e:
            training_status['log'].append("Error during training: " + str(e))
            training_status['is_complete'] = True
    
    threading.Thread(target=background_training, daemon=True).start()
    return jsonify({'status': 'success', 'message': 'Training started'})

@app.route('/api/evaluation_progress')
def evaluation_progress():
    """API endpoint for getting evaluation progress"""
    global evaluation_status_data
    return jsonify(evaluation_status_data)

@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate_view():
    global model, label_names
    
    if request.method == 'GET':
        model_loaded = model is not None
        has_previous_results = os.path.exists(os.path.join(RESULTS_FOLDER, 'evaluation_results.json'))
        return render_template('evaluate.html', 
                              model_loaded=model_loaded,
                              has_previous_results=has_previous_results)
    
    # Handle POST request
    if model is None:
        return jsonify({'status': 'error', 'message': 'No model loaded. Please load or train a model first.'})
    
    eval_files = request.files.getlist('eval_dataset')
    if not eval_files or len(eval_files) == 0:
        return jsonify({'status': 'error', 'message': 'No evaluation files selected'})
        
    # Create temporary directory for evaluation data
    timestamp = str(int(time.time()))
    eval_path = os.path.join(UPLOAD_FOLDER, 'evaluation', timestamp)
    os.makedirs(eval_path, exist_ok=True)
    
    # Save all files preserving folder structure
    for file in eval_files:
        if file.filename:
            filename = file.filename
            file_path = os.path.join(eval_path, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file.save(file_path)
    
    # Start evaluation in background thread
    thread = threading.Thread(target=evaluate_model, args=(eval_path,), daemon=True)
    thread.start()
    
    return jsonify({'status': 'success', 'message': 'Evaluation started'})

def evaluate_model(eval_folder_path):
    """Evaluate a trained model on a folder of images"""
    global model, label_names, evaluation_status_data
    
    evaluation_status_data['progress'] = 0
    evaluation_status_data['log'] = ["The evaluation began on " + time.strftime('%H:%M:%S')]
    evaluation_status_data['is_complete'] = False
    
    try:
        if model is None:
            evaluation_status_data['log'].append("Error: Model not loaded")
            evaluation_status_data['is_complete'] = True
            return
            
        # Collect image paths and expected labels from folder structure
        file_paths = []
        true_labels = []
        
        evaluation_status_data['log'].append(f"Checking the folder: {eval_folder_path}")
        
        # Walk through directory structure
        for root, dirs, files in os.walk(eval_folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Get class name from parent folder
                    label = os.path.basename(root)
                    
                    # Skip if this is the root upload directory (no class structure)
                    if label == os.path.basename(eval_folder_path):
                        continue
                        
                    # Only use if label is known to the model
                    if label in label_names:
                        file_path = os.path.join(root, file)
                        file_paths.append(file_path)
                        true_labels.append(label)
                        evaluation_status_data['log'].append(f"Image found: {file} label: {label}")
        
        if not file_paths:
            evaluation_status_data['log'].append("Error: No valid images were found for evaluation.")
            evaluation_status_data['log'].append(f"Expected folder structure with subfolders named: {', '.join(label_names)}")
            evaluation_status_data['is_complete'] = True
            return
            
        evaluation_status_data['log'].append(f"Found {len(file_paths)} image for evaluation")
        
        # Convert string labels to indices
        y_true = [label_names.index(label) for label in true_labels]
        y_pred = []
        y_scores = []
        
        evaluation_status_data['progress'] = 10
        
        # Get device
        device = next(model.parameters()).device
        
        # Create transformation pipeline
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        size = 224
        transform = Compose([Resize((size, size)), ToTensor(), normalize])
        
        total_images = len(file_paths)
        for i, file_path in enumerate(file_paths):
            try:
                # Load and preprocess image
                image = Image.open(file_path).convert('RGB')
                
                # Apply ELA transformation
                ela_image = get_ela_image(image)
                
                # Apply transformations
                pixel_values = transform(ela_image).unsqueeze(0)
                pixel_values = pixel_values.to(device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(pixel_values=pixel_values)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    y_scores.append(probs[0][1].item())
                    pred_class = torch.argmax(probs, dim=-1).item()
                
                y_pred.append(pred_class)
                
                # Update progress
                if i % 10 == 0 or i == total_images - 1:
                    progress = 10 + int(80 * (i + 1) / total_images)
                    evaluation_status_data['progress'] = progress
                    evaluation_status_data['log'].append(f"Processed {i+1}/{total_images} images")
            
            except Exception as e:
                evaluation_status_data['log'].append(f"Error processing {file_path}: {str(e)}")
                # Add a dummy prediction to maintain array length
                y_pred.append(0)

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_names, yticklabels=label_names)
        plt.xlabel('Prediksi')
        plt.ylabel('Benar')
        plt.title('Confusion Matrix - Evaluation')
        plt.tight_layout()
        cm_save_path = os.path.join(RESULTS_FOLDER, 'confusion_matrix.png')
        plt.savefig(cm_save_path)
        plt.close()  # Close the plot to free memory

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Evaluation')
        plt.legend(loc='lower right')
        roc_save_path = os.path.join(RESULTS_FOLDER, 'ROC.png')
        plt.savefig(roc_save_path)
        plt.close()
        
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
        
        # Save results
        results = {
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'label_names': label_names,
            'num_eval_images': total_images
        }
        
        with open(os.path.join(RESULTS_FOLDER, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f)
        
        evaluation_status_data['progress'] = 100
        evaluation_status_data['log'].append(f"Evaluation completed")
        evaluation_status_data['log'].append(f"Final accuracy: {report['accuracy']:.3f}")
        evaluation_status_data['is_complete'] = True
        
        # Clean up temporary files
        try:
            cleanup_path = os.path.join(eval_folder_path)
            evaluation_status_data['log'].append(f"Path: {str(cleanup_path)}")
            if os.path.exists(cleanup_path):
                shutil.rmtree(cleanup_path)
        except Exception as cleanup_error:
            evaluation_status_data['log'].append(f"Warning: File temp failed to delete: {str(cleanup_error)}")
        
    except Exception as e:
        evaluation_status_data['log'].append(f"Error evaluation: {str(e)}")
        evaluation_status_data['is_complete'] = True

@app.route('/load_model')
def load_model_route():
    global model, label_names
    try:
        model_dir = os.path.join(MODEL_FOLDER, "final_model")
        os.makedirs(model_dir, exist_ok=True)

        base_url = "https://huggingface.co/yuu18id/VIT-ELA-Image-Classifier/resolve/main"
        files = ["config.json", "model.safetensors", "training_args.bin"]

        for f in files:
            dst = os.path.join(model_dir, f)
            if not os.path.exists(dst):
                url = f"{base_url}/{f}"
                subprocess.run(["curl", "-L", url, "-o", dst], check=True)

        model = ViTForImageClassification.from_pretrained(model_dir)

        if not label_names:
            results_path = os.path.join(RESULTS_FOLDER, 'evaluation_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    label_names = results.get('label_names', [])

        return jsonify({
            "status": "success",
            "message": "Model loaded successfully",
            "labels": label_names
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error loading model: {str(e)}"
        })
    
@app.route('/unload_model')
def unload_model_route():
    """Route to unload the currently loaded model"""
    global model, label_names
    try:
        # Clear the model from memory
        if model is not None:
            # Move model to CPU and delete to free GPU memory
            model.cpu()
            del model
            model = None
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Clear label names
        label_names = []
        
        return jsonify({
            'status': 'success',
            'message': 'Model unloaded successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error unloading model: {str(e)}'
        })

@app.route('/classify', methods=['POST'])
def classify():
    global model, label_names
    
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded. Please load a model first.'})
    
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image file provided'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    try:
        image = Image.open(file.stream).convert('RGB')
        ela_image = get_ela_image(image)
        
        # Save ELA image for display
        ela_path = os.path.join(RESULTS_FOLDER, 'ela_image.jpg')
        ela_image.save(ela_path)
        
        # Prepare transformation
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        size = 224
        transform = Compose([Resize((size, size)), ToTensor(), normalize])
        pixel_values = transform(ela_image).unsqueeze(0)
        
        # Get device and move tensors
        device = next(model.parameters()).device
        pixel_values = pixel_values.to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
        
        probs_list = probs[0].cpu().tolist()
        predictions = [{'label': label, 'probability': prob} for label, prob in zip(label_names, probs_list)]
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'prediction': label_names[pred_class],
            'confidence': probs[0][pred_class].cpu().item(),
            'all_predictions': predictions,
            'ela_image': 'static/results/ela_image.jpg'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error processing image: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)