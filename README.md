# Domestic Animal Footprints Classification

A deep learning project for classifying domestic animal footprints using transfer learning with VGG16. This project achieves **99.38% test accuracy** in identifying footprints from 5 different animal classes: cat, dog, horse, rabbit, and sheep.

## 📋 Overview

This project implements a convolutional neural network (CNN) based on VGG16 architecture to classify animal footprints. The model uses transfer learning techniques with a two-stage training approach:
1. **Feature Extraction Stage**: Training only the top layers while keeping VGG16 base frozen
2. **Fine-tuning Stage**: Unfreezing and fine-tuning the top convolutional blocks of VGG16

## 🎯 Features

- **Transfer Learning**: Leverages pre-trained VGG16 weights from ImageNet
- **Data Augmentation**: Comprehensive augmentation to improve model generalization
- **Two-Stage Training**: Optimized training strategy for better performance
- **High Accuracy**: Achieves 99.38% test accuracy
- **Comprehensive Evaluation**: Includes confusion matrix and classification report

## 📊 Dataset

The dataset contains images of animal footprints organized in the following structure:

```
dataset/
├── cat/
├── dog/
├── horse/
├── rabbit/
└── sheep/
```

### Dataset Split

The notebook automatically splits the dataset into:
- **Training Set**: 80% (7,697 images)
- **Validation Set**: 10% (967 images)
- **Test Set**: 10% (962 images)

## 🛠️ Requirements

### Python Packages

```bash
tensorflow>=2.0.0
numpy
matplotlib
seaborn
scikit-learn
```

### Installation

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## 📁 Project Structure

```
.
├── Domestic_Animal_Footprints.ipynb    # Main notebook
├── dataset/                              # Original dataset directory
├── animal_dataset_split/                 # Split dataset (created by notebook)
│   ├── train/
│   ├── val/
│   └── test/
├── best_vgg16_animal.h5                  # Best model checkpoint
└── vgg16_animal_footprints_finetuned.h5 # Final trained model
```

## 🚀 Usage

### Running the Notebook

1. **Prepare your dataset**: 
   - Place your animal footprint images in the `dataset/` directory
   - Organize images into subdirectories by class (cat, dog, horse, rabbit, sheep)

2. **Open the notebook**:
   ```bash
   jupyter notebook Domestic_Animal_Footprints.ipynb
   ```

3. **Run all cells**: The notebook will:
   - Split the dataset into train/validation/test sets
   - Set up data augmentation
   - Build and compile the VGG16-based model
   - Train the model in two stages
   - Evaluate performance and generate visualizations
   - Save the final model

### Model Architecture

The model consists of:
- **Base Model**: VGG16 (pre-trained on ImageNet, frozen initially)
- **Global Average Pooling**: Reduces spatial dimensions
- **Dense Layers**: 
  - 512 units with ReLU activation and L2 regularization
  - 256 units with ReLU activation and L2 regularization
  - Output layer with softmax activation (5 classes)
- **Regularization**: Dropout layers (0.5 and 0.4) to prevent overfitting

### Training Parameters

- **Image Size**: 224 × 224 pixels
- **Batch Size**: 32
- **Stage 1 (Feature Extraction)**:
  - Learning Rate: 1e-4
  - Epochs: 25
  - Base model: Frozen
  
- **Stage 2 (Fine-tuning)**:
  - Learning Rate: 1e-5
  - Epochs: 15
  - Last 8 layers of VGG16: Unfrozen

### Data Augmentation

Training images are augmented with:
- Rotation (±25°)
- Width/Height shift (±20%)
- Shear transformation (±15°)
- Zoom (±25%)
- Horizontal flip
- Brightness adjustment (0.8-1.2x)

## 📈 Results

### Performance Metrics

- **Test Accuracy**: 99.38%
- **Test Loss**: 0.2286

### Classification Report

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Cat     | 0.99      | 0.98   | 0.99     | 192     |
| Dog     | 0.98      | 0.98   | 0.98     | 176     |
| Horse   | 1.00      | 1.00   | 1.00     | 201     |
| Rabbit  | 1.00      | 1.00   | 1.00     | 197     |
| Sheep   | 0.99      | 1.00   | 0.99     | 196     |
| **Avg** | **0.99**  | **0.99** | **0.99** | **962** |

## 📝 Notebook Workflow

1. **Imports**: Load necessary libraries
2. **Data Splitting**: Automatically split dataset into train/val/test
3. **Data Preprocessing**: Set up data generators with augmentation
4. **Model Building**: Create VGG16-based model architecture
5. **Stage 1 Training**: Train top layers (feature extraction)
6. **Stage 2 Training**: Fine-tune VGG16 layers
7. **Evaluation**: Test model performance and generate visualizations
8. **Model Saving**: Save the final trained model

## 🔧 Callbacks

The training uses three callbacks:
- **EarlyStopping**: Stops training if validation loss doesn't improve for 8 epochs
- **ReduceLROnPlateau**: Reduces learning rate by factor of 0.3 when validation loss plateaus
- **ModelCheckpoint**: Saves the best model based on validation loss

## 📊 Visualizations

The notebook generates:
- Sample training images with labels
- Training/validation accuracy and loss curves
- Confusion matrix heatmap
- Classification report

## 💾 Model Files

- `best_vgg16_animal.h5`: Best model checkpoint during training
- `vgg16_animal_footprints_finetuned.h5`: Final trained model

## 🔍 Key Techniques

1. **Transfer Learning**: Uses pre-trained VGG16 to leverage learned features
2. **Progressive Unfreezing**: Two-stage training for optimal fine-tuning
3. **Data Augmentation**: Increases dataset diversity and model robustness
4. **Regularization**: L2 regularization and dropout to prevent overfitting
5. **Learning Rate Scheduling**: Adaptive learning rate reduction

## 📚 References

- VGG16: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- Transfer Learning with Keras: [TensorFlow Documentation](https://www.tensorflow.org/guide/keras/transfer_learning)

## 🤝 Contributing

Feel free to submit issues or pull requests if you have suggestions for improvements.

## 📄 License

This project is open source and available for educational and research purposes.

---

**Note**: Make sure you have sufficient computational resources (GPU recommended) as training can take several hours depending on your hardware configuration.

