# Insect Classification using Fine-Tuned CNN (MobileNetV3Large)

This project benchmarks the performance of a **fine-tuned Convolutional Neural Network (CNN)** for image classification on **GPU vs CPU** using **TensorFlow**.

##  Project Summary
We used **MobileNetV3Large** pretrained on ImageNet and fine-tuned it to classify insect species from a custom dataset of 5 classes:
- **Bees**
- **Butterfly**
- **Moth**
- **Scorpion**
- **Spider**

The model is trained using data augmentation and evaluated based on accuracy and training time across GPU and CPU hardware platforms.

---

##  Project Structure

```
dataset/
    ├── train/          # Training images in 5 class folders
    └── validation/     # Validation images in 5 class folders

📄 insect_classification_colab.py       # Full training code
📄 Insect_Classification_Colab.ipynb    # Colab-ready notebook
📄 GPU_vs_CPU_Benchmark_Insect_Classification.ipynb  # Benchmark notebook
```

---

## Technologies Used

- **TensorFlow / Keras**
- **MobileNetV3Large**
- **Google Colab**
- **Kaggle API**
- **Python (ImageDataGenerator, Callbacks)**

---

## Evaluation Metrics

- Accuracy
- Training Time (CPU vs GPU)
- Validation Loss

---

## How to Run in Google Colab

1. Enable GPU runtime in Colab
2. Upload or download the dataset to `/dataset/train/` and `/dataset/validation/`
3. Open and run:
   - `Insect_Classification_Colab.ipynb` to train the model
   - `GPU_vs_CPU_Benchmark_Insect_Classification.ipynb` to compare hardware performance

---

## Results

| Hardware | Accuracy | Training Time |
|----------|----------|----------------|
| CPU      | ~87%     | ~12–18 mins     |
| GPU (Tesla T4) | ~87%     | ~25–30 seconds |

---

## Credits

Developed by Group 10 — University of South Florida  
Course: **Hardware Accelerators for Machine Learning**  
Spring 2024

---

## Dataset

Download from: [Kaggle – Insect Classification Dataset](https://www.kaggle.com/datasets/florianblume/insect-images-classification)

---

