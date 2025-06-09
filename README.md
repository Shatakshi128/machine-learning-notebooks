# Vision-Based Object Classification – Fruits360

## 🧠 Overview
This project is part of the Jayadhi AI/ML Developer – Entry-Level Screening Task (Option 5).  
It focuses on classifying fruit images using a lightweight MobileNetV2 model.

## 📊 Objective
To build a vision-based object classifier using transfer learning (MobileNetV2).

## 🗂️ Dataset
- **Name:** Fruits-360 Dataset
- **Structure:** Training and Test folders with subdirectories per class
- **Image size:** Resized to 100x100

## ⚙️ Model Details
- **Base Model:** MobileNetV2 (frozen)
- **Architecture:** Custom classifier with dropout
- **Optimizer:** Adam, learning rate decay and early stopping used
- **Output:** Softmax classification

## 📈 Evaluation
- Accuracy and loss graphs
- Final evaluation on test set
- Confusion matrix
- Classification report

## 💻 Dependencies
```bash
tensorflow
numpy
matplotlib
seaborn
scikit-learn
```

## ✅ How to Run
Run the `.ipynb` notebook in Jupyter or Google Colab.

## 📌 Notes
- Dataset path must point to `./fruits-360/Training` and `./fruits-360/Test`
- Training and evaluation work only if dataset is available

---
_This project is submitted as part of the AI/ML Developer Entry Screening - Option 5_
