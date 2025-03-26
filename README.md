# 🏥 ECG-Based Biometrics

## 🚀 Overview
This project implements **ECG-based biometric identification**, leveraging **signal processing and machine learning** to differentiate between individuals using their **unique heart activity patterns**.

### 🎯 Objective
- Preprocess **ECG signals** (resampling, filtering, normalization).  
- Segment the signals and extract **distinctive features** using **Discrete Cosine Transform (DCT)**.  
- Train and evaluate machine learning models (**K-Nearest Neighbors, SVM, and Decision Tree**) to classify ECG signals.  
- Provide a **user-friendly GUI** for file selection and real-time predictions.  

---

## 📌 Features
- ✅ **ECG Signal Preprocessing** (Resampling, Filtering, Normalization)  
- ✅ **Feature Extraction** (Autocorrelation & DCT)  
- ✅ **Multi-Model Classification** (KNN, SVM, Decision Tree)  
- ✅ **Graphical Visualizations** (ECG Plots at Different Stages)  
- ✅ **Interactive File Selection & Prediction Interface**  

---

## 🛠️ Technologies Used
- **Programming Language:** Python  
- **Signal Processing:** NumPy, SciPy  
- **Machine Learning:** Scikit-learn  
- **Visualization:** Matplotlib  
- **GUI:** Tkinter  

---

## 🚀 Usage
- Step 1: The program will train KNN, SVM, and Decision Tree models using ECG data.
- Step 2: It displays training accuracy for each model.
- Step 3: The user selects an ECG file for testing.
- Step 4: The program predicts which individual the ECG belongs to using KNN and SVM.
- Step 5: The processed ECG signals are visualized at different stages (original, resampled, filtered, normalized).

---
## GUI
  - ![image](https://github.com/user-attachments/assets/5803b216-7436-4304-9181-d083da164dfb)

- Show Accuracy button
  - ![image](https://github.com/user-attachments/assets/7dfb4cde-d823-4e69-b928-ae24006f5705)

- Start Prediction button
  - ![image](https://github.com/user-attachments/assets/d1ee5f86-830f-406d-ab28-a2c40ab82665)
  - ![image](https://github.com/user-attachments/assets/f6fd1af0-e23d-44f3-8c58-60ee1669dc87)

----
## Contributors
- Omar Ibrahim
- Sara Habib
- Malak Fekry
