# 🧠 Brain Tumor Detection Using GenAI

An AI-powered brain MRI image classification application built using **Python, TensorFlow, MobileNetV2, Streamlit, and Google Gemini**.

The application analyzes a brain MRI image using a deep-learning classification model and provides:

- 🧠 MRI image classification
- 📊 Prediction confidence
- 📈 Class probability distribution
- 🤖 AI-generated educational explanation

---

## 🌐 Live Demo

🚀 **Live Application:** Coming Soon

📂 **GitHub Repository:**  
https://github.com/arzootarana0828-beep/brain_tumor_detector

---

## 📌 Overview

This project demonstrates how **Deep Learning and Generative AI** can be integrated into an interactive web application.

The image classification model uses **MobileNetV2 transfer learning** to classify MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary

After the TensorFlow model generates a prediction, Google Gemini is used to provide an educational explanation of the result.

### Application Workflow

```text
MRI Image
     ↓
Streamlit Web Interface
     ↓
Image Preprocessing
     ↓
MobileNetV2 Deep Learning Model
     ↓
Prediction + Confidence
     ↓
Class Probabilities
     ↓
Google Gemini
     ↓
Educational Explanation