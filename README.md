# 🏋️ Body Performance Fitness Classification (Machine Learning Project)

🔗 **Live Demo**: https://body-performance-ml.streamlit.app/

An end-to-end **Data Science & Machine Learning project** that predicts a person’s **fitness performance class (A, B, C, D)** using body measurements and physical performance metrics.  
The project covers **data preprocessing, feature engineering, model training, evaluation, and deployment** using **Streamlit**.

---

## 📌 Project Overview

This application allows users to input their body and performance details and instantly receive a predicted **fitness class** based on a trained **XGBoost classification model**.

The project demonstrates a complete ML workflow from raw data to a deployed web app.

---

## 🎯 Fitness Classes

| Class | Description |
|------|------------|
| A | Excellent physical performance |
| B | Good physical performance |
| C | Average physical performance |
| D | Low physical performance |

---

## 🧠 Machine Learning Workflow

### 🔹 Data Preprocessing
- Removed duplicate records
- Handled outliers using **IQR-based capping**
- Encoded categorical features (gender)
- Applied **MinMaxScaler** for feature scaling

### 🔹 Feature Engineering
- Calculated **Body Mass Index (BMI)**  
  \[
  BMI = \frac{weight (kg)}{height (m)^2}
  \]

### 🔹 Model Training
Trained and evaluated multiple models:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- **XGBoost (Final Model)** ✅

XGBoost achieved the best performance and was selected for deployment.

---

## 🚀 Tech Stack

- **Language**: Python
- **Libraries**:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - joblib
  - streamlit
- **Deployment**: Streamlit Community Cloud
- **Version Control**: Git & GitHub

---

## 🖥️ Web Application Features

- Interactive Streamlit UI
- Real-time fitness class prediction
- Input validation
- Scaled inference using saved preprocessing objects
- Publicly deployed live demo

---

## 📂 Project Structure

BodyPerformanceDS/
│
├── app.py # Streamlit application
├── preprocess.py # Data preprocessing & model training
├── bodyPerformance.csv # Dataset
├── xgboost_model.pkl # Trained XGBoost model
├── scaler.pkl # Saved MinMaxScaler
├── requirements.txt # Project dependencies
├── .gitignore
└── README.md


---

## ▶️ Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/UsmanAli-1/Body-Performance-ML-Project.git
cd Body-Performance-ML-Project

pip install -r requirements.txt

streamlit run app.py
