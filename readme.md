# Machine Learning Assignment 2 – Classification Models & Deployment

## 1. Problem Statement
The goal of this assignment is to design, assess, and deploy several machine learning classification models using a real-world dataset. It showcases the complete machine learning pipeline, covering data preparation, model development, evaluation through widely used performance metrics, and the deployment of an interactive user interface using Streamlit.

---

## 2. Dataset Description
- **Dataset Name:** Heart Disease Dataset  
- **Source:** UCI Machine Learning Repository  
- **Problem Type:** Binary Classification  
- **Total Records:** ~1025  
- **Number of Features:** 13 (after encoding more features are generated)  
- **Target Variable:** `target`  
  - `0` → No Heart Disease  
  - `1` → Presence of Heart Disease  

The dataset contains clinical and demographic attributes such as age, sex, cholesterol,
resting blood pressure, maximum heart rate, chest pain type, and other medically relevant
features. Categorical variables were encoded using one-hot encoding before model training.

---

## 3. Models Implemented
The following six classification models were implemented using the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

---

## 4. Evaluation Metrics
Each model was evaluated using the following metrics:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The dataset was split into **80% training data and 20% testing data** using stratified
sampling to preserve class distribution.

---

## 5. Model Performance Comparison

| ML Model Name        | Accuracy | AUC    | Precision | Recall | F1 Score | MCC   |
|---------------------|----------|--------|-----------|--------|----------|-------|
| Logistic Regression | 0.8732   | 0.9454 | 0.8559    | 0.9048 | 0.8796   | 0.7471 |
| Decision Tree       | 0.9854   | 0.9857 | 1.0000    | 0.9714 | 0.9855   | 0.9712 |
| KNN                 | 0.8537   | 0.9603 | 0.8788    | 0.8286 | 0.8529   | 0.7088 |
| Naive Bayes         | 0.8049   | 0.8858 | 0.7982    | 0.8286 | 0.8131   | 0.6096 |
| Random Forest       | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000   | 1.0000 |
| XGBoost             | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000   | 1.0000 |

---

## 6. Observations on Model Performance

| ML Model Name        | Observation                                                                            |
|----------------------|----------------------------------------------------------------------------------------|
| Logistic Regression  | Provides a strong baseline performance and works well for linearly separable patterns. |
| Decision Tree        | Achieves very high accuracy but may be prone to overfitting due to its flexibility.    |
| KNN                  | Performance depends on distance calculations and is sensitive to feature scaling.      |
| Naive Bayes          | Computationally efficient but limited by the assumption of feature independence.       |
| Random Forest        | Achieves perfect classification by effectively capturing complex feature interactions. |
| XGBoost              | Demonstrates excellent performance due to boosting and regularization techniques.      |

---

## 7. Streamlit Web Application
An interactive Streamlit web application was developed and deployed using Streamlit
Community Cloud. The application includes the following features:

- CSV file upload option for test data  
- Model selection dropdown  
- Display of evaluation metrics  
- Visualization of confusion matrix  

The application is robust to different test splits of the dataset and does not rely on any
hard-coded local file paths.

---

## 8. Repository Structure
mlassignment2/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│ ├── utils.py
│ ├── logistic_model.py
│ ├── decision_tree_model.py
│ ├── knn_model.py
│ ├── naive_bayes_model.py
│ ├── random_forest_model.py
│ └── xgboost_model.py

---

## 9. Deployment
The Streamlit application was deployed using **Streamlit Community Cloud** by linking
the GitHub repository and selecting `app.py` as the entry point.

---

## 10. Tools & Technologies Used
- Python  
- scikit-learn  
- XGBoost  
- Pandas & NumPy  
- Matplotlib & Seaborn  
- Streamlit  

---

## 11. Execution Environment

