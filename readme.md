# Machine Learning Assignment 2 – Classification Models & Deployment

## 🧠 Problem Statement
The objective of this assignment is to design, evaluate, and deploy multiple machine learning classification models on a real-world dataset.  
The project covers:
- Data preprocessing and feature engineering  
- Model training and evaluation  
- Performance comparison using standard metrics  
- Deployment of an interactive UI using Streamlit  

---

## 📊 Dataset Description

- **Dataset Name:** Heart Disease Dataset  
- **Source:** UCI Machine Learning Repository  
- **Problem Type:** Binary Classification  
- **Total Records:** 1026  
- **Number of Features:** 13 (additional features generated after encoding)  
- **Target Variable:** `target`  
  - `0` → No Heart Disease  
  - `1` → Presence of Heart Disease  

---

## 🤖 Models Implemented

The following machine learning models were implemented and evaluated:

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)  

---

## 📈 Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.883 | 0.965 | 0.884 | 0.883 | 0.883 | 0.767 |
| Decision Tree | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| KNN | 0.857 | 0.959 | 0.857 | 0.857 | 0.857 | 0.714 |
| Naive Bayes | 0.818 | 0.899 | 0.818 | 0.818 | 0.818 | 0.636 |
| Random Forest (Ensemble) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| XGBoost (Ensemble) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

---

| **ML Model Name**            | **Observation about Model Performance**                                                                                                                                                                |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Logistic Regression**      | Performed well with high accuracy and AUC, indicating good class separation. However, it is slightly less effective than ensemble models in capturing complex non-linear patterns.                     |
| **Decision Tree**            | Achieved perfect scores across all metrics. While this indicates excellent performance, it may suggest overfitting, especially on smaller or simpler datasets.                                         |
| **KNN**                      | Delivered balanced and consistent performance across all metrics. Performance is good but slightly lower than Logistic Regression and ensemble models, likely due to sensitivity to data distribution. |
| **Naive Bayes**              | Showed moderate performance with comparatively lower accuracy and AUC. This is expected due to its strong feature independence assumption.                                                             |
| **Random Forest (Ensemble)** | Achieved perfect performance, demonstrating strong generalization and robustness by combining multiple decision trees and reducing overfitting.                                                        |
| **XGBoost (Ensemble)**       | Also achieved perfect scores. Its boosting mechanism effectively corrects previous errors, making it one of the best-performing models in this project.                                                |


---

## 🚀 Deployment
The trained models were deployed using **Streamlit**, providing an interactive web interface for real-time predictions and result visualization.

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
│ ├── function.py
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

