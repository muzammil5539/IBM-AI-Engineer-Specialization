# Machine Learning with Python

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📖 Course Overview

This course provides a comprehensive introduction to machine learning using Python. You'll learn fundamental ML concepts and techniques, from regression and classification to clustering and model evaluation. Each module includes hands-on labs with real-world datasets to reinforce your learning.

---

## 📚 Table of Contents

- [Module 2: Regression Models](#module-2-regression-models)
- [Module 3: Classification & Ensemble Methods](#module-3-classification--ensemble-methods)
- [Module 4: Clustering & Dimensionality Reduction](#module-4-clustering--dimensionality-reduction)
- [Module 5: Model Evaluation & Regularization](#module-5-model-evaluation--regularization)
- [Module 6: Final Projects](#module-6-final-projects)
- [Key Takeaways](#key-takeaways)

---

## Module 2: Regression Models

### 📊 Labs

### 📊 Labs

1. **Simple Linear Regression**  
   📂 [Simple-Linear-Regression.ipynb](module%202/Simple-Linear-Regression.ipynb)

2. **Multiple Linear Regression**  
   📂 [Mulitple-Linear-Regression.ipynb](module%202/Mulitple-Linear-Regression.ipynb)

3. **Logistic Regression**  
   📂 [Logistic_Regression.ipynb](module%202/Logistic_Regression.ipynb)

### 🎯 Key Concepts

- **Regression** models relationships between a continuous target variable and explanatory features
- **Simple Regression** uses a single independent variable to estimate a dependent variable
- **Multiple Regression** involves more than one independent variable for predictions
- **Applications**: Sales forecasting, cost estimation, rainfall prediction, disease spread modeling

### 📝 Technical Highlights

- **Ordinary Least Squares (OLS)**: Minimizes errors measured by Mean Squared Error (MSE)
- **Model Considerations**: OLS is easy to interpret but sensitive to outliers
- **Overfitting Risk**: Adding too many variables can lead to overfitting
- **Nonlinear Regression**: Models complex relationships using polynomial, exponential, or logarithmic functions
- **Logistic Regression**: Binary classifier using probability prediction with log-loss optimization
- **Optimization**: Gradient descent and stochastic gradient descent for efficient model training

---

## Module 3: Classification & Ensemble Methods

### 📊 Labs

1. **Multiclass Classification**  
   📂 [Multi-class_Classification.ipynb](module%203/Multi-class_Classification.ipynb)

2. **Decision Trees**  
   📂 [Decision_trees.ipynb](module%203/Decision_trees.ipynb)

3. **Regression Trees**  
   📂 [Regression_Trees_Taxi_Tip.ipynb](module%203/Regression_Trees_Taxi_Tip.ipynb)

4. **Decision Trees and SVM**  
   📂 [decision_tree_svm_ccFraud.ipynb](module%203/decision_tree_svm_ccFraud.ipynb)

5. **K-Nearest Neighbors (k-NN)**  
   📂 [KNN_Classification.ipynb](module%203/KNN_Classification.ipynb)

6. **Ensemble Learning (Random Forest + XGBoost)**  
   📂 [Random_ Forests _XGBoost.ipynb](module%203/Random_%20Forests%20_XGBoost.ipynb)

### 🎯 Key Concepts

- **Classification**: Supervised ML method for predicting labels on new data
- **Applications**: Churn prediction, customer segmentation, loan default prediction, drug prescription
- **Multiclass Strategies**: One-versus-all and one-versus-one approaches

### 📝 Technical Highlights

- **Decision Trees**: Classify data by testing features at each node, branching based on results
- **Split Quality Metrics**: Information gain and Gini impurity
- **Regression Trees**: Predict continuous values using MSE to measure split quality
- **K-Nearest Neighbors (k-NN)**: Assigns labels based on closest labeled data points
- **Support Vector Machines (SVM)**: Build classifiers by finding optimal hyperplane with maximum margin
- **Bias-Variance Tradeoff**: Managed through bagging, boosting, and random forests
- **Random Forests**: Use bagging with multiple decision trees on bootstrapped data to reduce variance

---

## Module 4: Clustering & Dimensionality Reduction

### 📊 Labs

1. **K-Means Customer Segmentation**  
   📂 [K-Means-Customer-Seg.ipynb](module%204/K-Means-Customer-Seg.ipynb)

2. **Comparing DBSCAN and HDBSCAN**  
   📂 [Comparing_DBScan_HDBScan.ipynb](module%204/Comparing_DBScan_HDBScan.ipynb)

3. **Principal Component Analysis (PCA)**  
   📂 [PCA.ipynb](module%204/PCA.ipynb)

4. **t-SNE and UMAP**  
   📂 [tSNE_UMAP.ipynb](module%204/tSNE_UMAP.ipynb)

### 🎯 Key Concepts

- **Clustering**: Unsupervised ML technique for grouping similar data
- **Applications**: Customer segmentation, anomaly detection, pattern discovery

### 📝 Technical Highlights

- **K-Means**: Partitions data into clusters based on distance between data points and centroids
- **Evaluation Methods**: Silhouette analysis, elbow method, Davies-Bouldin Index
- **DBSCAN**: Density-based algorithm that creates clusters based on density, handles irregular patterns
- **HDBSCAN**: Parameter-free variant using cluster stability
- **Hierarchical Clustering**: Divisive (top-down) or agglomerative (bottom-up) with dendrogram visualization
- **PCA**: Linear dimensionality reduction minimizing information loss while reducing noise
- **t-SNE & UMAP**: Map high-dimensional data to lower dimensions for visualization and analysis

---

## Module 5: Model Evaluation & Regularization

### 📊 Labs

1. **Evaluating Classification Models**  
   📂 [Evaluating_Classification_Models_v1.ipynb](module%205/Evaluating_Classification_Models_v1.ipynb)

2. **Evaluating Random Forest**  
   📂 [Evaluating_random_forest_v1.ipynb](module%205/Evaluating_random_forest_v1.ipynb)

3. **Evaluating K-Means Clustering**  
   📂 [Evaluating_k_means_clustering_v1.ipynb](module%205/Evaluating_k_means_clustering_v1.ipynb)

4. **Regularization in Linear Regression**  
   📂 [Regularization_in_LinearRegression_v1.ipynb](module%205/Regularization_in_LinearRegression_v1.ipynb)

5. **ML Pipelines and GridSearchCV**  
   📂 [ML_Pipelines_and_GridSearchCV.ipynb](module%205/ML_Pipelines_and_GridSearchCV.ipynb)

### 🎯 Key Concepts

- **Model Evaluation**: Assessing model performance on unseen data using train/test splits

### 📝 Technical Highlights

**Classification Metrics:**
- Accuracy, Confusion Matrix, Precision, Recall, F1 Score

**Regression Metrics:**
- MAE, MSE, RMSE, R-squared, Explained Variance

**Unsupervised Learning Metrics:**
- Silhouette Score, Davies-Bouldin Index, Adjusted Rand Index

**Dimensionality Reduction Metrics:**
- Explained Variance Ratio, Reconstruction Error, Neighborhood Preservation

**Model Validation:**
- Training, validation, and test set division
- K-fold and stratified cross-validation
- Regularization techniques: Ridge (L2) and Lasso (L1) regression

**Best Practices:**
- Prevent data leakage through proper data separation
- Address class imbalance issues
- Consider feature importance with caution
- Avoid over-reliance on automated processes

---

## Module 6: Final Projects

### 📊 Projects

1. **Practice Project**  
   📂 [Practice_Project_v1.ipynb](module%206/Practice_Project_v1.ipynb)

2. **Final Project: Australian Weather Prediction**  
   📂 [FinalProject_AUSWeather.ipynb](module%206/FinalProject_AUSWeather.ipynb)

### 🎯 Objectives

Apply all concepts learned throughout the course to real-world problems:
- Data preprocessing and exploration
- Feature engineering
- Model selection and training
- Hyperparameter tuning
- Model evaluation and comparison
- Results interpretation and presentation

---

## 🎓 Key Takeaways

### Regression
- Master simple and multiple linear regression techniques
- Understand nonlinear regression and polynomial fitting
- Apply logistic regression for binary classification
- Optimize models using gradient descent

### Classification
- Implement various classification algorithms (Decision Trees, k-NN, SVM)
- Build ensemble models (Random Forests, XGBoost)
- Handle multiclass classification problems
- Balance bias-variance tradeoffs

### Clustering
- Apply K-Means for customer segmentation
- Use density-based clustering (DBSCAN, HDBSCAN)
- Implement hierarchical clustering
- Choose appropriate clustering algorithms for different data patterns

### Dimensionality Reduction
- Reduce feature dimensions using PCA
- Visualize high-dimensional data with t-SNE and UMAP
- Improve model performance through dimension reduction
- Understand the tradeoffs between interpretability and dimensionality

### Model Evaluation
- Select appropriate metrics for different problem types
- Implement cross-validation techniques
- Apply regularization to prevent overfitting
- Build production-ready ML pipelines
- Tune hyperparameters systematically

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running the Labs
```bash
cd "1 - Machine Learning with Python"
jupyter notebook
```

---

## 📈 Next Steps

After completing this course, proceed to:
- **[Introduction to Deep Learning & Neural Networks with Keras](../2%20-%20Introduction%20to%20Deep%20Learning%20%26%20Neural%20Networks%20with%20Keras/README.md)**

---

**Happy Learning!** 🎓✨
