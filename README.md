# IBM-AI-Engineer-Specialization

coursera IBM Specialization

## Machine Learning with Python

### module 2

      Simple Regression

#### Lab File [ 1 - Machine Learning with Python/module 2/Simple-Linear-Regression.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%202/Simple-Linear-Regression.ipynb)

      Multiple Linear Regression

#### Lab File [1 - Machine Learning with Python/module 2/Mulitple-Linear-Regression.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%202/Mulitple-Linear-Regression.ipynb)

      Logistic Regression

#### Lab File [1 - Machine Learning with Python/module 2 Logistic Regression/Logistic_Regression.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%202%20Logistic%20Regression/Logistic_Regression.ipynb)

##### Module 2 Summary and Highlights

Regression models relationships between a continuous target variable and explanatory features, covering simple and multiple regression types.

Simple regression uses a single independent variable to estimate a dependent variable, while multiple regression involves more than one independent variable.

Regression is widely applicable, from forecasting sales and estimating maintenance costs to predicting rainfall and disease spread.

In simple linear regression, a best-fit line minimizes errors, measured by Mean Squared Error (MSE); this approach is known as Ordinary Least Squares (OLS).

OLS regression is easy to interpret but sensitive to outliers, which can impact accuracy.

Multiple linear regression extends simple linear regression by using multiple variables to predict outcomes and analyze variable relationships.

Adding too many variables can lead to overfitting, so careful variable selection is necessary to build a balanced model.

Nonlinear regression models complex relationships using polynomial, exponential, or logarithmic functions when data does not fit a straight line.

Polynomial regression can fit data but mayoverfit by capturing random noise rather than underlying patterns.

Logistic regression is a probability predictor and binary classifier, suitable for binary targets and assessing feature impact.

Logistic regression minimizes errors using log-loss and optimizes with gradient descent or stochastic gradient descent for efficiency.

Gradient descent is an iterative process to minimize the cost function, which is crucial for training logistic regression models.

### module 3

      Multiclass Classification

#### Lab File [1 - Machine Learning with Python/module 3/Multi-class_Classification.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%203/Multi-class_Classification.ipynb)

      Descision Trees

#### Lab File [1 - Machine Learning with Python/module 3/Decision_trees.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%203/Decision_trees.ipynb)

      Regression Trees

#### Lab File [1 - Machine Learning with Python/module 3/Regression_Trees_Taxi_Tip.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%203/Regression_Trees_Taxi_Tip.ipynb)

      Decision Trees and SVM

#### Lab File [1 - Machine Learning with Python/module 3/decision_tree_svm_ccFraud.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%203/decision_tree_svm_ccFraud.ipynb)

      KNN Classification

#### Lab File [1 - Machine Learning with Python/module 3/KNN_Classification.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%203/KNN_Classification.ipynb)

      Ensamble Learning (Random Forest + XGBoost)

#### Lab File [1 - Machine Learning with Python/module 3/Random\_ Forests \_XGBoost.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%203/Random_%20Forests%20_XGBoost.ipynb)

##### Module 3 Summary and Highlights

Classification is a supervised machine learning method used to predict labels on new data with applications in churn prediction, customer segmentation, loan default prediction, and multiclass drug prescriptions.

Binary classifiers can be extended to multiclass classification using one-versus-all or one-versus-one strategies.

A decision tree classifies data by testing features at each node, branching based on test results, and assigning classes at leaf nodes.

Decision tree training involves selecting features that best split the data and pruning the tree to avoid overfitting.

Information gain and Gini impurity are used to measure the quality of splits in decision trees.

Regression trees are similar to decision trees but predict continuous values by recursively splitting data to maximize information gain.

Mean Squared Error (MSE) is used to measure split quality in regression trees.

K-Nearest Neighbors (k-NN) is a supervised algorithm used for classification and regression by assigning labels based on the closest labeled data points.

To optimize k-NN, test various k values and measure accuracy, considering class distribution and feature relevance.

Support Vector Machines (SVM) build classifiers by finding a hyperplane that maximizes the margin between two classes, effective in high-dimensional spaces but sensitive to noise and large datasets.

The bias-variance tradeoff affects model accuracy, and methods such as bagging, boosting, and random forests help manage bias and variance to improve model performance.

Random forests use bagging to train multiple decision trees on bootstrapped data, improving accuracy by reducing variance.

### module 4

      K-Means-Customer-Seg

#### Lab File [1 - Machine Learning with Python/module 4/K-Means-Customer-Seg.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%204/K-Means-Customer-Seg.ipynb)

      Comparing_DBScan_HDBScan

#### Lab File [1 - Machine Learning with Python/module 4/Comparing_DBScan_HDBScan.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%204/Comparing_DBScan_HDBScan.ipynb)

      PCA

#### Lab File [1 - Machine Learning with Python/module 4/PCA.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%204/PCA.ipynb)

      tSNE_UMAP

#### Lab File [1 - Machine Learning with Python/module 4/tSNE_UMAP.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%204/tSNE_UMAP.ipynb)

##### Module 4 Summary and Highlights

Clustering is a machine learning technique used to group data based on similarity, with applications in customer segmentation and anomaly detection.

K-means clustering partitions data into clusters based on the distance between data points and centroids but struggles with imbalanced or non-convex clusters.

Heuristic methods such as silhouette analysis, the elbow method, and the Davies-Bouldin Index help assess k-means performance.

DBSCAN is a density-based algorithm that creates clusters based on density and works well with natural, irregular patterns.

HDBSCAN is a variant of DBSCAN that does not require parameters and uses cluster stability to find clusters.

Hierarchical clustering can be divisive (top-down) or agglomerative (bottom-up) and produces a dendrogram to visualize the cluster hierarchy.

Dimension reduction simplifies data structure, improves clustering outcomes, and is useful in tasks such as face recognition (using eigenfaces).

Clustering and dimension reduction work together to improve model performance by reducing noise and simplifying feature selection.

PCA, a linear dimensionality reduction method, minimizes information loss while reducing dimensionality and noise in data.

t-SNE and UMAP are other dimensionality reduction techniques that map high-dimensional data into lower-dimensional spaces for visualization and analysis.

### module 5

      Evaluating_Classification_Models_v1

#### Lab File [1 - Machine Learning with Python/module 5/Evaluating_Classification_Models_v1.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%205/Evaluating_Classification_Models_v1.ipynb)

      Evaluating_random_forest_v1

#### Lab File [1 - Machine Learning with Python/module 5/Evaluating_random_forest_v1.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%205/Evaluating_random_forest_v1.ipynb)

      Evaluating_k_means_clustering_v1

#### Lab File [1 - Machine Learning with Python/module 5/Evaluating_k_means_clustering_v1.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%205/Evaluating_k_means_clustering_v1.ipynb)

      Regularization_in_LinearRegression_v1

#### Lab File [1 - Machine Learning with Python/module 5/Regularization_in_LinearRegression_v1.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%205/Regularization_in_LinearRegression_v1.ipynb)

      ML_Pipelines_and_GridSearchCV

#### Lab File [1 - Machine Learning with Python/module 5/ML_Pipelines_and_GridSearchCV.ipynb](https://github.com/muzammil5539/IBM-AI-Engineer-Specialization/blob/main/1%20-%20Machine%20Learning%20with%20Python/module%205/ML_Pipelines_and_GridSearchCV.ipynb)

##### Module 4 Summary and Highlights

Supervised learning evaluation assesses a model's ability to predict outcomes for unseen data, often using a train/test split to estimate performance.

Key metrics for classification evaluation include accuracy, confusion matrix, precision, recall, and the F1 score, which balances precision and recall.

Regression model evaluation metrics include MAE, MSE, RMSE, R-squared, and explained variance to measure prediction accuracy.

Unsupervised learning models are evaluated for pattern quality and consistency using metrics like Silhouette Score, Davies-Bouldin Index, and Adjusted Rand Index.

Dimensionality reduction evaluation involves Explained Variance Ratio, Reconstruction Error, and Neighborhood Preservation to assess data structure retention.

Model validation, including dividing data into training, validation, and test sets, helps prevent overfitting by tuning hyperparameters carefully.

Cross-validation methods, especially K-fold and stratified cross-validation, support robust model validation without overfitting to test data.

Regularization techniques, such as ridge (L2) and lasso (L1) regression, help prevent overfitting by adding penalty terms to linear regression models.

Data leakage occurs when training data includes information unavailable in real-world data, which is preventable by separating data properly and mindful feature selection.

Common modeling pitfalls include misinterpreting feature importance, ignoring class imbalance, and relying excessively on automated processes without causal analysis.

Feature importance assessments should consider redundancy, scale sensitivity, and avoid misinterpretation, as well as inappropriate assumptions about causation.
