# Required Assignment 17.1: Comparing Classifiers

## Project Overview
This project focuses on optimizing the efficiency of a Portuguese banking institution's direct telemarketing campaigns for long-term term deposit subscriptions. By analyzing customer demographics, historical campaign interactions, and macroeconomic indicators, we build and evaluate multiple predictive classification models. 

The ultimate goal is to precisely target high-probability prospects, thereby minimizing operational costs, reducing wasted phone calls, and maximizing resource allocation for sales agents.

---

## Repository Structure
* **`prompt_III.ipynb`**: The primary Jupyter Notebook containing data ingestion, preprocessing, baseline evaluation, model training, and advanced hyperparameter tuning via GridSearchCV.
* **`data/`**: Directory containing the core dataset asset (`bank-additional-full.csv`).
* **`README.md`**: A high-level executive summary of project findings, organization, and optimized machine learning benchmarks.

---

## Business Understanding & Objective
Direct marketing via telephone incurs substantial human and financial costs. Traditionally, cold-calling strategies yield low conversion rates. 

**Core Objective:** To leverage and compare machine learning classifiers—specifically **Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Support Vector Machines (SVM)**—to accurately predict whether a client will subscribe to a term deposit (target variable `y`). The optimal model must balance high predictive accuracy with manageable computational overhead (training time) to guide future marketing deployments.

---

## Data Analysis & Preprocessing
The dataset tracks 21 features across thousands of historical client interactions. Key preprocessing steps implemented in the pipeline include:

* **Handling Missing Values:** Explicit string indicators labeled as `'unknown'` (found in categorical features like `job`, `marital`, `education`, `default`, `housing`, and `loan`) were systematically accounted for in the feature engineering stage.
* **Categorical Encoding & Scaling:** String features were transformed into numerical matrices via structural preprocessor pipelines (`X_train_proc`, `X_test_proc`) to ensure structural compatibility across all model types.
* **Feature Alignment:** Structural features were isolated into client metrics, current campaign variables, and social/economic context attributes (such as `euribor3m` and `nr.employed`).

---

## Model Evaluation & Performance Benchmarks
We established an initial baseline model using a simple majority classifier (initial baseline accuracy of approximately **88.7%**) and compared it against four machine learning algorithms.

### 1. Baseline Model Comparisons (Default Parameters)
| Classifier | Train Time (s) | Train Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | ~0.33s | 90.0% | 89.8% |
| **Decision Tree** | ~0.43s | 100.0% | 86.39% |
| **KNN** | ~0.08s | 91.1% | 87.82% |
| **SVM** | ~41.34s | 90.2% | 89.7% |

### 2. Hyperparameter Optimization Results (Problem 11 via GridSearchCV)
To prevent model overfitting and maximize test accuracy, we implemented a robust **5-Fold Cross-Validation Grid Search** to optimize structural limits for the Decision Tree and neighborhood constraints for KNN:

* **Optimized Decision Tree Classifier:**
  * **Best Parameters Found:** `{'max_depth': 3, 'criterion': 'entropy', 'min_samples_split': 2, 'class_weight': None}`
  * **Performance Impact:** Capping the maximum depth regularized the tree, preventing it from overfitting to the training set. This significantly lifted the **Test Accuracy from 86.39% up to 88.68%**. *(Note: While a depth of 5 performed well on average across secondary parameter variations, depth 3 provided the absolute peak individual model configuration).*
* **Optimized K-Nearest Neighbors (KNN) Classifier:**
  * **Best Parameters Found:** `{'n_neighbors': 15, 'weights': 'uniform'}`
  * **Performance Impact:** Expanding the voting neighborhood to 15 neighbors smoothed decision boundaries, increasing the **Test Accuracy from 87.82% up to 88.63%**.

---

## Final Recommendations & Next Steps
1. **Production Deployment:** Deploy the **Logistic Regression** model or the **Optimized Decision Tree (max_depth=3)** for the core operational filtering queue. Both achieve robust test accuracy (~89.8% and ~88.7% respectively) with rapid execution constraints. 
2. **Avoid High-Cost Architecture:** Do not scale raw, default Support Vector Machines (SVM) into active production. While
