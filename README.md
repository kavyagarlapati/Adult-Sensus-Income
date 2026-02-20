# Adult Census Income Classification

This notebook demonstrates the implementation of supervised machine learning models to predict whether an individual’s annual income exceeds $50,000 based on demographic and socio-economic attributes.

The analysis is based on the U.S. Census Bureau’s Current Population Survey (CPS) data and focuses on preprocessing structured data, performing domain analysis, and evaluating multiple classification algorithms using Scikit-learn.

---

## Notebook Overview

The objective of this notebook is to classify individuals into income categories:

* Income ≤ 50K
* Income > 50K

The notebook includes:

* Domain analysis of census-based attributes
* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Handling categorical and numerical variables
* Pipeline-based preprocessing
* Model training and evaluation
* Cross-validation for performance comparison

The purpose is to build a structured machine learning workflow for real-world socio-economic classification problems.

---

## Dataset

The dataset is derived from the Adult Census Income dataset (UCI repository), downloaded using KaggleHub.

It contains demographic and employment-related attributes such as:

* Age
* Workclass
* Education
* Marital Status
* Occupation
* Relationship
* Race
* Sex
* Capital Gain
* Capital Loss
* Hours per Week
* Native Country

### Target Variable

* income :

  * ≤ 50K
  * > 50K

The dataset reflects weighted population estimates from the U.S. Census Bureau.

---

## Domain Analysis

Income level is influenced by multiple demographic and socio-economic factors, including:

* Education level and years of schooling
* Occupation type
* Working hours per week
* Marital status
* Capital gains and losses
* Employment category

Understanding these variables helps in identifying patterns that contribute to higher or lower income classifications.

---

## Data Preprocessing

The notebook includes:

* Initial data inspection (`head()`, `info()`, `describe()`)
* Missing value verification
* Categorical and numerical feature separation
* Label encoding
* Feature scaling using `MinMaxScaler`
* Custom transformation classes using `TransformerMixin`
* Pipeline-based preprocessing for numerical and categorical features
* One-hot encoding using `pd.get_dummies()`
* Train-test split

Custom preprocessing pipelines are built to ensure structured and reusable data transformation.

---

## Mathematical Foundation

The implemented models follow supervised classification principles where:

* Input features (X) represent demographic and employment attributes
* Target variable (y) represents income category
* The model learns decision boundaries to classify individuals based on feature patterns

Performance evaluation includes:

* Accuracy
* Precision
* Recall
* Confusion Matrix
* Classification Report
* ROC Curve and AUC
* Cross-validation (10-fold)

---

## Workflow

Import required libraries
Download and load dataset
Perform exploratory data analysis
Separate categorical and numerical features
Apply feature scaling and encoding
Build preprocessing pipelines
Split data into training and testing sets
Train classification models
Evaluate models using multiple metrics
Perform cross-validation for performance validation

---

## Models Implemented

* Logistic Regression
* Decision Tree Classifier
* K-Nearest Neighbors (KNN)

Each model is trained and evaluated, and cross-validation is used to compare average accuracy scores.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* KaggleHub

---

## Training Credit

This notebook was completed as part of the Machine Learning training conducted by Codinza.

### Learning Outcome

* Gained practical experience in handling real-world census datasets
* Built preprocessing pipelines using Scikit-learn
* Applied feature scaling and one-hot encoding
* Implemented and compared multiple classification models
* Used cross-validation for robust performance evaluation
* Strengthened understanding of supervised learning for socio-economic prediction tasks

---

## Author

Kavya Garlapati || Developer || [LinkedIn ↗](https://www.linkedin.com/in/kavya-garlapati/)

