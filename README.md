# Machine Learning Practicals

This repository contains my Machine Learning laboratory practicals, implemented in Python using Jupyter Notebook and Google Colab.

## Student Details

* **Name:** Hanshal Bobate
* **Section:** B
* **Roll No.:** 40
* **Batch:** B3

## Practicals

| Lab | Practical | Notebook |
| --- | --- | --- |
| 1 | Data Preprocessing on Titanic Dataset | [`Lab_1_Data_Preprocessing_Titanic.ipynb`](Lab_1_Data_Preprocessing_Titanic.ipynb) |
| 2 | Linear Regression for Price Prediction | [`Lab_2_Linear_Regression.ipynb`](Lab_2_Linear_Regression.ipynb) |
| 3 | Find-S and Candidate Elimination | [`Lab_3_Find_S_Candidate_Elimination.ipynb`](Lab_3_Find_S_Candidate_Elimination.ipynb) |
| 4 | Wine Quality Prediction | [`Lab_4_Wine_Quality.ipynb`](Lab_4_Wine_Quality.ipynb) |
| 5 | K-Nearest Neighbours Classification | [`Lab_5_K-Nearest_Neighbours.ipynb`](Lab_5_K-Nearest_Neighbours.ipynb) |
| 6 | Perceptron Learning Algorithm | [`Lab_6_Perceptron_Learning_Algorithm.ipynb`](Lab_6_Perceptron_Learning_Algorithm.ipynb) |
| 7 | Naïve Bayes Classification | [`Lab_7_Naïve_Bayes_Classification.ipynb`](Lab_7_Naïve_Bayes_Classification.ipynb) |

### Lab 1: Data Preprocessing on Titanic Dataset

* **Objective:** To study and apply comprehensive data preprocessing techniques on the Titanic dataset, preparing raw tabular data for machine learning algorithms.
* **Dataset used:** Titanic Dataset (`Titanic-Dataset.csv`, fetched automatically via `kagglehub` from `yasserh/titanic-dataset`).
* **Main concepts demonstrated:** Data inspection and statistics, missing data identification and visualization, outlier detection and filtering, categorical encoding, exploratory data analysis (EDA), feature scaling, and train-test splitting.
* **Major preprocessing/modeling steps:**
  1. Automated dataset retrieval using `kagglehub`.
  2. Structural analysis (`shape`, `info()`, `describe()`) and missing value detection (`isnull().sum()`, missingness heatmap).
  3. Feature selection and dropping highly incomplete columns (e.g., `Cabin`).
  4. Outlier removal on numerical attributes (`Fare`, `Age`) using Interquartile Range (IQR) thresholding.
  5. Categorical encoding using label mapping (`Sex`, `Embarked`).
  6. Univariate, bivariate, and multivariate Exploratory Data Analysis (countplots, correlation matrices).
  7. Feature scaling on numerical inputs using `StandardScaler`.
  8. Dataset partitioning into training and testing subsets via `train_test_split`.

### Lab 2: Linear Regression for Price Prediction

* **Objective:** To develop, evaluate, and compare Simple Linear Regression, Multiple Linear Regression, and Regularized Regression (Ridge & Lasso) models for housing price prediction.
* **Dataset used:** USA Housing Dataset (`USA_Housing.csv`).
* **Main concepts demonstrated:** Exploratory data analysis, IQR outlier removal, Simple vs. Multiple Linear Regression, model evaluation metrics (MAE, MSE, RMSE, R² score), and regression visualization.
* **Major preprocessing/modeling steps:**
  1. Data loading and initial inspection (`head()`, `tail()`, `info`, `describe()`, missing check).
  2. Removal of non-numeric identifier columns (`Address`).
  3. Outlier filtering using IQR on `Price` and `Avg. Area Income`.
  4. Exploratory visual analysis (heatmaps, boxplots, histograms).
  5. Training and evaluating a Simple Linear Regression model (`Avg. Area Income` -> `Price`).
  6. Building a Multiple Linear Regression model across 5 demographic/spatial features.
  7. Implementing Ridge and Lasso Regression with 5-fold cross-validation hyperparameter tuning (`GridSearchCV`).
  8. Performance evaluation using MAE, MSE, RMSE, and R² metrics, alongside actual-vs-predicted visualizations.

### Lab 3: Find-S and Candidate Elimination

* **Objective:** To implement the Find-S and Candidate Elimination algorithms on a concept-learning dataset and compare the resulting hypotheses.
* **Dataset used:** Enjoy Sports dataset (`enjoy.csv`).
* **Main concepts demonstrated:** Inductive learning, hypothesis space, concept learning, and version space boundary refinement.
* **Major steps:**
  1. Load and inspect the labelled training examples.
  2. Initialize the most specific hypothesis for Find-S.
  3. Generalize the hypothesis using positive examples only.
  4. Maintain the specific and general boundaries using Candidate Elimination.
  5. Interpret and compare the final learned hypotheses.

### Lab 4: Wine Quality Prediction

* **Objective:** To analyze the wine quality dataset and build classification models to predict wine quality based on physicochemical characteristics.
* **Dataset used:** Wine Quality Dataset (`winequality.csv` or equivalent dataset used in the notebook).
* **Main concepts demonstrated:** Data exploration, feature analysis, classification modeling, and performance assessment.
* **Major steps:**
  1. Load and inspect the dataset.
  2. Analyze feature distributions and relationships with the target quality score.
  3. Preprocess data and prepare features for model training.
  4. Train and evaluate classification models for predicting wine quality.
  5. Review accuracy and predictive performance using model metrics.

### Lab 5: K-Nearest Neighbours Classification

* **Objective:** To explore the diabetes dataset and examine classification behavior through K-Nearest Neighbours (KNN).
* **Dataset used:** Diabetes Dataset (`diabetes.csv`).
* **Main concepts demonstrated:** Exploratory data analysis, feature influence, distance-based classification, and model evaluation.
* **Major steps:**
  1. Import required libraries and read the dataset.
  2. Inspect records, structure, and summary statistics.
  3. Check missing values and duplicate rows.
  4. Perform univariate and multivariate analysis of features.
  5. Study relationships between predictor variables and outcome labels.
  6. Analyze correlations and evaluate the relevance of features for KNN classification.

### Lab 6: Perceptron Learning Algorithm

* **Objective:** To study the seed classification dataset and build a neural-network-style a classification model using the Perceptron learning approach.
* **Dataset used:** Seeds Dataset (`seeds_new.csv`).
* **Main concepts demonstrated:** Feature preprocessing, standardization, classification, and training/validation analysis.
* **Major steps:**
  1. Load and preview the dataset.
  2. Inspect structure, descriptive statistics, missing values, and outliers.
  3. Explore class distribution and feature-to-class relationships.
  4. Split features and labels into train/test sets.
  5. Standardize numerical features using `StandardScaler`.
  6. Build a multi-layer neural network classifier.
  7. Train the model and evaluate it using classification metrics and confusion matrix.

### Lab 7: Naïve Bayes Classification

* **Objective:** To classify raisin varieties using a Gaussian Naïve Bayes model and assess its performance on the Raisin dataset.
* **Dataset used:** Raisin Grains Dataset (`Raisin_Grains_Dataset.csv`).
* **Main concepts demonstrated:** Feature analysis, class encoding, probabilistic classification, and classifier evaluation.
* **Major steps:**
  1. Import the libraries and machine learning utilities.
  2. Load and inspect the dataset.
  3. Perform EDA: missing values, duplicates, feature distributions, outliers, and correlations.
  4. Encode the target class labels.
  5. Split data into training and testing sets using stratified sampling.
  6. Train a `GaussianNB` classifier.
  7. Compare predicted and actual labels, and compute accuracy, precision, recall, F1-score, classification report, confusion matrix, and training/testing accuracy.

## Technologies Used

* **Python**: Core programming language.
* **Jupyter Notebook / Google Colab**: Interactive development environment.
* **Pandas**: Tabular data manipulation, cleaning, and inspection.
* **NumPy**: Numerical computing and array operations.
* **Matplotlib**: Data visualization and plot layout formatting.
* **Seaborn**: Statistical graphics (heatmaps, boxplots, countplots).
* **Scikit-learn**: Data preprocessing (`StandardScaler`), train-test splitting (`train_test_split`), regression modeling (`LinearRegression`, `Ridge`, `Lasso`), KNN, Naïve Bayes, and evaluation metrics.
* **KaggleHub**: Programmatic API for fetching datasets from Kaggle (`yasserh/titanic-dataset`).
* **TensorFlow / Keras**: Neural network modeling for the perceptron-based classification labs.

## Repository Structure

```text
Machine-Learning-Practicals/
│
├── README.md
├── Lab_1_Data_Preprocessing_Titanic.ipynb
├── Lab_2_Linear_Regression.ipynb
├── Lab_3_Find_S_Candidate_Elimination.ipynb
├── Lab_4_Wine_Quality.ipynb
├── Lab_5_K-Nearest_Neighbours.ipynb
├── Lab_6_Perceptron_Learning_Algorithm.ipynb
├── Lab_7_Naïve_Bayes_Classification.ipynb
└── other dataset files (if used in individual notebooks)
```

## How to Run

### Google Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File** -> **Open notebook**, then choose the **GitHub** tab.
3. Search for the GitHub repository: `HanshalBobate/Machine-Learning-Practicals`.
4. Select any practical notebook from Lab 1 to Lab 7.
5. Execute cells sequentially (`Shift + Enter`).
   * *Lab 1 Note:* `kagglehub` will automatically download the required dataset upon execution.
   * *Lab 2 Note:* Upload `USA_Housing.csv` to your Colab session environment before running the dataset loading cell.
   * *Other Labs:* Ensure the corresponding dataset file is available in the runtime environment, as required by each notebook.

### Jupyter Notebook / JupyterLab
1. Clone the repository:
   ```bash
   git clone https://github.com/HanshalBobate/Machine-Learning-Practicals.git
   cd Machine-Learning-Practicals
   ```
2. Install required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn kagglehub tensorflow
   ```
3. Start Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
4. Open and run the desired notebook (`Lab_1_Data_Preprocessing_Titanic.ipynb` through `Lab_7_Naïve_Bayes_Classification.ipynb`).

## Learning Outcomes

These practicals demonstrate essential end-to-end Machine Learning techniques:
* **Data Inspection:** Assessing dataset structure, statistics, types, and missing values.
* **Missing-Value Handling:** Detecting missing data patterns and eliminating uninformative/incomplete features.
* **Data Preprocessing:** Structuring, cleaning, and validating raw datasets for model input.
* **Outlier Handling:** Identifying and filtering extreme observations using Interquartile Range (IQR) bounds.
* **Categorical Encoding:** Converting non-numeric features into numerical values using mapping techniques.
* **Feature Scaling:** Standardizing feature distributions using `StandardScaler`.
* **Train/Test Splitting:** Dividing data into training and validation sets to evaluate model generalization.
* **Linear Regression:** Building Simple, Multiple, and Regularized (Ridge & Lasso) Linear Regression models.
* **Concept Learning:** Implementing Find-S and Candidate Elimination for hypothesis generation.
* **Classification Algorithms:** Applying KNN, Perceptron-based learning, and Naïve Bayes approaches.
* **Prediction:** Performing model inference and generating predictions for novel input data.
* **Regression & Classification Evaluation:** Assessing model performance quantitatively with metrics such as MAE, MSE, RMSE, R², accuracy, precision, recall, and F1-score.

## Author

**Hanshal Bobate**
