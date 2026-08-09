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
* **Main concepts demonstrated:** Exploratory data analysis, IQR outlier removal, Simple vs. Multiple Linear Regression, model evaluation metrics (MAE, MSE, RMSE, R² score), interactive user prediction, and L1/L2 regularization with hyperparameter tuning via cross-validation.
* **Major preprocessing/modeling steps:**
  1. Data loading and initial inspection (`head()`, `tail()`, `info`, `describe()`, missing check).
  2. Removal of non-numeric identifier columns (`Address`).
  3. Outlier filtering using Interquartile Range (IQR) on `Price` and `Avg. Area Income`.
  4. Exploratory visual analysis (heatmaps, boxplots, histograms).
  5. Training and evaluating a **Simple Linear Regression** model (`Avg. Area Income` -> `Price`) with regression plot visualizations and custom input prediction.
  6. Building a **Multiple Linear Regression** model across 5 demographic/spatial features and calculating feature coefficients.
  7. Implementing **Ridge Regression (L2)** and **Lasso Regression (L1)** with 5-fold cross-validation hyperparameter tuning (`GridSearchCV`) for optimal alpha selection.
  8. Performance evaluation using MAE, MSE, RMSE, and R² metrics, alongside actual vs. predicted price visualizations.

## Technologies Used

* **Python**: Core programming language.
* **Jupyter Notebook / Google Colab**: Interactive development environment.
* **Pandas**: Tabular data manipulation, cleaning, and inspection.
* **NumPy**: Numerical computing and array operations.
* **Matplotlib**: Data visualization and plot layout formatting.
* **Seaborn**: Statistical graphics (heatmaps, boxplots, countplots).
* **Scikit-learn**: Data preprocessing (`StandardScaler`), train-test splitting (`train_test_split`), regression modeling (`LinearRegression`, `Ridge`, `Lasso`), model evaluation (`mean_absolute_error`, `mean_squared_error`, `root_mean_squared_error`, `r2_score`), and hyperparameter tuning (`GridSearchCV`).
* **KaggleHub**: Programmatic API for fetching datasets from Kaggle (`yasserh/titanic-dataset`).

## Repository Structure

```text
Machine-Learning-Practicals/
│
├── README.md
├── Lab_1_Data_Preprocessing_Titanic.ipynb
└── Lab_2_Linear_Regression.ipynb
```

## How to Run

### Google Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File** -> **Open notebook**, then choose the **GitHub** tab.
3. Search for the GitHub repository: `HanshalBobate/Machine-Learning-Practicals`.
4. Select `Lab_1_Data_Preprocessing_Titanic.ipynb` or `Lab_2_Linear_Regression.ipynb`.
5. Execute cells sequentially (`Shift + Enter`).
   * *Lab 1 Note:* `kagglehub` will automatically download the required dataset upon execution.
   * *Lab 2 Note:* Upload `USA_Housing.csv` to your Colab session environment before running the dataset loading cell.

### Jupyter Notebook / JupyterLab
1. Clone the repository:
   ```bash
   git clone https://github.com/HanshalBobate/Machine-Learning-Practicals.git
   cd Machine-Learning-Practicals
   ```
2. Install required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
   ```
3. Start Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
4. Open and run the desired notebook (`Lab_1_Data_Preprocessing_Titanic.ipynb` or `Lab_2_Linear_Regression.ipynb`).

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
* **Prediction:** Performing model inference and generating predictions for novel input data.
* **Regression Evaluation:** Evaluating model performance quantitatively with MAE, MSE, RMSE, and R² scores, supported by graphical diagnostic plots.

## Author

**Hanshal Bobate**
