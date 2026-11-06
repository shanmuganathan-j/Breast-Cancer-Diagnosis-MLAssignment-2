# Breast Cancer Diagnosis: Classifier Comparison (Streamlit)

## 1. Problem Statement
Early and accurate classification of a breast tumour as **benign** or **malignant** helps doctors decide on follow-up. The goal of this project is to build and compare **five machine learning classifiers** on the same dataset, evaluate them with six metrics, and expose them through an interactive **Streamlit web app** where a user can upload test data, choose a model and see its performance.

## 2. Dataset Description
| Item | Detail |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic (downloaded directly from https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip by `model/train_models.py`) |
| Type | Binary classification |
| Instances | 569 (minimum required: 500) |
| Features | 30 numeric features (minimum required: 12) |
| Target | `diagnosis`: **1 = malignant (212 rows)**, **0 = benign (357 rows)** |

**Citation:** Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

The 30 features are computed from digitised images of a fine-needle aspirate of a breast mass. For each of 10 cell-nucleus properties (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) the **mean**, **standard error** and **worst** value are recorded. There are no missing values.

**Train/test split:** 80 % train (455 rows) / 20 % test (114 rows), stratified, `random_state = 2026`. The 114 held-out rows are saved as `test_data.csv` and are the file to upload in the app.

## 3. GitHub Repository Link
**GitHub Repository:** https://github.com/shanmuganathan-j/Breast-Cancer-Diagnosis-MLAssignment-2

**Live Streamlit app:** https://breast-cancer-diagnosis-mlassignment-2.streamlit.app/

### Repository structure
```
project-folder/
|-- app.py                  # Streamlit app
|-- requirements.txt
|-- README.md
|-- test_data.csv           # held-out test data (features + diagnosis)
|-- data/
|   `-- breast_cancer_full.csv  # raw file downloaded from UCI
`-- model/
    |-- train_models.py     # downloads data, trains, evaluates and saves all 5 models
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- k_nearest_neighbor.joblib
    |-- naive_bayes.joblib
    |-- support_vector_machine.joblib
    `-- metrics.csv
```
Each `.joblib` file is a scikit-learn `Pipeline`, so scaling (where needed) is saved inside the model and no separate preprocessing file is required.

## 4. Models Used

### Comparison table (evaluated on the 114-row test set)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.9987 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| Decision Tree | 0.9737 | 0.9785 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| K-Nearest Neighbor | 0.9737 | 0.9970 | 0.9535 | 0.9762 | 0.9647 | 0.9439 |
| Naive Bayes (Gaussian) | 0.9474 | 0.9927 | 0.9286 | 0.9286 | 0.9286 | 0.8869 |
| Support Vector Machine | 0.9825 | 1.0000 | 0.9545 | 1.0000 | 0.9767 | 0.9633 |

Malignant (1) is treated as the positive class for precision, recall and F1.

Settings: Logistic Regression (StandardScaler, `max_iter=2000`), Decision Tree (`max_depth=5`, `min_samples_leaf=3`), KNN (StandardScaler, `k=7`, distance-weighted), Gaussian Naive Bayes (defaults), SVM (StandardScaler, RBF kernel, `C=2`, `probability=True`).

### Observations on model performance
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong for a simple linear model: 97.4 % accuracy and an AUC of 0.9987, so it ranks malignant vs benign almost perfectly. It made 1 false positive and 2 false negatives. The scaled features and near-linear class boundary suit it well. |
| Decision Tree | Same accuracy (97.4 %) and confusion matrix as Logistic Regression, but the lowest AUC (0.9785). A tree outputs coarse leaf probabilities, so its ranking quality is weaker even though its hard predictions are good. Also the model most sensitive to the particular train/test split. |
| KNN | 97.4 % accuracy with the second-best recall (0.9762, only 1 missed malignant case) but more false alarms (2) than Logistic Regression. Performance depends on feature scaling, which the pipeline handles. |
| Naive Bayes | Weakest model: 94.7 % accuracy, MCC 0.8869, with 3 false positives and 3 false negatives. Its assumption that features are independent is violated here, since radius, perimeter and area are highly correlated. It is still fast and has a high AUC (0.9927). |
| Support Vector Machine | Best on accuracy (98.3 %), recall (1.0), F1, MCC (0.9633) and AUC (1.0). It missed no malignant tumour (0 false negatives), at the cost of 2 false positives. The RBF kernel captures the smooth non-linear boundary well. |
| **Overall winner** | **Support Vector Machine**: highest on five of six metrics and zero false negatives, which matters most in cancer screening. Logistic Regression is the best simple and interpretable alternative. Caveat: with only 114 test rows, a one-sample difference moves accuracy by about 0.9 points, so the differences between the top four models are small. |

## 5. Streamlit App Features
The app uses a sidebar **Project details** menu with four pages:

| Page | What it shows |
|---|---|
| **Overview** | Problem statement, key dataset numbers, how to use the app, and the held-out test results from training |
| **Evaluation metrics and Classification report** | Model dropdown, Accuracy / AUC / Precision / Recall / F1 / MCC, confusion matrix (counts or %), ROC curve, classification report, and a downloadable predictions file |
| **Compare models** | Side-by-side table of all 5 models, best model per metric, bar charts, combined ROC curves, and a downloadable comparison table |
| **Dataset and methodology** | Dataset description, class distribution, feature correlation heatmap, preprocessing, model settings, metric definitions and limitations |

Other features:
- Upload a test CSV from the sidebar (must contain all 30 feature columns and `diagnosis`) or download the sample `test_data.csv`
- Clear error messages for missing columns, missing values or a single-class file
- Models are loaded from `model/`; nothing is retrained on uploaded data

## 6. Run Locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
