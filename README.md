# 🌋Escalation Level Detection

## 📖 Description
Machine learning system for classifying substance use escalation levels using demographic data from the National Survey of Drug Use and Health (NSDUH). Analyzes patterns across 2002-2018 data to predict four risk categories with comprehensive model evaluation.

---

## 📌 Key Features
- Hybrid sampling (SMOTE + Undersampling) for class imbalance
- Multiple classifier comparison (XGBoost, Random Forest, SVM)
- GridSearchCV hyperparameter optimization
- SHAP value analysis for feature importance
- Comprehensive performance visualizations
- Demographic feature correlation analysis

---

## 🚀 Setup & Installation

### 1️. Clone the repository:
```bash
git clone https://github.com/XTheShadow/Escelation-Level_Detection.git
cd Escelation-Level_Detection
```


### 2️. Set up the virtual environment:
```bash
python -m venv .venv
```


#### Windows:
```bash
.\.venv\Scripts\activate
```

#### macOS/Linux:
```bash
source .venv/bin/activate
```


### 3️. Install dependencies:
```bash
pip install -r requirements.txt
```

#### 4. Download the NSDUH dataset from [Kaggle](https://www.kaggle.com/datasets/adamhamrick/national-survey-of-drug-use-and-health-20022018).

#### 5. Place the TSV file in the "dataset" folder.

### 6. Execute the analysis script:
```bash
python escalation_level_detection.py
```


---

## 🛠️ Methodology

### Data Pipeline
- **Preprocessing:**
  - Missing value handling for substance columns
  - Label encoding for categorical features
  - StandardScaler normalization


- **Class Balancing:**
  - Strategic under-sampling (RandomUnderSampler)
  - SMOTE oversampling for minority classes

    
### Machine Learning Models
- **XGBoost:** Optimized with GridSearchCV (learning_rate=0.1, max_depth=6)
- **Random Forest:** 100 estimators with Gini impurity
- **SVM:** RBF kernel with C=1.0

---


## 🎯 Key Results

### Model Performance
```bash
| Model        | Accuracy | Cross-Val Score |
|--------------|----------|-----------------|
| XGBoost      | 0.89     | 0.88            |
| Random Forest| 0.87     | 0.85            |
| SVM          | 0.83     | 0.81            |
```


### Insights
- Top predictors: **County type (COUTYP2)**, **Age category (CATAGE)**
- Secondary factors: **Income level**, **Employment status**
- Hybrid sampling improved minority class recall by 32%


### Visualizations
- Class distribution before/after sampling
- Feature importance comparison
- Confusion matrices per model
- SHAP summary plots
- Feature correlation heatmap

---

## 📜 License
This project is licensed under the MIT License.