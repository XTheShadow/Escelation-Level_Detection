import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import shap
from imblearn.under_sampling import RandomUnderSampler
import joblib

# Defining the  columns we will use
selected_columns = [
    'YEAR', 'IRSEX', 'CATAGE', 'INCOME', 'EMPSTATY', 'EDUCCAT2', 'COUTYP2',
    'ALCAFU', 'MJAGE', 'COCAGE', 'HERAGE', 'CIGAFU', 'IRHALAGE'
]

# Setting dtypes to optimize memory usage
dtypes = {
    'YEAR': 'int16', 'IRSEX': 'category', 'CATAGE': 'category', 'INCOME': 'category',
    'EMPSTATY': 'category', 'EDUCCAT2': 'category', 'COUTYP2': 'category',
    'ALCAFU': 'float16', 'MJAGE': 'float16', 'COCAGE': 'float16',
    'HERAGE': 'float16', 'CIGAFU': 'float16', 'IRHALAGE': 'float16'
}

# Loading the data with the selected columns
data_chunks = []
for chunk in pd.read_csv('dataset/NSDUH_2002_2018_Tab.tsv', sep='\t', usecols=selected_columns, dtype=dtypes, chunksize=5000, low_memory=False):
    data_chunks.append(chunk)
data = pd.concat(data_chunks, ignore_index=True)

# Step 1: Data Preprocessing
# Filling NaN values in drug usage columns with -1 assuming NaN means no use
drug_columns = ['ALCAFU', 'MJAGE', 'COCAGE', 'HERAGE', 'CIGAFU', 'IRHALAGE']
data[drug_columns] = data[drug_columns].fillna(-1)

# Step 2: Defining the Escalation Levels based on substance usage
def determine_escalation(row):
    # List of substances
    substances = [row[col] for col in drug_columns]
    substance_count = sum(1 for substance in substances if substance == 1)  # Using this We can Count substances with value 1

    # Define escalation levels
    if substance_count == 0:  # No substance use
        return "Never Used"
    elif substance_count == 1:  # Single substance user
        return "Single Substance User"
    elif substance_count == 2:  # Dual substance user
        return "Dual Substance User"
    else:  # Multiple substance user
        return "Multiple Substance User"

# Applying the updated function to determine escalation levels
data['Escalation_Level'] = data.apply(determine_escalation, axis=1)

# Encoding Escalation_Level to numeric for model training
label_encoder = LabelEncoder()
data['Escalation_Level'] = label_encoder.fit_transform(data['Escalation_Level'])

# Verifying escalation levels
escalation_labels = label_encoder.inverse_transform(sorted(data['Escalation_Level'].unique()))

# Features and target based on user-related data
X = data[['YEAR', 'IRSEX', 'CATAGE', 'INCOME', 'EMPSTATY', 'EDUCCAT2', 'COUTYP2']]
y = data['Escalation_Level']

# Encoding categorical demographic features
for col in X.select_dtypes(include='category').columns:
    X[col] = label_encoder.fit_transform(X[col])

# Checking the class distribution
print("Class distribution before splitting:")
print(y.value_counts())

# Step 3: Here we apply Under-sampling to balance the classes
#undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
#X_res, y_res = undersampler.fit_resample(X, y)

# Step 1: Limit the larger classes
undersampler = RandomUnderSampler(sampling_strategy={0: 21846, 2: 21846, 3: 21846}, random_state=42)
X_under, y_under = undersampler.fit_resample(X, y)

# Step 2: Combine under-sampled data with the minority class (class 1)
X_balanced = pd.concat([X_under, X[y == 1]], axis=0)
y_balanced = pd.concat([y_under, y[y == 1]], axis=0)

# Step 3: Apply SMOTE to minority classes (class 1) to balance the dataset
smote = SMOTE(sampling_strategy={1: 21846}, random_state=42)
X_res, y_res = smote.fit_resample(X_balanced, y_balanced)


# Verifying the new class distribution
print("New class distribution after Hybrid sampling:")
print(pd.Series(y_res).value_counts())

# Step 4: Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# Step 5: Scaling Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Define parameter grid for XGBoost
param_grid = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6],
    'n_estimators': [200, 300],
    'subsample': [0.8]
}


# Hyperparameter tuning with GridSearchCV
grid_search = GridSearchCV(XGBClassifier(eval_metric='mlogloss', random_state=42), param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Retrieve the best estimator
best_xgb = grid_search.best_estimator_
print(f"Best Parameters for XGBoost: {grid_search.best_params_}")

# Step 6: Define classifiers & Training and Evaluating the Models
classifiers = {
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost Classifier": grid_search.best_estimator_,
    "Support Vector Machine": SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
}

# Cross-validation for all classifiers
print("Cross-Validation Performance:")
for name, model in classifiers.items():
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    print(f"{name} - Cross-Validation Mean Accuracy: {np.mean(scores):.2f}")


joblib.dump(best_xgb, "best_xgboost_model.pkl")
print("Best XGBoost model saved as best_xgboost_model.pkl")


print("Classification Model Performance:")
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} - Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred, target_names=escalation_labels))
    print("-" * 40)


# Explain feature importance with SHAP for XGBoost
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer(X_test)

# Generate one summary plot for all features
shap.summary_plot(shap_values, X_test, feature_names=X.columns, max_display=10)



# Step 7: Applying Cross-Validation with Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X_res, y_res, cv=5)
print("Cross-Validation Scores (Random Forest):", scores)
print("Mean Cross-Validation Score:", np.mean(scores))


# Plotting class distribution before sampling
plt.figure(figsize=(10, 6))
data['Escalation_Level'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Class Distribution Before Sampling')
plt.xlabel('Escalation Levels')
plt.ylabel('Count')
plt.xticks(ticks=range(len(escalation_labels)), labels=escalation_labels, rotation=30,)
plt.show()

# Plotting class distribution after sampling
plt.figure(figsize=(10, 6))
pd.Series(y_res).value_counts().plot(kind='bar', color='orange')
plt.title('Class Distribution After Undersampling')
plt.xlabel('Escalation Levels')
plt.ylabel('Count')
plt.xticks(ticks=range(len(escalation_labels)), labels=escalation_labels, rotation=30,)
plt.show()


# Plotting Feature Importances
rf.fit(X_train, y_train)
importances = rf.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[sorted_indices], align="center", color='purple')
plt.xticks(range(X.shape[1]), feature_names[sorted_indices], rotation=30, )
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


# Generating and plotting confusion matrix for each classifier
for name, model in classifiers.items():
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=escalation_labels, cmap='Blues', xticks_rotation=30, )
    plt.title(f"Confusion Matrix: {name}")
    plt.show()

# Bar chart comparing accuracies of classifiers
classifier_accuracies = {name: accuracy_score(y_test, model.predict(X_test)) for name, model in classifiers.items()}
plt.figure(figsize=(10, 6))
plt.bar(classifier_accuracies.keys(), classifier_accuracies.values(), color='teal')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=30, 
)
plt.tight_layout()
plt.show()

# Computing correlation matrix for numerical features
correlation_matrix = data[drug_columns + ['INCOME']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.show()