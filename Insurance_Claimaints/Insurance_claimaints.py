# 📦 Logistic Regression on Insurance Claimants Dataset

# ================================
# 📚 IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score
)

# ================================
# 📂 LOAD DATA
# ================================
df = pd.read_csv('claimants.csv')
df.drop(columns=['CASENUM'], inplace=True)

# ================================
# 🧹 DATA CLEANING
# ================================
df.drop_duplicates(inplace=True)
df['CLMAGE'].fillna(df['CLMAGE'].median(), inplace=True)
for col in ['ATTORNEY', 'CLMSEX', 'CLMINSUR', 'SEATBELT']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# ================================
# 🎯 FEATURES & TARGET
# ================================
X = df.drop(columns='ATTORNEY')
y = df['ATTORNEY']

# ================================
# 🧪 TRAIN-TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# 🔁 PIPELINE: IMPUTE + SCALE + MODEL
# ================================
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(solver='liblinear'))
])

pipeline.fit(X_train, y_train)

# ================================
# 📈 EVALUATION
# ================================
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy :", accuracy_score(y_test, y_test_pred))
print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred))
print("ROC AUC Score (Train):", roc_auc_score(y_train, y_train_pred))
print("ROC AUC Score (Test):", roc_auc_score(y_test, y_test_pred))
