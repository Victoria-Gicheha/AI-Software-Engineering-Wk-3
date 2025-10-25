# ================================================================
# Task 1: Classical Machine Learning with Scikit-learn
# Dataset: Iris Species Dataset
# Goal:
#   - Preprocess the data (handle missing values, encode labels)
#   - Train a Decision Tree Classifier to predict iris species
#   - Evaluate using accuracy, precision, and recall
# ================================================================

# === 1. Import required libraries ===
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import os

# === 2. Load the Iris dataset ===
iris = datasets.load_iris()

# Features and labels
X = iris.data
y = iris.target

# Feature names and target names
feature_names = iris.feature_names
target_names = iris.target_names

# Convert to DataFrame for easier analysis
df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, target_names)

# Display the first few rows
print("Sample of the dataset:")
print(df.head(), "\n")

# === 3. Preprocessing ===
# 3.1 Handle missing values (for demonstration; Iris dataset has none)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(df[feature_names])

# 3.2 Encode labels (convert text labels to integers)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['species'])

# === 4. Split the data into training and testing sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === 5. Train the Decision Tree Classifier ===
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# === 6. Evaluate the model ===
y_pred = clf.predict(X_test)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

print("=== Evaluation Metrics ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall: {recall:.4f}\n")

# Detailed report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# === 7. Visualize Confusion Matrix ===
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Label the axes
plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, rotation=45)
plt.yticks(range(len(label_encoder.classes_)), label_encoder.classes_)

# Show counts in the cells
for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, str(val), ha='center', va='center', color='black')

plt.tight_layout()
plt.show()

# === 8. Cross-validation (optional, for better performance check) ===
cv_scores = cross_val_score(clf, X_imputed, y_encoded, cv=5, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean CV Accuracy: {:.4f}\n".format(cv_scores.mean()))

# === 9. View Decision Tree Rules ===
print("=== Decision Tree Rules ===")
print(export_text(clf, feature_names=feature_names))

# === 10. Save Model and Preprocessing Objects ===
os.makedirs('iris_artifacts', exist_ok=True)
joblib.dump(clf, 'iris_artifacts/decision_tree_iris.joblib')
joblib.dump(imputer, 'iris_artifacts/imputer.joblib')
joblib.dump(label_encoder, 'iris_artifacts/label_encoder.joblib')
print("\nSaved model and preprocessing files in 'iris_artifacts/' folder.")
