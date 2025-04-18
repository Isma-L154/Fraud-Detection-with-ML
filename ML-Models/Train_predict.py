import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ========== Set Up File Paths ==========

# Get the absolute path of the current script (train_predict.py)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the 'Data' directory (go one level up and into 'Data')
DATA_DIR = os.path.join(BASE_PATH, "..", "Data")

PREDICTS_DIR = os.path.join(DATA_DIR, "Predicts")

# Define full paths to each relevant file
ORIGINAL_DATA_PATH = os.path.join(DATA_DIR, "Seguros.csv")               # Original dataset with 'Fraude' column
NEW_DATA_PATH = os.path.join(DATA_DIR, "SegurosNuevos.csv")              # New records to classify
OUTPUT_DATA_PATH = os.path.join(PREDICTS_DIR, "SegurosML_Predict.csv")   # Output with predictions

# ========== Load and Prepare Training Data ==========

# Load original data
OrgData = pd.read_csv(ORIGINAL_DATA_PATH)

# Initialize label encoder
LabelNum = LabelEncoder()

# Encode all categorical features except the target column
for col in OrgData.columns:
    if OrgData[col].dtype == 'object' and col.lower() != 'fraude':
        OrgData[col] = LabelNum.fit_transform(OrgData[col].astype(str))

# Separate features (X) and target variable (y)
x = OrgData.drop(columns=["Fraude"])  # Independent variables
y = OrgData["Fraude"].apply(lambda x: 1 if str(x).strip().lower() == "si" else 0)  # Target as binary (1 = fraud)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# ========== Model Selection Function ==========

def select_model(model_name):
    """Dynamically select and return a machine learning model based on the input."""
    if model_name == "random_forest":
        return RandomForestClassifier()
    elif model_name == "logistic_regression":
        return LogisticRegression()
    elif model_name == "svm":
        return SVC()
    elif model_name == "decision_tree":
        return DecisionTreeClassifier()
    elif model_name == "knn":
        return KNeighborsClassifier()
    elif model_name == "gradient_boosting":
        return GradientBoostingClassifier()
    else:
        raise ValueError(f"Model {model_name} is not supported. Please choose from: random_forest, logistic_regression, svm, decision_tree, knn, gradient_boosting.")

# ========== Choose the Model Dynamically ==========

# Specify the model you want to use here (for example: 'random_forest', 'logistic_regression', etc.)
selected_model = "random_forest"  # You can change this value

# Get the model based on user selection
Model = select_model(selected_model)

# Train the model
Model.fit(X_train, y_train)

# Predict on test set and print evaluation
Predict = Model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, Predict))

# ========== Load New Data for Prediction ==========

# Read the new dataset
NewData = pd.read_csv(NEW_DATA_PATH)

# Drop any columns not used during training
for col_to_drop in ["ID", "Fraude"]:
    if col_to_drop in NewData.columns:
        NewData = NewData.drop(columns=[col_to_drop])

# Encode categorical features in the new data
for col in NewData.columns:
    if NewData[col].dtype == 'object':
        NewData[col] = LabelNum.fit_transform(NewData[col].astype(str))

# ========== Predict and Save Results ==========

# Predict fraud for the new dataset
NewPredict = Model.predict(NewData)

# Add predicted labels
NewData["Fraude"] = ["SI" if pred == 1 else "NO" for pred in NewPredict]

# Save the predictions to a new CSV file
NewData.to_csv(OUTPUT_DATA_PATH, index=False)
print(f"✅ Prediction complete. File saved at: {OUTPUT_DATA_PATH}")
