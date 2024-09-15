# Import necessary libraries
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the sample dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the XGBoost model with hyperparameters
xgb_model = xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=4, min_child_weight=1,
                              gamma=0, subsample=0.8, colsample_bytree=0.8, random_state=42)

# Set up MLflow experiment
mlflow.set_experiment("XGBoost_Breast_Cancer_Classification")

# Start an MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 4)
    # ... other parameters

    # Train the model
    xgb_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = xgb_model.predict(X_test_scaled)
    
    # Log metrics
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    
    # Register the model in MLflow
    mlflow.xgboost.log_model(xgb_model, "xgboost_model")
    
    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.2f}")