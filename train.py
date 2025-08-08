import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import os
import seaborn as sns
import json

# Load dataset
df = sns.load_dataset("penguins")
df = df.dropna()

# Encode label
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

# Save class names
os.makedirs("app/data", exist_ok=True)
with open("app/data/label_classes.json", "w") as f:
    json.dump(list(le.classes_), f)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=["sex", "island"])

# Save columns order to ensure consistency during inference
feature_columns = [col for col in df.columns if col != "species"]
with open("app/data/preprocess_meta.json", "w") as f:
    json.dump({"feature_columns": feature_columns}, f)

X = df[feature_columns]
y = df["species"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    use_label_encoder=False,
    eval_metric="mlogloss"
)
model.fit(X_train, y_train)

# Evaluate
print("\nTrain Report:")
print(classification_report(y_train, model.predict(X_train), target_names=le.classes_))
print("\nTest Report:")
print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, model.predict(X_test)))

# Save model
model.save_model("app/data/model.json")
print("✅ Model and metadata saved.")
