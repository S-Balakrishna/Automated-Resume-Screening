import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ensure models directory exists
models_dir = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(models_dir, exist_ok=True)

# Load dataset
file_path = os.path.join(os.path.dirname(__file__), "../datasets/resume_data.csv")
df = pd.read_csv(file_path)

# Validate required columns
required_columns = {"Resume", "Category"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Dataset must contain columns: {required_columns}")

# Text Vectorization
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X = tfidf.fit_transform(df["Resume"])
y = df["Category"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Training Complete! Accuracy: {accuracy * 100:.2f}%")

# Save Model and Vectorizer
pickle.dump(model, open(os.path.join(models_dir, "trained_model.pkl"), "wb"))
pickle.dump(tfidf, open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "wb"))

print("✅ Model and vectorizer saved successfully!")
