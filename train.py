from pathlib import Path
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = Path("data")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

def load_dataset():
    fake_path = DATA_DIR / "Fake.csv"
    true_path = DATA_DIR / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            "Place Fake.csv and True.csv inside the data/ folder. "
            "Download them from the Kaggle dataset linked in README.md."
        )

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    for frame in (fake, true):
        for col in ("title", "text"):
            if col not in frame.columns:
                frame[col] = ""

    fake["label"] = 0  # FAKE
    true["label"] = 1  # REAL

    data = pd.concat([fake, true], ignore_index=True)
    data["title"] = data["title"].fillna("").astype(str)
    data["text"] = data["text"].fillna("").astype(str)
    data["content"] = (data["title"] + " " + data["text"]).str.strip()
    data = data[data["content"].str.len() > 30]
    data = data.drop_duplicates(subset=["content"])
    return data[["content", "label"]]

def main():
    data = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        data["content"],
        data["label"],
        test_size=0.20,
        random_state=42,
        stratify=data["label"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    print(f"Training on {len(X_train):,} articles...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, target_names=["FAKE", "REAL"], output_dict=True
    )
    matrix = confusion_matrix(y_test, predictions).tolist()

    joblib.dump(model, MODEL_DIR / "fake_news_pipeline.joblib")
    metrics = {
        "accuracy": round(float(accuracy), 4),
        "classification_report": report,
        "confusion_matrix": matrix,
        "test_rows": int(len(X_test)),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nTest accuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["FAKE", "REAL"]))
    print("Confusion matrix [ [fake→fake, fake→real], [real→fake, real→real] ]:")
    print(matrix)
    print("\nSaved model to model/fake_news_pipeline.joblib")

if __name__ == "__main__":
    main()
