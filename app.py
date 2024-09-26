import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from preprocessing import Preprocessing


def load_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path, sep=",", header="infer")
    df.columns = [
        "id",
        "text",
        "rating",
        "category",
        "product_name",
        "product_id",
        "sold",
        "shop_id",
        "url",
    ]
    return df


def clean_data(df, pr):
    """Clean the review text data."""
    df["text"] = (
        df["text"].apply(pr.processtext).apply(pr.stem).apply(pr.remove_stopwords)
    )
    df = df.drop_duplicates()
    return df


def label_data(df):
    """Label the reviews based on rating."""
    df["label"] = df["rating"].apply(lambda x: True if x > 4 else False)
    return df[["text", "label"]]


def train_model(X_train, y_train):
    """Train the Naive Bayes model with grid search."""
    pipeline = Pipeline(
        [
            ("bow", CountVectorizer(strip_accents="ascii", lowercase=True)),
            ("tfidf", TfidfTransformer()),
            ("classifier", MultinomialNB()),
        ]
    )

    parameters = {
        "bow__ngram_range": [(1, 1), (1, 2)],
        "tfidf__use_idf": (True, False),
        "classifier__alpha": (1e-2, 1e-3),
    }

    grid = GridSearchCV(pipeline, cv=10, param_grid=parameters, verbose=1)
    grid.fit(X_train, y_train)

    return grid


def predict_and_save(model, df):
    """Predict labels and save results to CSV."""
    df["predicted_label"] = model.predict(df["text"])
    df.to_csv("predicted_model.csv", header=True, index=False, encoding="utf-8")


def main():
    # Load and preprocess data
    pr = Preprocessing()
    tokopedia_reviews = load_data("product_reviews_dirty.csv")
    tokopedia_reviews = clean_data(tokopedia_reviews, pr)
    tokopedia_reviews = label_data(tokopedia_reviews)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        tokopedia_reviews["text"], tokopedia_reviews["label"], train_size=0.75
    )

    # Train model
    model = train_model(X_train, y_train)
    joblib.dump(model, "model.pkl")

    # Load model and predict
    model = joblib.load("model.pkl")
    predict_and_save(model, tokopedia_reviews)

    # Evaluate model performance
    print("Confusion Matrix:\n", confusion_matrix(y_test, model.predict(X_test)))
    print("DONE!")


if __name__ == "__main__":
    main()
