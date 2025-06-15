import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from mlflow.models.signature import infer_signature
from preprocess import preprocess

def train():
    X_train, X_test, y_train, y_test = preprocess()

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", acc)

        # Signature & input example
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(1)

        # Log model with metadata
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example
        )

if __name__ == "__main__":
    train()
