import os
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    args, _ = parser.parse_known_args()

    # Load Preprocessed dataset
    train_data = pd.read_csv(os.path.join(args.train, "train.csv"), header=None)
    test_data = pd.read_csv(os.path.join(args.test, "test.csv"), header=None)

    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # # train model
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)

    model.fit(X_train,y_train)

    # # evalute model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    # write structured classification metrics
    report_dict = {
        "classification_metrics": {
            "accuracy": {"value": float(accuracy)},
            "f1_score": {"value": float(f1)}
        }
    }

    with open(os.path.join(args.output_data_dir, "evaluation.json"), "w") as f:
        json.dump(report_dict, f)

    print (f"Test evaluation accuracy : {accuracy}")

    # # save model artifact
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    print("model artifact exported safely")


