import os
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    # load raw data
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    scaler = StandardScaler()
    

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_trf = scaler.fit_transform(X_train)
    X_test_trf = scaler.transform(X_test)
    # train data after scaling
    train_df = pd.DataFrame(X_train_trf, columns=iris.feature_names)
    train_df['target'] = y_train.values
    # test data after scaling
    test_df = pd.DataFrame(X_test_trf, columns=iris.feature_names)
    test_df['target'] = y_test.values    
    # print (y_train.values)
    os.makedirs("/opt/ml/processing/train", exist_ok=True)
    os.makedirs("/opt/ml/processing/test", exist_ok=True)

    train_df.to_csv("/opt/ml/processing/train/train.csv", index=False, header=False)
    test_df.to_csv("/opt/ml/processing/test/test.csv", index=False, header=False)

    print("Preprocessing completed, split dataset written into outputs")

    # ml.c4.xlarge -- > test step pipeline
