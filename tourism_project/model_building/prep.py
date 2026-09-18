
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project root = tourism_project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "tourism.csv"

df = pd.read_csv(RAW_PATH)

df.drop(columns=["CustomerID"], inplace=True)

target = "ProdTaken"

X = df.drop(columns=[target])
y = df[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation completed successfully.")
print("Xtrain:", Xtrain.shape)
print("Xtest:", Xtest.shape)
print("ytrain:", ytrain.shape)
print("ytest:", ytest.shape)
