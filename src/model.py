import numpy as np
import pandas as pd
import joblib

from sklearn import preprocessing
from sklearn import ensemble
from sklearn import impute

from sklearn.compose import ColumnTransformer

if __name__ == "__main__":
    df = pd.read_csv("./input/train.csv").drop("Id", axis=1)
    y = df.SalePrice
    df = df.drop("SalePrice", axis=1)
    categorical_features = [
        feature for feature in df.columns if df[feature].dtype=="O"
    ]
    numerical_features = [
        feature for feature in df.columns if df[feature].dtype!="O" and feature!="SalePrice"
    ]
    x_features = numerical_features+categorical_features
    
    # Cleaning the data
    replace_with_none = ["MiscFeature", "Fence", "GarageCond", "GarageQual", "GarageFinish", "GarageType", "FireplaceQu", "BsmtFinType2", "BsmtFinType1", "BsmtExposure", "BsmtCond","BsmtQual", "MasVnrType", "Alley", "PoolQC"]
    replace_with_0 = ["MasVnrArea", "GarageYrBlt"]
    df.loc[:, replace_with_none] = df[replace_with_none].fillna("None")
    df.loc[:, replace_with_0] = df[replace_with_0].fillna(0)
    imputer = ColumnTransformer([
        ("num_imp", impute.SimpleImputer(strategy="median"), numerical_features),
        ("cat_imp", impute.SimpleImputer(strategy="most_frequent"), categorical_features),
    ])
    imputer.fit(df[x_features])
    joblib.dump(imputer, "./pipelines/imputer.joblib")
    df[x_features] = imputer.transform(df)

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numerical_features),
        ("cat", preprocessing.OrdinalEncoder(), categorical_features),
    ])
    preprocessor.fit(df)
    joblib.dump(preprocessor, "./pipelines/preprocessor.joblib")
    X = preprocessor.transform(df)

    model = ensemble.GradientBoostingRegressor(alpha=0.1, max_depth=7, subsample=0.7)
    model.fit(X, y)
    joblib.dump(model, "./models/gradient_boosting_regressor.joblib")
