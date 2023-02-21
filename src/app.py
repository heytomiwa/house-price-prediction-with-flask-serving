import config
import flask
import pandas as pd
import joblib
import time
from flask import Flask
from flask import request

app = Flask(__name__)

# MODEL = None
# IMPUTER = None
# PREPPROCESSOR = None
MODEL = joblib.load(config.MODEL_PATH)
IMPUTER = joblib.load(config.IMPUTER_PATH)
PREPPROCESSOR = joblib.load(config.PREPROCESSOR_PATH)

@app.route("/")
def home():
    return "<h3>Sklearn Prediction Container"

def imputer(payload):
    replace_with_none = ["MiscFeature", "Fence", "GarageCond", "GarageQual", "GarageFinish", "GarageType", "FireplaceQu", "BsmtFinType2", "BsmtFinType1", "BsmtExposure", "BsmtCond","BsmtQual", "MasVnrType", "Alley", "PoolQC"]
    replace_with_0 = ["MasVnrArea", "GarageYrBlt"]
    payload.loc[:, replace_with_none] = payload[replace_with_none].fillna("None")
    payload.loc[:, replace_with_0] = payload[replace_with_0].fillna(0)
    payload[config.x_features] = IMPUTER.transform(payload)
    return payload

def preprocessor(payload):
    return PREPPROCESSOR.transform(payload)

def saleprice_prediction(payload):
    inference_payload = pd.DataFrame(payload)
    imputed_payload = imputer(inference_payload)
    # numerical_features = [feature for feature in inference_payload.columns if inference_payload[feature].dtype!="0"]
    # categorical_features = [feature for feature in inference_payload.columns if inference_payload[feature].dtype=="0"]
    # features = numerical_features+categorical_features
    preprocessed_payload = preprocessor(imputed_payload)
    prediction = list(MODEL.predict(preprocessed_payload))
    return prediction

@app.route("/predict", methods=["POST"])
def predict():
    prediction = saleprice_prediction(request.json)
    return flask.jsonify({"prediction": prediction})

# if __name__ == "__main__":
#     MODEL = joblib.load(config.MODEL_PATH)
#     IMPUTER = joblib.load(config.IMPUTER_PATH)
#     PREPPROCESSOR = joblib.load(config.PREPROCESSOR_PATH)

#     app.run(host="0.0.0.0")