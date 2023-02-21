import requests
import json
import pandas as pd

url = "http://127.0.0.1:5000/predict"

data = pd.read_csv("input/test.csv")
data = data.drop("Id", axis=1).head(1).to_dict()

input_data = json.dumps(data)

headers = {"Content-Type": "application/json"}
input = {"input": data}

resp = requests.post(url, input_data, headers=headers)
print(resp.text)
