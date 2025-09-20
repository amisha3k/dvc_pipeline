import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
import mlflow
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
import os 
from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URL']="https://dagshub.com/amisha3k/dvc_pipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="amisha3k"
os.environ['MLFLOW_TRACKING_PASSWORD']="e454077689b7938f4a4a5f3fc65e4a051bd61360"

#load parameters from params.yaml
params=yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns=['Outcome'])
    y=data["Outcome"]


    mlflow.set_tracking_uri("https://dagshub.com/amisha3k/dvc_pipeline.mlflow")


    #load model from disk
    model=pickle.load(open(model_path,'rb'))

    predictions=model.predict(X)
    accuracy=accuracy_score(y,predictions)

    #log metrics to mlflow
    mlflow.log_metric("accuracy",accuracy)
    print("Model accuracy:{accuracy}")

if __name__=="__main__":
    evaluate(params["data"],params["model"])
