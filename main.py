from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
import os,sys
from sensor.logger import logging
from sensor.pipeline import training_pipeline
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.pipeline.prediction_pipeline import Prediction
import os
from sensor.utils.main_util import read_yaml_file
from fastapi import FastAPI, UploadFile, File
from sensor.constant.application import APP_HOST, APP_PORT
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response, FileResponse
from sensor.ml.model.estimator import ModelResolver,TargetValueMapping
from sensor.utils.main_util import load_object
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from sensor.data_access.sensor_data import SensorData

env_file_path=os.path.join(os.getcwd(),"env.yaml")

def set_env_variable(env_file_path):

    if os.getenv('MONGO_DB_URL',None) is None:
        env_config = read_yaml_file(env_file_path)
        os.environ['MONGO_DB_URL']=env_config['MONGO_DB_URL']

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:

        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/predict")
async def predict_route(csv_file: UploadFile=File(...)):
    try:
        #get data from user csv file
        #convert csv file to dataframe
        df=None
        df = pd.read_csv(csv_file.file, na_values='na')
        
        prediction_pl = Prediction(df)
        pred_df = prediction_pl.initiate_prediction()

        file_path = os.path.join('output', 'output.csv')
        if not os.path.exists('output'):
            os.makedirs('output')

        elif os.path.isfile(os.path.join('output', 'output.csv')):
            os.remove(file_path)

        pred_df.to_csv(file_path)
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type='text/csv')
        else:
            return 'No file'
    except Exception as e:
        raise Response(f"Error Occured! {e}")


if __name__=="__main__":
    
    try:
        
        set_env_variable(env_file_path)
        app_run(app, host=APP_HOST, port=APP_PORT)
    except Exception as e:
        SensorException(e,sys)
