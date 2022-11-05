from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.utils.main_util import read_yaml_file
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH
import os


if __name__=='__main__':
    #mongodb_client = MongoDBClient()
    #print(mongodb_client.database.list_collection_names())
    #print(MONGODB_URL_KEY)
    training_pipeline = TrainPipeline()
    training_pipeline.run_pipeline()
    #print("Hello World")
