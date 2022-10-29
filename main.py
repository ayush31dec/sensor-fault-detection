from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.pipeline.training_pipeline import TrainPipeline


if __name__=='__main__':
    mongodb_client = MongoDBClient()
    print(mongodb_client.database.list_collection_names())
    #print(MONGODB_URL_KEY)
    training_pipeline = TrainPipeline()
    training_pipeline.run_pipeline()