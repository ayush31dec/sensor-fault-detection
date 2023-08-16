from sensor.entity.artifact_entity import ClassificationMetricArtifact
from sensor.exception import SensorException
from sklearn.metrics import f1_score,precision_score,recall_score
import os,sys

def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    try:
        model_f1_score = float(f1_score(y_true, y_pred))
        model_recall_score = float(recall_score(y_true, y_pred))
        model_precision_score=float(precision_score(y_true,y_pred))

        # classsification_metric =  
        return ClassificationMetricArtifact(f1_score=model_f1_score,
                    precision_score=model_precision_score, 
                    recall_score=model_recall_score)
    except Exception as e:
        raise SensorException(e,sys)