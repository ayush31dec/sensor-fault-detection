import numpy as np
import sklearn
from xgboost import XGBClassifier

class TargetValueMapping:
    def __init__(self):
        self.neg: int = 0
        self.pos: int = 1

    def to_dict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


#Write a code to train model and check the accuracy.

class SensorModel:

    def __init__(self, train_np: np.array, test_np: np.array) -> None:
        self.train_np_fe = train_np[:,:-1]
        self.train_np_t = train_np[:,-1]

        self.test_np_fe = test_np[:,:-1]
        self.test_np_t = test_np[:,-1]

        self.model = XGBClassifier()

    def check_accuracy(self, actual, pred) -> float:
        score = sklearn.metrics.accuracy_score(actual, pred)
        return score

    def model_check(self):
        self.model.fit(self.train_np_fe, self.train_np_t)
        train_pred = self.model.predict(self.train_np_fe)
        test_pred = self.model.predict(self.test_np_fe)

        train_score = self.check_accuracy(self.train_np_t, train_pred)
        test_score = self.check_accuracy(self.test_np_t, test_pred)

        return train_score, test_score

    def get_best_model(self):
        
        tr_sc, test_sc = self.model_check()
        print(tr_sc, test_sc)