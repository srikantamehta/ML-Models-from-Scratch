from src.data_preprocessor import DataProcessor
from src.cross_validation import CrossValidation
from src.evaluation import Evaluation
from models.knn import KNN
from models.null_model import NullModelClassification, NullModelRegression
from data_configs.configs import *


config = breast_cancer_config

data_processor = DataProcessor(config=config)
cross_validator = CrossValidation(config=config)
classification_nullmodel = NullModelClassification(config=config)
regression_nullmodel = NullModelRegression(config=config)
knn_model = KNN(config)

raw_data = data_processor.load_data()

raw_data_2 = raw_data.drop(columns='Sample code number')

data_1 = data_processor.impute_missing_values(raw_data_2)

data_2 = data_processor.encode_nominal_features(data_1)

data_3 = data_processor.encode_ordinal_features(data_2)

data_train, data_val = cross_validator.random_partition(data_3, random_state=42)

results = knn_model.knn_classifier(data_val,data_train, 2)

print(results)

score = Evaluation().zero_one_loss(results['Class'],results['Predicted Class'])

print(score)