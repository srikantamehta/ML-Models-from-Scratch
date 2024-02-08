from src.data_preprocessor import DataProcessor
from src.cross_validation import CrossValidation
from src.evaluation import Evaluation
from models.knn import KNN
from models.null_model import NullModelClassification, NullModelRegression
from data_configs.configs import *


config = albalone_config

data_processor = DataProcessor(config=config)
cross_validator = CrossValidation(config=config)
classification_nullmodel = NullModelClassification(config=config)
regression_nullmodel = NullModelRegression(config=config)


raw_data = data_processor.load_data()

# print(raw_data)

data_1 = data_processor.impute_missing_values(raw_data)

# print(data_1)

data_2 = data_processor.encode_nominal_features(data_1)

# print(data_2)

data_3 = data_processor.encode_ordinal_features(data_2)

data_train, data_test = cross_validator.random_partition(data_3)

knn_model = KNN(config, data_test, data_train)

data_test_drop = data_test.drop(config['target_column'], axis=1)

distances = knn_model.k_nearest_neighbors(data_test_drop.iloc[1], 2)

print(distances)





