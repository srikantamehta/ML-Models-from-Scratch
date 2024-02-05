from src.data_preprocessor import DataProcessor
from src.cross_validation import CrossValidation
from data_configs.configs import albalone_config, breast_cancer_config, car_config, forest_fires_config, house_votes_84_config

data_processor = DataProcessor(config=house_votes_84_config)
cross_validator = CrossValidation(config=house_votes_84_config)

raw_data = data_processor.load_data()

print(raw_data)

data_1 = data_processor.impute_missing_values(raw_data)

print(data_1)

data_2 = data_processor.encode_nominal_features(data_1)

print(data_2)

data_3 = data_processor.encode_ordinal_features(data_2)

print(data_3)