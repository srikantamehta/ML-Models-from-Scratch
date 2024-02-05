from src.data_preprocessor import DataProcessor
from src.cross_validation import CrossValidation
from data_configs.configs import albalone_config, breast_cancer_config

data_processor = DataProcessor(config=breast_cancer_config)
cross_validator = CrossValidation(config=breast_cancer_config)

raw_data = data_processor.load_data()

data_1 = data_processor.impute_missing_values(raw_data)

data_2 = data_processor.encode_ordinal_features(data_1)

data_3 = data_processor.encode_nominal_features(data_2)

data_4 = data_processor.discretize_feature_equal_width(data_3,['Clump Thickness'],2)

print(data_4)
