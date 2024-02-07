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

# print(data_3)

# data_train_80, data_val_20 = cross_validator.random_partition(data_3,0.2)

# i=0
# for data_fold_train, data_fold_test in cross_validator.cross_validation(data_train_80, n_splits=2, n_repeats=5, random_state=None,stratify=False):

#     # Extracting the target column for the train and test folds
#     y_train = data_fold_train[config['target_column']]
#     y_test = data_fold_test[config['target_column']]

#     # Print run number
#     print(f"Run {i}")
#     i += 1

#     # Print class distribution
#     print(f"Train fold class distribution:\n{y_train.value_counts(normalize=True)}")
#     print(f"Test fold class distribution:\n{y_test.value_counts(normalize=True)}\n")
# # classification_result = classification_nullmodel.naive_classifier(data_3)

# # print(classification_result)

# data_3['Predicted Class'] = classification_result

# # print(data_3)

# score = Evaluation.zero_one_loss(data_3[config['target_column']],data_3['Predicted Class'])

# print(score)
knn_model = KNN(data=data_3, config=config)
knn_model.knn_classifier(2)


