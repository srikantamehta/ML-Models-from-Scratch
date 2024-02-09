from src.data_preprocessor import DataProcessor
from src.cross_validation import CrossValidation
from src.evaluation import Evaluation
from models.knn import KNN
from models.null_model import NullModelClassification, NullModelRegression
from data_configs.configs import *

# Import any additional functions or modules needed for hyperparameter tuning

# Define your configuration
config = breast_cancer_config

# Initialize necessary instances
data_processor = DataProcessor(config=config)
cross_validator = CrossValidation(config=config)
classification_nullmodel = NullModelClassification(config=config)
regression_nullmodel = NullModelRegression(config=config)

# Load and preprocess your raw data
raw_data = data_processor.load_data()
raw_data_2 = raw_data.drop(columns='Sample code number')
data_1 = data_processor.impute_missing_values(raw_data_2)
data_2 = data_processor.encode_nominal_features(data_1)
data_3 = data_processor.encode_ordinal_features(data_2)

# Define hyperparameters for hyperparameter tuning
all_hyperparameters = [[2], [3]]  # Example hyperparameters

# Perform hyperparameter tuning
best_hyperparameters, best_performance = cross_validator.hyperparameter_tuning(
    data=data_3,  # Use preprocessed data for tuning
    all_hyperparameters=all_hyperparameters,
    train_model=train_model,  # Define your train_model function
    test_model=test_model,    # Define your test_model function
    calculate_average_performance=calculate_average_performance,  # Define your calculate_average_performance function
    pick_best_hyperparameters=pick_best_hyperparameters,          # Define your pick_best_hyperparameters function
    val_size=0.2,     # Specify validation set size
    n_splits=5,       # Specify number of splits for cross-validation
    n_repeats=2,      # Specify number of repeats for cross-validation
    random_state=42   # Specify random seed for reproducibility
)

# Print the best hyperparameters and their performance
print(f"Best Hyperparameters: {best_hyperparameters}")
print(f"Best Performance: {best_performance}")

# Use the best hyperparameters to train and test your final model
final_model = train_model_with_best_hyperparameters(best_hyperparameters, data_3)
final_performance = test_model(final_model, test_data)
print(f"Final Model Performance: {final_performance}")
