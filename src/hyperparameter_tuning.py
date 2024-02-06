# from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

# class CrossValidation:

#     def __init__(self, config):
#         self.config = config

#     def random_partition(self, data, val_size=0.2, random_state=None):
#         data = data.copy()
#         train_data, val_data = train_test_split(data, test_size=val_size, random_state=random_state)
#         return train_data, val_data
    
#     def cross_validation(self, data, n_splits=10, n_repeats=1, random_state=None):
#         data = data.copy()
#         X = data.drop(columns=[self.config['target_column']])
#         y = data[self.config['target_column']]

#         rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
#         for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
#             yield data.iloc[train_index], data.iloc[test_index]

# # Define your dataset and configuration
# # data = ...  # Your DataFrame
# # my_config = ...  # Your configuration dictionary

# # Initialize CrossValidation instance with your configuration
# cross_validator = CrossValidation(config=my_config)

# # Step 1: Partition your data into training (80%) and validation (20%)
# data_train_80, data_val_20 = cross_validator.random_partition(data, val_size=0.2, random_state=42)

# # Define your hyperparameters (this is just an example)
# all_hyperparameters = [[2], [3]]  # or a more complex structure if you have multiple hyperparameters

# # Store the results for different hyperparameters
# hyperparameter_results = []

# # Perform cross-validation for each set of hyperparameters
# for hyperparams in all_hyperparameters:
#     results = []
    
#     # Step 2: Perform 5x2 cross-validation
#     for data_fold_train, data_fold_test in cross_validator.cross_validation(data_train_80, n_splits=2, n_repeats=5, random_state=42):
        
#         # Step 2b: Hyperparameter tuning and model training/testing
#         # Train on the train fold, test on the validation set (data_val_20)
#         model = train_model(data_fold_train, hyperparams)
#         result = test_model(model, data_val_20)
        
#         # Collect and store results
#         results.append(result)
    
#     # Calculate and store the average performance for these hyperparameters
#     average_performance = calculate_average_performance(results)
#     hyperparameter_results.append((hyperparams, average_performance))

# # Step 3: Determine the best hyperparameters based on the cross-validation results
# best_hyperparameters, best_performance = pick_best_hyperparameters(hyperparameter_results)

# # Print the best hyperparameters and their performance
# print(f"Best Hyperparameters: {best_hyperparameters}")
# print(f"Best Performance: {best_performance}")

# # Steps 4 & 5: Train the model on the entire 80% data using best_hyperparameters and test on the held-out 20% data.
# # ... (similar structure to Step 2)

# # You may need to define or import train_model, test_model, calculate_average_performance, and pick_best_hyperparameters functions/methods as per your specific implementation requirements.
