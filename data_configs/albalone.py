# abalone_config.py

albalone_config = {
    'name': 'Abalone',
    'file_path': 'datasets/abalone.data',
    'separator': ',',  
    'column_names': [
        'Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
        'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'
    ],
    'target_column': 'Rings',
    'missing_values': None,  # Specific representation if any, like '?'
    'nominal_features': ['Sex'],  # Nominal (unordered categorical) features
    'ordinal_features': [],  # If any, like ['Size']
    'numeric_features': [
        'Length', 'Diameter', 'Height', 'Whole weight',
        'Shucked weight', 'Viscera weight', 'Shell weight'
    ],
    'features_to_encode': ['Sex'],  # For one-hot encoding
    'features_to_discretize': [],  # If any, like ['Age']
    'z_score_standardization': True,  # Whether to apply z-score standardization
    'cross_validation': {
        'method': '5x2',  # or 'k-fold'
        'k': 10,  # if using k-fold
    }
}

