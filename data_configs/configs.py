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
    'ordinal_features': {},  # If any, like ['Size']
    'numeric_features': [
        'Length', 'Diameter', 'Height', 'Whole weight',
        'Shucked weight', 'Viscera weight', 'Shell weight'
    ]
}


breast_cancer_config = {
    'name': 'Breast Cancer Wisconsin (Original)',
    'file_path': 'datasets/breast-cancer-wisconsin.data',
    'separator': ',',  
    'column_names': [
        'Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
        'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
        'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'
    ],
    'missing_values': '?',  # Represented by '?'
    'nominal_features': ['Sample code number', 'Bare Nuclei'],  
    'ordinal_features': {},  # If any, like ['Size']
    'numeric_features': [
        'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
        'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bland Chromatin',
        'Normal Nucleoli', 'Mitoses'
    ]
}

