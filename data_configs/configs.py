albalone_config = {
    'name': 'Abalone',
    'file_path': 'datasets/abalone.data',
    'separator': ',',  
    'column_names': [
        'Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
        'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'
    ],
    'has_header': False,
    'target_column': 'Rings',
    'missing_values': None,  
    'nominal_features': ['Sex'],  
    'ordinal_features': {},  
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
    'has_header': False,
    'target_column': 'Class',
    'missing_values': '?',  
    'nominal_features': ['Sample code number', 'Bare Nuclei'],  
    'ordinal_features': {},  
    'numeric_features': [
        'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
        'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bland Chromatin',
        'Normal Nucleoli', 'Mitoses'
    ]
}

car_config = {
    'name': 'Car Evaluation',
    'file_path': 'datasets/car.data',
    'separator': ',',  
    'column_names': [
        'buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Class'
    ],
    'has_header': False,
    'target_column': 'Class',
    'missing_values': None,
    'nominal_features': [],  
    'ordinal_features': {
        'buying': ['vhigh', 'high', 'med', 'low'],  
        'maint': ['vhigh', 'high', 'med', 'low'],  
        'doors': ['2', '3', '4', '5more'],
        'persons': ['2', '4', 'more'],
        'lug_boot': ['small', 'med', 'big'],
        'safety': ['low', 'med', 'high']
    },
    'numeric_features': []
}

forest_fires_config = {
    'name': 'Forest Fires',
    'file_path': 'datasets/forestfires.data',  
    'separator': ',', 
    'column_names': [
        'X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI',
        'temp', 'RH', 'wind', 'rain', 'area'
    ],
    'has_header': True,
    'target_column': 'area',
    'missing_values': None,  
    'nominal_features': [],  
    'ordinal_features': {
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'day': ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    },
    'numeric_features': [
        'X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI',
        'temp', 'RH', 'wind', 'rain'
    ]
}

house_votes_84_config = {
    'name': 'House Votes 84',
    'file_path': 'datasets/house-votes-84.data',  
    'separator': ',',  
    'column_names': [
        'Class Name', 'handicapped-infants', 'water-project-cost-sharing', 
        'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 
        'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 
        'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending', 
        'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa'
    ],
    'has_header': False,
    'target_column': 'Class Name',
    'missing_values': None,  
    'nominal_features': [
        'handicapped-infants', 'water-project-cost-sharing', 
        'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 
        'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 
        'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending', 
        'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa'
    ],
    'ordinal_features': {}, 
    'numeric_features': []
}

machine_config = {
    'name': 'Relative CPU Performance',
    'file_path': 'datasets/machine.data',
    'separator': ',', 
    'column_names': [
        'vendor_name', 'model_name', 'MYCT', 'MMIN', 'MMAX', 'CACH',
        'CHMIN', 'CHMAX', 'PRP', 'ERP'
    ],
    'has_header': False,
    'target_column': 'PRP',  
    'missing_values': None,  
    'nominal_features': ['vendor_name', 'model_name'],  
    'ordinal_features': {},  
    'numeric_features': ['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'ERP'],  
}
