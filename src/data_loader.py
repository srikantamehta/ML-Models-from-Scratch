import pandas as pd

def load_data(config):
    """
    Load data from a file specified in the config.

    :param config: Configuration dictionary for the dataset.
    :return: Loaded DataFrame.
    """
    file_path = config['file_path']
    separator = config['separator']
    column_names = config['column_names']

    data = pd.read_csv(file_path, sep=separator, names=column_names, header=None)

    return data
