import pandas as pd

class DataLoader:
    def __init__(self, config):
        self.file_path = config['file_path']
        self.separator = config['separator']
        self.column_names = config['column_names']
        # Other config parameters can be initialized here if needed

    def load_data(self):
        # Load the data with the specified column names
        data = pd.read_csv(self.file_path, sep=self.separator, names=self.column_names, header=None)
        return data
