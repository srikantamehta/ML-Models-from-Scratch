from src.data_loader import DataLoader

loader = DataLoader('datasets/abalone.data')

data = loader.load_data()

print(data)