from src.data_loader import DataLoader 
from data_configs.albalone import albalone_config 

loader = DataLoader(config=albalone_config)

albalone_data = loader.load_data()

print(albalone_data)