from src.data_loader import load_data
from data_configs.albalone import albalone_config 

albalone_data = load_data(config=albalone_config)

print(albalone_data)