import subprocess
import yaml
from Main import main
path_to_yaml = 'cfg.yaml'


class Infomation():
    def __init__(self,config):
        self.model = config['model']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.test_batch_size = config['test_batch_size']
        self.epochs = config['epochs']
        self.data_path = config['data_path']
    
    def __str__(self) -> str:
        return self.model

    
try: 
    with open (path_to_yaml, 'r') as file:
        config = yaml.safe_load(file)
        print(config)
        info = Infomation(config=config)
        print(info.model)

except Exception as e:
    print('Error reading the config file\n',e)


main(info)
