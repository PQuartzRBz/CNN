import subprocess
import yaml

path_to_yaml = 'cfg.yaml'
try: 
    with open (path_to_yaml, 'r') as file:
        config = yaml.safe_load(file)
        subprocess.run('python Model.py --batch-size=64 --test-batch-size=64 --epochs=10 --lr=0.001 --data-path="dataset\model1"',shell=True)
except Exception as e:
    print('Error reading the config file\n',e)

