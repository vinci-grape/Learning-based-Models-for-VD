from time import  time
import torchinfo
from pathlib import Path

class RunTimeCounter:
    def __init__(self):
        self.start_time = time()

    def reset(self):
        self.start_time = time()

    def stop(self,info:str):
        total_time = time() - self.start_time
        time_info = f'[Time]:{info} {round(total_time,2)}s'
        with Path('./storage/time_info.log').open('a') as file:
            file.write(time_info + '\n')
        print(time_info)
        self.start_time = time()

class ModelParameterCounter:
    def summary(self , model , model_name:str):
        summary_info = str(torchinfo.summary(model))
        with Path('./storage/model_info.log').open('a') as file:
            file.write(f'=============== {model_name} ===============')
            file.write(summary_info + '\n')
            file.write('\n' * 5)
        print(summary_info)


