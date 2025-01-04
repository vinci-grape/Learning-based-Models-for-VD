
import json
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pickle
import re

all_gpus = [0,1,2]
need_run = [True] * len(all_gpus)

if __name__ == '__main__':
    split_cnt = len(all_gpus)
    start_time = datetime.now()
    all_p = []
    save_dir = Path(f"../../storage/results/devign/vul4c_dataset")

    cache_files = [i for i in os.listdir(save_dir) if re.match(r'test_interpret_[0-9]+.pkl',i)]

    find_max_split_cnt = -2
    for cache_f in cache_files:
        cache_split_idx = cache_f[len("test_interpret_"):]
        cache_split_idx = int(cache_split_idx.split('.')[0])
        find_max_split_cnt = max(find_max_split_cnt,cache_split_idx)
        need_run[cache_split_idx] = False


    if find_max_split_cnt + 1 == split_cnt:
        print(f'using cache, {cache_files}')
    else:
        print('cache not hit')
        need_run = [True] * len(all_gpus)


    for idx,(gpu,run) in enumerate(zip(all_gpus,need_run)):
        if run:
            new_env = os.environ.copy()
            new_env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            print(f'begin {idx} GPU:{gpu}')
            p = subprocess.Popen(f'python main.py --dataset=vul4c_dataset --interpret_split_idx={idx} --interpret_total_split={split_cnt}', shell=True,
                                 env=new_env,stdout=subprocess.DEVNULL)
            all_p.append(p)

    [x.wait() for x in all_p ]

    # merge result
    print('begin merge result')
    all_data = []
    for idx in range(len(all_gpus)):
        save_file_name = save_dir / f"test_interpret_{idx}.pkl"
        all_data.extend(pickle.load(save_file_name.open(mode='rb')))
        # os.remove(save_file_name)

    end_time = datetime.now()
    print('merge result done!!!!!')
    print(f'Time:{(end_time - start_time).seconds}s Total:{len(all_data)}')

    pickle.dump(all_data,(save_dir / "test_interpret_merge.pkl" ).open(mode='wb'))

