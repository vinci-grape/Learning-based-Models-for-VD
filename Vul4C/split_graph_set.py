import json
import sys

import numpy
from pathlib import Path
import shutil
import os

if Path('processed/split_graph').exists():
    print('already split, manual remove directory needed!!')
    sys.exit(0)


base_dir = Path("processed/graph")
split_FOLD = 3
split_result = [[] for i in range(split_FOLD)]
print(split_result)
cnt = 0
for repo in base_dir.iterdir():
    for commit in repo.iterdir():
        FOLD = cnt % split_FOLD
        split_result[FOLD].append(commit)
        cnt+=1

print(cnt)
print(len(split_result))

for fold in range(split_FOLD):
    fold_path = Path(f'processed/split_graph/split_{fold}')
    fold_path.mkdir(parents=True,exist_ok=True)
    print(f'copy split fold [{fold}]')
    for commit in split_result[fold]:
        cp_src = str(commit)
        cp_dst = fold_path / '/'.join(str(commit).split('/')[-2:])
        shutil.copytree(cp_src,cp_dst)
    os.system(f"zip -r {str(fold_path.parent)}/split{fold}.zip {str(fold_path)}")

print('*'*50)
print('Split Done!')
for fold in range(split_FOLD):
    print(f'Split[{fold}]:{len(split_result[fold])}')
