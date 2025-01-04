
from pathlib import Path
import pandas as pd

src_dir = Path(__file__).parent.parent.parent / "vul4c_unexecuted_code_dataset"
dst_dir = src_dir
dst_dir.mkdir(parents=True,exist_ok=True)

splits_name = ['train','test','valid']

for split in splits_name:
    def strip_func(f:str):
        f = f.split('\n')
        f = [ line.strip() for line in f ]
        return '\n'.join(f)


    split_path = src_dir / f"{split}.json"
    save_path = dst_dir / f"{split}_svuld.json"

    data = pd.read_json(split_path)
    data['func'] = data['func'].map(lambda x:strip_func(x))
    data['func_after'] = data['func_after'].map(lambda x:strip_func(x))

    new_data = pd.DataFrame()
    new_data['id'] = data['id']
    new_data['index'] = new_data['id']
    new_data['code'] = data['func']
    new_data['contrast'] = data['func_after']
    new_data['label'] = data['vul']


    if split == 'train':
        vul_cnt = len(new_data[new_data['label'] == 1])
        non_vul = new_data[new_data['label'] == 0].sample(n=vul_cnt , random_state=3333)
        new_data = new_data[new_data['label'] == 1]
        new_data = pd.concat([new_data , non_vul]).sample(frac=1 , random_state=2023)

    # print(new_data[  new_data['label'] == 1 ]['contrast'])
    new_data.reset_index(drop=True,inplace=True)
    new_data.to_json(save_path,orient='records')
