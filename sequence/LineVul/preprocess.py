from pathlib import Path
import pandas as pd

src_dir = Path(__file__).parent.parent.parent / "vul4c_dataset"
dst_dir = Path(__file__).parent / "storage/dataset"
dst_dir.mkdir(parents=True,exist_ok=True)

splits_name = ['train','test','valid']

for split in splits_name:
    def strip_func(f:str):
        f = f.split('\n')
        f = [ line.strip() for line in f ]
        return '\n'.join(f)


    split_path = src_dir / f"{split}.json"
    save_path = dst_dir / f"{split}.json"

    data = pd.read_json(split_path)
    data['func'] = data['func'].map(lambda x:strip_func(x))

    if split == 'train':
        vul_cnt = len(data[data['vul'] == 1])
        non_vul = data[data['vul'] == 0].sample(n=vul_cnt , random_state=3333)
        data = data[data['vul'] == 1]
        data = pd.concat([data , non_vul]).sample(frac=1 , random_state=2023)

    data.to_json(save_path)
