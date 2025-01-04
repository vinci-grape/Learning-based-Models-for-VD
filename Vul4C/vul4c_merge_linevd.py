
import pandas as pd


final_data_set = pd.DataFrame()

for split in ['train','valid','test']:
    data = pd.read_json(f'../vul4c_dataset/{split}_linevd.json')
    data['label'] = split if split != 'valid' else 'val'
    data['split_id'] = data['id']
    final_data_set = pd.concat([final_data_set,data])

final_data_set.reset_index(inplace=True,drop=True)
final_data_set.reset_index(inplace=True)
final_data_set['id'] = final_data_set['index']
final_data_set.drop('index', axis=1, inplace=True)

print(final_data_set)
print(final_data_set.keys())
print(len(final_data_set))

final_data_set.to_json('../vul4c_dataset/vul4c_dataset_linevd_final.json')