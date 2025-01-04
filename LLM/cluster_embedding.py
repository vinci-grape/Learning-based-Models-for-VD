from transformers import RobertaModel, RobertaTokenizer
import json
import pandas as pd
from tqdm import tqdm
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base").cuda()

train_set = pd.read_json('../vul4c_dataset/train.json')
test_set = pd.read_json('../vul4c_dataset/test.json')
vul_len = len(train_set[train_set['vul'] == 1])
train_set = train_set[(train_set['func'].str.split('\n').str.len() < 30) & (train_set['func'].str.split('\n').str.len() > 10)]
train_set = pd.concat([
    train_set[train_set['vul'] == 1] ,
    train_set[train_set['vul'] == 0].sample(n=vul_len,random_state=2023)
])

def get_set_embedding(data_set:pd.DataFrame):
    data_set = data_set.to_dict('records')
    embedding_result = []
    for i in tqdm(range(0, len(data_set), 24)):
        sub_set = data_set[i:i + 24]
        funcs = [item['func'] for item in sub_set]
        vuls = [item['vul'] for item in sub_set]
        ids = [item['id'] for item in sub_set]
        x = tokenizer(funcs, return_tensors='pt', padding=True, truncation=True).to('cuda')
        y = model(**x)
        y = y[0][:, 0, :]
        y = y.cpu().tolist()

        for j in range(len(sub_set)):
            embedding_result.append(
                {
                    'id': sub_set[j]['id'],
                    'func': funcs[j],
                    'vul': sub_set[j]['vul'],
                    'embedding': y[j],
                }
            )
    assert len(data_set) == len(embedding_result)
    return embedding_result

train_embedding_result = get_set_embedding(train_set)
test_embedding_result = get_set_embedding(test_set)

print(f'train_embedding_set = {len(train_embedding_result)}')
print(f'test_embedding_set = {len(test_embedding_result)}')
pd.DataFrame(train_embedding_result).to_pickle('../vul4c_dataset/train_knn_embedding.pkl')
pd.DataFrame(test_embedding_result).to_pickle('../vul4c_dataset/test_knn_embedding.pkl')
