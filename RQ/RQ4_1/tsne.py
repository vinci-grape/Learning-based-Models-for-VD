import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import torch
import pickle

def cal_center(items):
    x = np.mean(items[:,0])
    y = np.mean(items[:,1])
    return x, y

def ed(m, n):
 return np.sqrt(np.sum((m - n) ** 2))

def tSNE_visualization(model_name: str, input, label):
    assert len(input) == len(label)

    t_sne = TSNE(n_components=2, init='pca', random_state=0)
    x = t_sne.fit_transform(input)
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    X = (x - x_min) / (x_max - x_min)

    plt.figure(model_name)

    plt.scatter(X[label == 0, 0], X[label == 0, 1], label="non-vul", marker='o', color='green', facecolors='none',
                alpha=0.2, s=15)
    plt.scatter(X[label == 1, 0], X[label == 1, 1], label="vul", marker='+', color='red')

    vul_center = cal_center(X[label == 1])
    non_vul_center = cal_center(X[label == 0])
    print(ed(np.array([vul_center[0], vul_center[1]]), np.array([non_vul_center[0], non_vul_center[1]])))
    print(vul_center,non_vul_center)

    # for i in range(x.shape[0]):
    #     if label[i] == 0:
    #         print('hello')
    #         plt.text(x[i, 0], x[i, 1], 'o',
    #                  fontdict={'weight': 'bold', 'size': 9})
    #     else:
    #         plt.text(x[i, 0], x[i, 1], '+',
    #                  color=plt.cm.Set1(0),
    #                  fontdict={'weight': 'bold', 'size': 9})

    # plt.title(f"{model_name} tSNE visualization")
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig(f'./result/svg/{model_name}.svg',bbox_inches='tight',dpi=400,transparent=True)
    plt.savefig(f'./result/pdf/{model_name}.pdf',bbox_inches='tight',dpi=400,transparent=True)
    plt.savefig(f'./result/png/{model_name}.png',bbox_inches='tight',dpi=400,transparent=True)
    plt.show()
    return plt


np.random.seed(2013)

def tSNE_preprocess(model_name: str, data_file: str):
    data = pickle.load(open(data_file, mode='rb'))
    indices = list(range(len(data)))
    indices = np.random.choice(indices, size=10000, replace=False)
    data = np.array(data,dtype=object)
    data = data[indices]
    intput = np.array([x[0] for x in data])
    labels = np.array([x[1] for x in data])
    print('data loaded!')
    return tSNE_visualization(model_name, intput, labels)


data = [
    ('LineVul', '../../sequence/LineVul/storage/test_tSNE_embedding.pkl'),
    ('SVulD', '../../sequence/SVulD/storage/test_tSNE_embedding.pkl'),
    ('Reveal', '../../Graph/storage/results/reveal/vul4c_dataset/test_tSNE_embedding.pkl'),
    ('IVdetect', '../../Graph/storage/results/ivdetect/vul4c_dataset/test_tSNE_embedding.pkl'),
    ('Devign', '../../Graph/storage/results/devign/vul4c_dataset/test_tSNE_embedding.pkl')
]

for d in data:
    tSNE_preprocess(d[0],d[1])

