import json
from glob import glob
from pathlib import Path

import pandas as pd
from utils.utils import processed_dir, dfmp
from functools import partial
class BigVulDataset:
    """Represent BigVul as graph dataset."""
    DATASET = None

    def __init__(self, df, dataset, partition="train", vulonly=False, sample=-1, splits="default", not_balance=False):
        """Init class."""
        # Get finished samples
        self.finished = [
            int(Path(i).name.split(".")[0])
            for i in glob(str(f"{Path(__file__).parent.parent.parent}/{dataset}/{partition}/*nodes*"))
        ]
        BigVulDataset.DATASET = dataset

        self.df = df
        self.mix = None
        self.partition = partition
        self.df = self.df[self.df.id.isin(self.finished)]

        # filter large functions
        print(f'{partition} before large:', len(self.df))
        self.df = self.df[self.df['func'].map(lambda x: len(x.splitlines())) <= 500]
        print(f'{partition} after large:', len(self.df))

        # Balance training set
        if partition == "train" and not not_balance:
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0].sample(len(vul), random_state=0)
            self.df = pd.concat([vul, nonvul])

        # Correct ratio for test set
        # if partition == "test" or partition == "val":
        # vul = self.df[self.df.vul == 1]
        # nonvul = self.df[self.df.vul == 0]
        # nonvul = nonvul.sample(min(len(nonvul), len(vul) * 20), random_state=0)
        # self.df = pd.concat([vul, nonvul])

        # Small sample (for debugging):
        if sample > 0:
            self.df = self.df.sample(sample, random_state=0)

        # Filter only vulnerable
        if vulonly:
            self.df = self.df[self.df.vul == 1]
        # Mix patches
        if dataset == 'bigvul_mix' and partition == 'train':
            self.df = pd.concat([self.df, self.mix])

        # Filter out samples with no lineNumber from Joern output
        self.df["valid"] = dfmp(
            self.df, partial(BigVulDataset.check_validity,partition = partition), "id", desc="Validate Samples: ", ordr=True, workers=32,
        )
        print(f'{partition} before valid:', len(self.df))
        self.df = self.df[self.df.valid]
        print(f'{partition} after valid:', len(self.df))


    def itempath(_id,partition:str):
        """Get itempath path from item id."""
        return Path(__file__).parent.parent.parent / f"{BigVulDataset.DATASET}/{partition}/{_id}"

    def check_validity(_id , partition:str):
        """Check whether sample with id=_id has node/edges.

        Example:
        _id = 1320
        with open(str(svd.processed_dir() / f"bigvul/before/{_id}.c") + ".nodes.json", "r") as f:
            nodes = json.load(f)
        """
        valid = 0
        try:
            with open(str(BigVulDataset.itempath(_id,partition)) + ".nodes.json", "r") as f:
                nodes = json.load(f)
                lineNums = set()
                for n in nodes:
                    if "lineNumber" in n.keys():
                        lineNums.add(n["lineNumber"])
                        if len(lineNums) > 1:
                            valid = 1
                            break
                if valid == 0:
                    return False
            with open(str(BigVulDataset.itempath(_id,partition)) + ".edges.json", "r") as f:
                edges = json.load(f)
                if type(edges[0]) == list:
                    edge_set = set([i[2] for i in edges])
                else:
                   edge_set = set([i['etype'] for i in edges])
                if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                    return False
                return True
        except Exception as E:
            print(E, str(BigVulDataset.itempath(_id,partition)))
            return False


    def get_vul_label(self, _id):
        """Obtain vulnerable or not."""
        df = self.df[self.df.id == _id]
        label = df.vul.item()
        return label

    def __getitem__(self, idx):
        """Must override."""
        return self.df.iloc[idx].to_dict()

    def __len__(self):
        """Get length of dataset."""
        return len(self.df)

    def __repr__(self):
        """Override representation."""
        vulnperc = round(len(self.df[self.df.vul == 1]) / len(self), 3)
        return f"BigVulDataset(partition={self.partition}, samples={len(self)}, vulnperc={vulnperc})"


print(__file__)