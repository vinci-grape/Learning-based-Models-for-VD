import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent.parent.parent)))

from utils.utils import processed_dir, dfmp, get_dir, cache_dir, tokenize
from utils.joern import rdg, get_node_edges, drop_lone_nodes
from utils.dclass import BigVulDataset
import utils.glove as glove
import torch
import dgl
from pathlib import Path
import pickle
import re
import pandas as pd
import networkx as nx
import numpy as np
from functools import partial


def topsort(edges: list, node_num: int) -> bool:
    q = []
    in_degrees = [0] * node_num
    for e in edges:
        in_degrees[e[1]] += 1
    for i in range(node_num):
        if in_degrees[i] == 0:
            q.append(i)
    cnt = 0
    while len(q) != 0:
        cur = q.pop(0)
        cnt += 1
        for e in edges:
            if e[0] == cur:
                nxt = e[1]
                in_degrees[nxt] -= 1
                if in_degrees[nxt] == 0:
                    q.append(nxt)
    return cnt == node_num


class BigVulDatasetIVDetect(BigVulDataset):
    """IVDetect version of BigVul."""

    def __init__(self, need_code_lines=False, **kwargs):
        """Init."""
        super(BigVulDatasetIVDetect, self).__init__(**kwargs)
        # Load Glove vectors.
        glove_path = processed_dir() / f"{kwargs['dataset']}/glove_False/vectors.txt"
        self.emb_dict, _ = glove.glove_dict(glove_path)
        self.need_code_lines = need_code_lines

        # filter large functions
        print(f'{kwargs["partition"]} LOCAL before large:', len(self.df))
        ret = dfmp(
            self.df,
            partial(BigVulDatasetIVDetect._feat_ext_itempath, partition=self.partition),
            "id",
            ordr=True,
            desc="Cache features: ",
            workers=32
        )
        self.df = self.df[ret]
        print(f'{kwargs["partition"]} LOCAL after large:', len(self.df))

        # Get mapping from index to sample ID.
        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        print(self.df.columns, self.df.shape)
        self.idx2id = pd.Series(self.df.id.values, index=self.df.idx).to_dict()

    def item(self, _id, is_eval=False):
        """Get item data."""
        # print(f'{self.partition} item: {_id}')
        n, _ = feature_extraction(BigVulDataset.itempath(_id, self.partition))
        n.subseq = n.subseq.apply(lambda x: glove.get_embeddings(x, self.emb_dict, 200))
        n.nametypes = n.nametypes.apply(
            lambda x: glove.get_embeddings(x, self.emb_dict, 200)
        )
        n.data = n.data.apply(lambda x: glove.get_embeddings(x, self.emb_dict, 200))
        n.control = n.control = n.control.apply(lambda x: glove.get_embeddings(x, self.emb_dict, 200))

        asts = []

        def ast_dgl(row, lineid):
            if len(row) == 0:
                return None
            '''
            row example
            [[0, 0, 0, 0, 0, 0], 
             [1, 2, 3, 4, 5, 6], 
             ['int alloc addbyter int output FILE data', 'int output', 'FILE data', '', 'int', 'int output', 'FILE data']]

            '''
            outnode, innode, ndata = row
            edges = [(x, y) for x, y in zip(outnode, innode)]
            if not topsort(edges, len(ndata)):
                return None
            g = dgl.graph((outnode, innode))
            g.ndata["_FEAT"] = torch.Tensor(
                np.array(glove.get_embeddings_list(ndata, self.emb_dict, 200))
            )
            g.ndata["_ID"] = torch.Tensor([_id] * g.number_of_nodes())
            g.ndata["_LINE"] = torch.Tensor([lineid] * g.number_of_nodes())
            return g

        for row in n.itertuples():
            asts.append(ast_dgl(row.ast, row.id))

        return {"df": n, "asts": asts}

    def _feat_ext_itempath(_id, partition: str):
        """Run feature extraction with itempath."""
        n, e = feature_extraction(BigVulDataset.itempath(_id, partition))
        return 0 < len(n) <= 500

    def cache_features(self):
        """Save features to disk as cache."""
        dfmp(
            self.df,
            partial(BigVulDatasetIVDetect._feat_ext_itempath, partition=self.partition),
            "id",
            ordr=False,
            desc="Cache features: ",
            workers=32
        )

    def __getitem__(self, idx):
        """Override getitem."""
        _id = self.idx2id[idx]
        n, e = feature_extraction(BigVulDataset.itempath(_id, self.partition))

        # n["vuln"] = n.id.map(self.get_vuln_indices(_id)).fillna(0)
        g = dgl.graph(e)
        g.ndata["_LINE"] = torch.Tensor(n["id"].astype(int).to_numpy())  
        label = self.get_vul_label(_id)
        g.ndata["_LABEL"] = torch.Tensor([label] * len(n))
        g.ndata["_SAMPLE"] = torch.Tensor([_id] * len(n))  # SAMPLE ID
        g.ndata["_PAT"] = torch.Tensor([False] * len(n))

        # Add edges between each node and itself to preserve old node representations
        # print(g.number_of_nodes(), g.number_of_edges())
        g = dgl.add_self_loop(g)
        if self.need_code_lines:
            return g, n['code'].fillna("").tolist()
        else:
            return g


def feature_extraction(filepath):
    """Extract relevant components of IVDetect Code Representation.
    """
    cache_name = "_".join(str(filepath).split("/")[-3:])
    cachefp = get_dir(cache_dir() / f"ivdetect_feat_ext/{BigVulDataset.DATASET}") / Path(cache_name).stem
    try:
        with open(cachefp, "rb") as f:
            return pickle.load(f)
    except:
        pass

    try:
        nodes, edges = get_node_edges(filepath)
    except:
        return None
    # 1. Generate tokenised subtoken sequences
    subseq = (
        nodes.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)  
        .groupby("lineNumber")
        .head(1)
    )
    subseq = subseq[["lineNumber", "columnNumber", "code"]].copy()
    # subseq.code = subseq.local_type + " " + subseq.code
    # subseq = subseq.drop(columns="local_type")
    subseq = subseq[~subseq.eq("").any(1)]

    subseq = subseq.fillna("")
    subseq = subseq[subseq.code != " "]
    subseq.lineNumber = subseq.lineNumber.astype(int)
    # subseq.columnNumber = subseq.columnNumber.astype(int)
    subseq['raw_code'] = subseq['code']
    subseq = subseq.sort_values("lineNumber")
    # subseq.to_csv("subseq.csv")
    subseq.code = subseq.code.apply(tokenize)

    subseq = subseq.set_index("lineNumber").to_dict()["code"]
    # print("=================================")
    # print(subseq)

    # 2. Line to AST
    ast_edges = rdg(edges, "ast")

    ast_nodes = drop_lone_nodes(nodes, ast_edges)
    # print(f"type {ast_nodes.lineNumber}")
    # print(ast_nodes.info())
    ast_nodes = ast_nodes[ast_nodes['lineNumber'] != ""]
    ast_nodes.lineNumber = ast_nodes.lineNumber.astype(int)
    ast_nodes["lineidx"] = ast_nodes.groupby("lineNumber").cumcount().values  
    ast_edges = ast_edges[ast_edges.line_out == ast_edges.line_in]  

    ast_dict = pd.Series(ast_nodes.lineidx.values, index=ast_nodes.id).to_dict()  # ast nodeid2lineidx

    ast_edges.innode = ast_edges.innode.map(ast_dict)  
    ast_edges.outnode = ast_edges.outnode.map(ast_dict)
    ast_edges = ast_edges.groupby("line_in").agg({"innode": list, "outnode": list})  
    # print('-----')
    # print(ast_edges)
    # '''
    #                                               innode                               outnode
    # line_in
    # 1.0                           [1, 2, 3, 4, 5, 6]                    [0, 0, 0, 0, 0, 0]
    # '''
    ast_nodes.code = ast_nodes.code.fillna("").apply(tokenize)
    nodes_per_line = (
        ast_nodes.groupby("lineNumber").agg({"lineidx": list}).to_dict()["lineidx"]
    )

    ast_nodes = ast_nodes.groupby("lineNumber").agg({"code": list})
    # ast_edges.to_csv("ast_edges.csv")
    # ast_nodes.to_csv("ast_nodes.csv")
    # print(f'------------------------{ast_edges["line_in"].dtype()}-------------------------------')

    ast = ast_edges.join(ast_nodes, how="inner")

    if ast.empty:
        return [], []
    ast["ast"] = ast.apply(lambda x: [x.outnode, x.innode, x.code], axis=1)
    ast = ast.to_dict()["ast"]

    # If it is a lone node (nodeid doesn't appear in edges) or it is a node with no
    # incoming connections (parent node), then add an edge from that node to the node
    # with id = 0 (unless it is zero itself).
    # DEBUG:
    # import sastvd.helpers.graphs as svdgr
    # svdgr.simple_nx_plot(ast[20][0], ast[20][1], ast[20][2])
    for k, v in ast.items():
        allnodes = nodes_per_line[k]
        outnodes = v[0]
        innodes = v[1]
        lonenodes = [i for i in allnodes if i not in outnodes + innodes]
        parentnodes = [i for i in outnodes if i not in innodes]
        for n in set(lonenodes + parentnodes) - set([0]):
            outnodes.append(0)
            innodes.append(n)
        ast[k] = [outnodes, innodes, v[2]]

    # 3. Variable names and types
    reftype_edges = rdg(edges, "reftype")
    reftype_nodes = drop_lone_nodes(nodes, reftype_edges)
    reftype_nx = nx.Graph()
    reftype_nx.add_edges_from(reftype_edges[["innode", "outnode"]].to_numpy())
    reftype_cc = list(nx.connected_components(reftype_nx))

    varnametypes = list()
    # for cc in reftype_cc:
    #     cc_nodes = reftype_nodes[reftype_nodes.id.isin(cc)]
    #     if sum(cc_nodes["_label"] == "IDENTIFIER") == 0:
    #         continue
    #     if sum(cc_nodes["_label"] == "TYPE") == 0:
    #         continue
    #     var_type = cc_nodes[cc_nodes["_label"] == "TYPE"]
    #     print('varTYPE',var_type)
    for cc in reftype_cc:  
        cc_nodes = reftype_nodes[reftype_nodes.id.isin(cc)]
        if sum(cc_nodes["_label"] == "IDENTIFIER") == 0:
            continue
        if sum(cc_nodes["_label"] == "TYPE") == 0:
            continue
        var_type = cc_nodes[cc_nodes["_label"] == "TYPE"].head(1).name.item()
        # code_ = cc_nodes[cc_nodes["_label"] != "IDENTIFIER"].code.item()
        # name_ = cc_nodes[cc_nodes["_label"] != "IDENTIFIER"].name.item()
        # var_type = code_[0:code_.find(name_)].strip()
        for idrow in cc_nodes[cc_nodes["_label"] == "IDENTIFIER"].itertuples():
            varnametypes += [[idrow.lineNumber, var_type, idrow.name]]

    nametypes = pd.DataFrame(varnametypes, columns=["lineNumber", "type", "name"])

    nametypes = nametypes.drop_duplicates().sort_values("lineNumber")
    nametypes.type = nametypes.type.apply(tokenize)
    nametypes.name = nametypes.name.apply(tokenize)
    nametypes["nametype"] = nametypes.type + " " + nametypes.name
    nametypes = nametypes.groupby("lineNumber").agg({"nametype": lambda x: " ".join(x)})

    nametypes = nametypes.to_dict()["nametype"] 

    # 4/5. Data dependency / Control dependency context
    # Group nodes into statements
    nodesline = nodes[nodes['lineNumber'] != ""].copy()
    nodesline.lineNumber = nodesline.lineNumber.astype(int)
    # nodesline.columnNumber = nodesline.columnNumber.astype(int)
    nodesline = (
        nodesline.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )

    edgesline = edges.copy()
    edgesline.innode = edgesline.line_in
    edgesline.outnode = edgesline.line_out
    nodesline.id = nodesline.lineNumber
    edgesline = rdg(edgesline, "pdg")
    nodesline = drop_lone_nodes(nodesline, edgesline)
    # Drop duplicate edges
    edgesline = edgesline.drop_duplicates(subset=["innode", "outnode", "etype"])

    if len(edgesline) > 0:
        edgesline["etype"] = edgesline.apply(
            lambda x: "DDG" if x.etype == "REACHING_DEF" else x.etype, axis=1
        )
        edgesline = edgesline[edgesline.innode.apply(lambda x: isinstance(x, float))]
        edgesline = edgesline[edgesline.outnode.apply(lambda x: isinstance(x, float))]
    edgesline_reverse = edgesline[["innode", "outnode", "etype"]].copy()
    edgesline_reverse.columns = ["outnode", "innode", "etype"]
    uedge = pd.concat([edgesline, edgesline_reverse])
    uedge = uedge[uedge.innode != uedge.outnode]
    uedge = uedge.groupby(["innode", "etype"]).agg({"outnode": set})
    uedge = uedge.reset_index()
    if len(uedge) > 0:
        uedge = uedge.pivot("innode", "etype", "outnode")
        if "DDG" not in uedge.columns:
            uedge["DDG"] = None
        if "CDG" not in uedge.columns:
            uedge["CDG"] = None
        uedge = uedge.reset_index()[["innode", "CDG", "DDG"]]
        uedge.columns = ["lineNumber", "control", "data"]
        uedge.control = uedge.control.apply(
            lambda x: list(x) if isinstance(x, set) else []
        )
        uedge.data = uedge.data.apply(lambda x: list(x) if isinstance(x, set) else [])
        data = uedge.set_index("lineNumber").to_dict()["data"]
        control = uedge.set_index("lineNumber").to_dict()["control"]
    else:
        data = {}
        control = {}

    # Generate PDG
    # print(filepath)
    # print('subseq:', subseq)
    pdg_nodes = nodesline.copy()

    pdg_nodes = pdg_nodes[["id", "code"]].sort_values("id")  
    pdg_nodes["subseq"] = pdg_nodes.id.map(subseq).fillna("")  
    pdg_nodes["ast"] = pdg_nodes.id.map(ast).fillna("")
    pdg_nodes["nametypes"] = pdg_nodes.id.map(nametypes).fillna("")

    pdg_nodes = pdg_nodes[pdg_nodes.id.isin(list(data.keys()) + list(control.keys()))]

    pdg_nodes["data"] = pdg_nodes.id.map(data)
    pdg_nodes["control"] = pdg_nodes.id.map(control)
    pdg_nodes.data = pdg_nodes.data.map(lambda x: ' '.join([subseq[i] for i in x if i in subseq]))
    pdg_nodes.control = pdg_nodes.control.map(lambda x: ' '.join([subseq[i] for i in x if i in subseq]))
    pdg_edges = edgesline.copy()
    pdg_nodes = pdg_nodes.reset_index(drop=True).reset_index()
    pdg_dict = pd.Series(pdg_nodes.index.values, index=pdg_nodes.id).to_dict()
    pdg_edges.innode = pdg_edges.innode.map(pdg_dict)
    pdg_edges.outnode = pdg_edges.outnode.map(pdg_dict)
    pdg_edges = pdg_edges.dropna()
    pdg_edges = (pdg_edges.outnode.tolist(), pdg_edges.innode.tolist())

    # Cache
    with open(cachefp, "wb") as f:
        pickle.dump([pdg_nodes, pdg_edges], f)
    return pdg_nodes, pdg_edges
