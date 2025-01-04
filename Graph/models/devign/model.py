import torch
from dgl.nn.pytorch import GatedGraphConv
from torch import nn
import torch.nn.functional as f
import dgl


class DevignModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types=3, num_steps=6):
        super(DevignModel, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=2)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, g = None , graph=None,feat=None,eweight=None,output_tSNE_embedding = False , tSNE_embedding_shape = -1 ):
        if graph is not None:
            assert graph is not None and feat is not None
            g = graph

        features = g.ndata['_WORD2VEC']
        if feat is not None:
            features = feat
        # print(f"features: {features.shape}")
        edge_types = g.edata["_ETYPE"]
        outputs = self.ggnn(g, features, edge_types)
        g.ndata['GGNNOUTPUT'] = outputs
        # x_i, _ = dgl.unbatch(g, features)
        # print(x_i.shape)
        # h_i, _ = dgl.unbatch(g, outputs)
        # print(h_i.shape)
        x_i, h_i = self.unbatch_features(g)
        x_i = torch.stack(x_i)
        h_i = torch.stack(h_i)
        # print(f"h_i {h_i.shape} x_i {x_i.shape}")

        c_i = torch.cat((h_i, x_i), dim=-1)

        # batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(
            f.relu(
                self.conv_l1(h_i.transpose(1, 2))
            )
        )
        Y_2 = self.maxpool2(
            f.relu(
                self.conv_l2(Y_1)
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            f.relu(
                self.conv_l1_for_concat(c_i.transpose(1, 2))
            )
        )
        Z_2 = self.maxpool2_for_concat(
            f.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))    # [bs_size, xx ,2]
        avg = before_avg.mean(dim=1).squeeze(dim=1) # # [bs_size, 2]
        if output_tSNE_embedding:
            tSNE_embedding = before_avg.detach().cpu().flatten(start_dim=1)
            tSNE_embedding = torch.concat(
                [tSNE_embedding, torch.zeros(tSNE_embedding.shape[0], tSNE_embedding_shape - tSNE_embedding.shape[1])],
                dim=1)
            return self.softmax(avg) , tSNE_embedding
        return self.softmax(avg)

    def unbatch_features(self, g):
        x_i = []
        h_i = []
        max_len = -1
        for g_i in dgl.unbatch(g):
            x_i.append(g_i.ndata['_WORD2VEC'])
            h_i.append(g_i.ndata['GGNNOUTPUT'])
            max_len = max(g_i.number_of_nodes(), max_len)
            max_len = max(10,max_len)
        for i, (v, k) in enumerate(zip(x_i, h_i)):
            x_i[i] = torch.cat(
                (v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad,
                                device=v.device)), dim=0)
            h_i[i] = torch.cat(
                (k, torch.zeros(size=(max_len - k.size(0), *(k.shape[1:])), requires_grad=k.requires_grad,
                                device=k.device)), dim=0)
        return x_i, h_i


