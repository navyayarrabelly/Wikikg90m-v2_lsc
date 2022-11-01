import torch
from torch_geometric.nn import GCNConv,RGATConv
from torch_geometric.utils import to_undirected
import torch.nn.functional as F




class RGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_relations):
        super().__init__()
        
        self.conv1 = RGATConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGATConv(hidden_channels, hidden_channels, num_relations)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type).relu()
        x = self.lin(x)
        return x
        #return F.log_softmax(x, dim=-1)

class AEModel (torch.nn.Module): 
    def __init__(self, args, in_channels, hidden_channels, out_channels,
                  rel_feat_path, rel_out_dim,device):
        super().__init__()
        self.device = torch.device(device)
        self.rel_feat_bert = torch.load(rel_feat_path)
        self.rel_feat_bert= self.rel_feat_bert.to(device)
        self.num_relations= self.rel_feat_bert.size()[0]
        self.rel_out_dim = rel_out_dim
        self.rel_in_channels= self.rel_feat_bert.size()[1]
       
        self.rel_lin = torch.nn.Linear(self.rel_in_channels, hidden_channels,device=device)
        self.rel_feat= torch.empty((self.num_relations,self.rel_out_dim))
        self.gamma= 10.0

        self.encoder= RGAT(in_channels, args.hidden_channels, args.out_emb_dim,
                self.num_relations)
    def updateRelationEmbeddings(self): 
        x = self.rel_lin(self.rel_feat_bert)
        x = F.relu(x)
        x = F.dropout(x, p=0.3)
        self.rel_feat= x

    def RotatE(self, head, tail, edge_type):
        relation = self.rel_feat[edge_type]
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 1)
        return score

    def TransE(self, head, tail, edge_type):
        relation = self.rel_feat[edge_type]
        score = (head + relation) - tail

        score = self.gamma - torch.norm(score, p=1, dim=1)
        return score
        #self.rel_feat.requires_grad=True

        
        
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)



def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

