import argparse

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_sparse import SparseTensor
#from torch_geometric.data import GraphSAINTRandomWalkSampler
from torch_geometric.nn import GCNConv,RGATConv
from torch_geometric.utils import to_undirected

from ogb.linkproppred import Evaluator
from dataset_pyg import PygLinkPropPredDataset
from logger import Logger
from graph_saint_loader import *
from ae_model import *


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GCNInference(torch.nn.Module):
    def __init__(self, weights):
        super(GCNInference, self).__init__()
        self.weights = weights

    def forward(self, x, adj):
        for i, (weight, bias) in enumerate(self.weights):
            x = adj @ x @ weight + bias
            x = np.clip(x, 0, None) if i < len(self.weights) - 1 else x
        return x

def train_ae(model, predictor, loader, optimizer, device):
    model.train()

    total_loss = total_examples = 0
    for data, node_idx, edge_idx in loader:
        
        optimizer.zero_grad()
        model.updateRelationEmbeddings()
        # data.edge_type= torch.randint(0,1300,(data.edge_index.size()[1],))
        data = data.to(device)
        h = model.encoder(data.x, data.edge_index, data.edge_reltype.view(-1,))

        src, dst = data.edge_index

        pos_score= model.TransE(h[src],h[dst],data.edge_reltype.view(-1,))
        #pos_loss = -torch.log(pos_out + 1e-15).mean()

        pos_loss = -(F.logsigmoid(pos_score+1e-15).mean())
        
        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, data.x.size(0), src.size(),
                                dtype=torch.long, device=device)
        neg_score= model.TransE(h[src],h[dst],data.edge_reltype.view(-1,))

        neg_loss = -(F.logsigmoid(neg_score+1e-15).mean())
        
        loss = (pos_loss + neg_loss)/2

        loss.backward()
        optimizer.step()

        num_examples = src.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

def train(model, predictor, loader, optimizer, device):
    model.train()

    total_loss = total_examples = 0
    for data, node_idx, edge_idx in loader:
        
        optimizer.zero_grad()
       
        # data.edge_type= torch.randint(0,1300,(data.edge_index.size()[1],))
        data = data.to(device)
        h = model(data.x, data.edge_index)
        #h = model(data.x, data.edge_index, data.edge_reltype.view(-1,))

        src, dst = data.edge_index
        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, data.x.size(0), src.size(),
                                dtype=torch.long, device=device)
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = src.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(args,loader,model, predictor, data, split_edge, evaluator, batch_size, device):
    model.eval()
    model.encoder.eval()

    print('Evaluating full-batch GNN on CPU...')
    with torch.no_grad():

        # weights = [(conv.lin.weight.t().cpu().detach().numpy(),
        #             conv.bias.cpu().detach().numpy()) for conv in model.convs]
        # model = GCNInference(weights)

        # x = data.x.numpy()
        # adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=data.edge_type)
        # adj = adj.set_diag()
        # deg = adj.sum(dim=1)
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        # adj = adj.to_scipy(layout='csr')

        # h = torch.from_numpy(model(x, adj)).to(device)
        num_nodes= data.num_nodes
        h= torch.empty((num_nodes,args.out_emb_dim), dtype= torch.float32, device='cpu')
        for data, node_idx, edge_idx  in loader:
            
            data.edge_type= data.edge_reltype.view(-1,)
            data = data.to(device)
            if(args.mode=="ae"):
                h_batch = model.encoder(data.x, data.edge_index, data.edge_reltype.view(-1,))
            else:
                h_batch = model(data.x, data.edge_index)
            #h_batch = model(data.x, data.edge_index, data.edge_type.view(-1,))
            h[node_idx] = h_batch.cpu()        

        def test_split(split):
            source = split_edge[split]['head'].to(device)
            target = split_edge[split]['tail'].to(device)
            relation= split_edge[split]['relation'].to(device)
            target_neg = split_edge[split]['tail_neg'].to(device)

            pos_preds = []
            for perm in DataLoader(range(source.size(0)), batch_size):
                src, dst, relation_batch = source[perm], target[perm], relation[perm]
                src_vec, dst_vec = h[src].to(device), h[dst].to(device)
                if(args.mode=="ae"): 
                    pos_preds += [model.TransE(src_vec, dst_vec,relation_batch.view(-1,)).squeeze().cpu()]
                else:
                    pos_preds += [predictor(src_vec, dst_vec).squeeze().cpu()]
            pos_pred = torch.cat(pos_preds, dim=0)

            neg_preds = []
            source = source.view(-1, 1).repeat(1, 500).view(-1)
            relation = relation.view(-1, 1).repeat(1, 500).view(-1)
            target_neg = target_neg.view(-1)
            for perm in DataLoader(range(source.size(0)), batch_size):
                src, dst_neg,relation_batch = source[perm], target_neg[perm], relation[perm]
                src_vec, dst_neg_vec = h[src.cpu()].to(device), h[dst_neg.cpu()].to(device)
                if(args.mode=="ae"): 
                    neg_preds += [model.TransE(src_vec, dst_neg_vec, relation_batch.view(-1,)).squeeze().cpu()]
                else:
                    neg_preds += [predictor(src_vec, dst_neg_vec).squeeze().cpu()]
                #neg_preds += [predictor(src_vec,dst_neg_vec).squeeze().cpu()]
            neg_pred = torch.cat(neg_preds, dim=0).view(-1, 500)

            return evaluator.eval({
                'y_pred_pos': pos_pred,
                'y_pred_neg': neg_pred,
            })['mrr_list'].mean().item()

    train_mrr = test_split('eval_train')
    valid_mrr = test_split('valid')
    print("train-mrr: "+str(train_mrr))
    print("valid-mrr: "+str(valid_mrr))
    #test_mrr = test_split('test')

    return  train_mrr,valid_mrr,valid_mrr
    #return train_mrr, valid_mrr, test_mrr


def main():
    parser = argparse.ArgumentParser(description='OGBL-Citation2 (GraphSAINT)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--walk_length', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--mode', type=str, default="ae")
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--out_emb_dim', type=int, default=256)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--rel_feat_path', type=str, default="/data/ogb/ogbl_wikikg2/smore_folder/sample/raw/rel_feat.pt")
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(root='/data/ogb/ogbl_wikikg2/smore_folder', name='wikikg2_v2_sample')
    split_edge = dataset.get_edge_split()
    data = dataset[0]
    #data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    #data.edge_type=data.edge_reltype

    loader = GraphSAINTRandomWalkSampler(data, batch_size=args.batch_size,
                                         walk_length=args.walk_length,
                                         num_steps=args.num_steps,
                                         sample_coverage=0,
                                         save_dir=dataset.processed_dir)

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['head'].numel())[:15000]
    split_edge['eval_train'] = {
        'head': split_edge['train']['head'][idx],
        'relation': split_edge['train']['relation'][idx],
        'tail': split_edge['train']['tail'][idx],
        'tail_neg': split_edge['valid']['tail_neg'],
    }

    # model = GCN(data.x.size(-1), args.hidden_channels, args.hidden_channels,
    #             args.num_layers, args.dropout).to(device)
    
    # model = RGAT(data.x.size(-1), args.hidden_channels, args.hidden_channels,
    #             num_relations=1500).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)
    if(args.mode=="ae"):
        model = AEModel( args, data.x.size(-1),  args.hidden_channels, out_channels=args.out_emb_dim,
                rel_feat_path= args.rel_feat_path, rel_out_dim= args.out_emb_dim, device=args.device)
    model = model.to(args.device)
    evaluator = Evaluator(name='ogbl-wikikg2')
    logger = Logger(args.runs, args)

    run_idx = 0

    while run_idx < args.runs:
        # model.reset_parameters()
        # predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        run_success = True
        for epoch in range(1, 1 + args.epochs):
            loss = train_ae(model, predictor, loader, optimizer, device)
            print(
                f'Run: {run_idx + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}'
            )
            if loss > 2.:
                run_success = False
                logger.reset(run_idx)
                print('Learning failed. Rerun...')
                break

            if epoch >=1 and epoch % args.eval_steps == 0:
                result = test(args,loader,model, predictor, data, split_edge, evaluator,
                              batch_size=1024, device=device)
                logger.add_result(run_idx, result)

                train_mrr, valid_mrr, test_mrr = result
                print(f'Run: {run_idx + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {train_mrr:.4f}, '
                      f'Valid: {valid_mrr:.4f}, '
                      f'Test: {test_mrr:.4f}')

        print('GraphSAINT')
        if run_success:
            logger.print_statistics(run_idx)
            run_idx += 1

    print('GraphSAINT')
    logger.print_statistics()

if __name__ == "__main__":
    main()
