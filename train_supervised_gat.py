import math
import logging
import time
import sys
import random
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score, \
    confusion_matrix
import torch.nn as nn
import torch.nn.functional as F

from model.tgn import TGN
from utils.utils import EarlyStopMonitor, get_neighbor_finder, MLP
from utils.data_processing import compute_time_statistics, get_data_node_classification
from evaluation.evaluation import eval_node_classification

import scipy.sparse as sp

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='concat_half_v3.10')  #   1kxCz6tt2MU_v3.10  concat_half_v3.10  concat_week_v3.10
parser.add_argument('--n_decoder', type=int, help='Number of ensemble decoder',
                    default=2)
parser.add_argument('--label', type=str, help='Label type(eg. superchat or membership)',
                    choices=['superchat', 'membership'], default='superchat')
parser.add_argument('--decoder', type=str, help='Type of decoder', choices=['GBDT', 'XGB'],
                    default='GBDT')
parser.add_argument('--n_estimators', type=int, help='Number of estimators in decoder',
                    default=3000)
parser.add_argument('--max_depth', type=int, help='Number of maximum depth in decoder',
                    default=20)
parser.add_argument('--dataset_r1', type=float, default=0.90, help='Validation dataset ratio')
parser.add_argument('--dataset_r2', type=float, default=0.95, help='Test dataset ratio')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='tgn-attn-1kxCz6tt2MU_v2', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.2, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=128, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=128, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
parser.add_argument('--aggregator', type=str, default="mean", help='Type of message '
                                                                   'aggregator')
parser.add_argument('--updater_type', type=str, default="lstm", choices=[
    "lstm", "gru", "rnn"],help='Type of updater')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=64, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=64, help='Dimensions of the memory for '
                                                               'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--use_validation', action='store_true',
                    help='Whether to use a validation set')
parser.add_argument('--new_node', action='store_true', help='model new node')
parser.add_argument('--nb_heads', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden', type=int, default=1, help='Number of hidden units.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

args.original_encoder = False
args.original_decoder = False

torch.autograd.set_detect_anomaly(True)
# args.original_encoder = True
args.use_memory = args.original_encoder
# args.use_memory = True

args.uniform = False

# args.use_source_embedding_in_message = True
# args.use_validation = True
# args.use_destination_embedding_in_message = True

DATASET_R1 = args.dataset_r1
DATASET_R2 = args.dataset_r2
DECODER = args.decoder
if args.original_decoder:
    N_DECODERS = 1
else:
    N_DECODERS = args.n_decoder
TAG = args.label
BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = False  # args.uniform
NEW_NODE = args.new_node
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = False  # args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f"./saved_models/{args.prefix}-{args.data}" + "\
  node-classification.pth"
get_checkpoint_path = lambda \
        epoch: f"./saved_checkpoints/{args.prefix}-{args.data}-{epoch}" + "\
  node-classification.pth"

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("log/{}.log".format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

torch.cuda.empty_cache()

# Set device
device_string = "cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)

# gpu_tracker = MemTracker()
# print('Initial')
# gpu_tracker.track()
# full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

full_data, node_features, edge_features, update_records, train_data, val_data, test_data = \
    get_data_node_classification(DATA, TAG, DATASET_R1, DATASET_R2, NODE_DIM, device,
                                 use_validation=args.use_validation)

n_nodes = node_features.shape[0]
labels = np.zeros((n_nodes))
#print(node_features.shape, full_data.n_unique_nodes)
n_train_edges = train_data.n_interactions


#train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM, max_node_idx=max_idx)

#DTargs = DTArgs(args.bs, args.node_dim, args.n_epoch, args.lr, device)

# Compute time statistics
#mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
#    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

adj_full = np.zeros((n_nodes, n_nodes))
for idx in range(n_train_edges):
    src = train_data.sources[idx]
    dst = train_data.destinations[idx]
    labels[src] = max(labels[src], train_data.labels[idx])
    adj_full[src, dst] += 1
    adj_full[dst, src] += 1

adj_train = sp.csr_matrix(adj_full)
adj_train = adj_train + adj_train.T.multiply(adj_train.T > adj_train) - adj_train.multiply(adj_train.T > adj_train)
adj_train = normalize(adj_train + sp.eye(adj_train.shape[0]))

for idx in range(val_data.n_interactions):
    src = val_data.sources[idx]
    dst = val_data.destinations[idx]
    labels[src] = max(labels[src], val_data.labels[idx])
    adj_full[src, dst] += 1
    adj_full[dst, src] += 1

adj_val = sp.csr_matrix(adj_full)
# build symmetric adjacency matrix
adj_val = adj_val + adj_val.T.multiply(adj_val.T > adj_val) - adj_val.multiply(adj_val.T > adj_val)
adj_val = normalize(adj_val + sp.eye(adj_val.shape[0]))

for idx in range(test_data.n_interactions):
    src = test_data.sources[idx]
    dst = test_data.destinations[idx]
    labels[src] = max(labels[src], test_data.labels[idx])
    adj_full[src, dst] += 1
    adj_full[dst, src] += 1

adj_test = sp.csr_matrix(adj_full)

adj_test = adj_test + adj_test.T.multiply(adj_test.T > adj_test) - adj_test.multiply(adj_test.T > adj_test)
adj_test = normalize(adj_test + sp.eye(adj_test.shape[0]))

#print(labels.dtype, labels.shape, np.count_nonzero(labels))

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = "cuda:{}".format(GPU) if input.is_cuda else 'cpu'
        #print(input.type())
        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)





for i in range(args.n_runs):
    results_path = "results/{}_node_classification_{}.pkl".format(args.prefix,
                                                                  i) if i > 0 else "results/{}_node_classification.pkl".format(
        args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)

    # Initialize Model
    model = SpGAT(nfeat=node_features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.drop_out,
                nheads=args.nb_heads,
                alpha=args.alpha)
    optimizer = torch.optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    model = model.to(device)
    node_features = torch.from_numpy(node_features).float().to(device)
    adj_train = sparse_mx_to_torch_sparse_tensor(adj_train).to(device)
    adj_val = sparse_mx_to_torch_sparse_tensor(adj_val).to(device)
    adj_test = sparse_mx_to_torch_sparse_tensor(adj_test).to(device)


    #print('Adj type:', adj_train.type())
    # if DECODER=='GBDT':
    #  decoder = GradientBoostingClassifier(max_depth=args.max_depth, n_estimators=args.n_estimators, learning_rate=LEARNING_RATE)
    # elif DECODER=='XGB':
    #  decoder = xgb.XGBClassifier(max_depth=args.max_depth, learning_rate=LEARNING_RATE, n_estimators=args.n_estimators,
    #                            objective='reg:logistic', use_label_encoder=False)
    # decoder = SoftDecisionTree(DTargs).to(device)


    # params = [decoder.parameters() for decoder in decoders]
    # params = params + list(tgn.parameters())

    val_aucs = []
    train_losses = []
    val_accs = []
    val_recs = []
    val_pres = []
    val_cms = []

    early_stopper = EarlyStopMonitor(max_round=args.patience)
    for epoch in range(args.n_epoch):
        start_epoch = time.time()


        model = model.train()
        # decoder = decoder.train()
        loss = 0
        train_auc = 0
        train_pre = 0
        optimizer.zero_grad()

        pred = model(node_features, adj_train.to_dense())

        # labels_batch_torch = torch.from_numpy(labels_batch).long().to(device)
        with torch.no_grad():
            ones = torch.sparse.torch.eye(2)
            labels_batch_onehot = ones.index_select(0, torch.from_numpy(labels).int()).to(device)
        # labels_batch_torch = torch.from_numpy(labels_batch).float().to(device)
            '''
            weight = torch.from_numpy(np.array([1.0 if i==0 else 10.0 for i in labels_batch]).astype(np.float32)).to(device)
            decoder_loss_criterion = torch.nn.BCELoss(weight=weight)
            pred = decoder(source_embedding).sigmoid()
            decoder_loss = decoder_loss_criterion(pred, labels_batch_torch)
            decoder_loss.backward()
            decoder_optimizer.step()
            loss += decoder_loss.item()
            train_losses.append(loss / num_batch)
      
            val_auc, val_acc, val_rec, val_pre, val_cm = eval_node_classification(tgn, decoder, val_data, full_data.edge_idxs, BATCH_SIZE,
                                             n_neighbors=NUM_NEIGHBORS)
              '''

            # pred = decoder(source_embedding).sigmoid()

            #  decoder_loss_criterion = torch.nn.MSELoss()
            # print('auc:', roc_auc_score(labels_batch[sample_index], pred_u))
        torch.cuda.empty_cache()
        decoder_loss = 0
        if (torch.isfinite(pred) == False).nonzero().shape[0] != 0:
            print("max and min and inf of pos_prob: ", min(pred), max(pred),
            (torch.isfinite(pred) == False).nonzero().shape[0])
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        pos_count = np.count_nonzero(labels)
        neg_count = n_train_edges - pos_count
        pos_weight = neg_count / (pos_count + 1)

        with torch.no_grad():
            pred_label = torch.argmax(pred, dim=1).detach().cpu().numpy()
            if len(np.unique(labels)) == 2:
                train_auc = roc_auc_score(labels, pred_label)
                train_pre = precision_score(labels, pred_label)

        decoder_loss_criterion = torch.nn.BCEWithLogitsLoss()
        #decoder_loss_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
        decoder_loss = decoder_loss_criterion(pred, labels_batch_onehot)
        decoder_loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        train_losses.append(decoder_loss.item())
        loss += decoder_loss

        model = model.eval()
        pred_val = model(node_features, adj_val.to_dense())
        pred_label = torch.argmax(pred_val, dim=1).detach().cpu().numpy()


        val_acc = accuracy_score(labels, pred_label)
        val_pre = precision_score(labels, pred_label)
        val_rec = recall_score(labels, pred_label)
        # print('acc:{}, pre:{}, rec:{}'.format(acc, pre, rec))
        val_cm = confusion_matrix(labels, pred_label)
        # print('confusion matrix', cm)
        # print(data.labels.shape, pred_prob.shape, pred_prob)
        val_auc = roc_auc_score(labels, pred_label)

        val_aucs.append(val_auc)
        val_accs.append(val_acc)
        val_recs.append(val_rec)
        val_pres.append(val_pre)
        val_cms.append(val_cm)

        pickle.dump({
            "val_aps": val_aucs,
            "val_acc": val_accs,
            "val_rec": val_recs,
            "val_pre": val_pres,
            "val_cm": val_cms,
            "train_losses": train_losses,
            "epoch_times": [0.0],
            "new_nodes_val_aps": [],
        }, open(results_path, "wb"))

        logger.info(
            f"Epoch {epoch}: train loss: {loss}, train auc{train_auc}, train pre{train_pre},  val auc: {val_auc}, val acc: {val_acc}, "
            f"val rec: {val_rec}, val pre: {val_pre}, val cm: {val_cm}, time: {time.time() - start_epoch}")

    model = model.eval()
    pred_test = model(node_features, adj_test.to_dense())
    pred_label = torch.argmax(pred_test, dim=1).detach().cpu().numpy()

    test_acc = accuracy_score(labels, pred_label)
    test_rec = precision_score(labels, pred_label)
    test_pre = recall_score(labels, pred_label)
    # print('acc:{}, pre:{}, rec:{}'.format(acc, pre, rec))
    test_cm = confusion_matrix(labels, pred_label)
    # print('confusion matrix', cm)
    # print(data.labels.shape, pred_prob.shape, pred_prob)
    test_auc = roc_auc_score(labels, pred_label)


    pickle.dump({
        "val_aps": val_aucs,
        "val_acc": val_accs,
        "val_rec": val_recs,
        "val_pre": val_pres,
        "val_cm": val_cms,
        "test_ap": test_auc,
        "test_acc": test_acc,
        "test_rec": test_rec,
        "test_pre": test_pre,
        "test_cm": test_cm,
        "train_losses": train_losses,
        "epoch_times": [0.0],
        "new_nodes_val_aps": [],
        "new_node_test_ap": 0,
    }, open(results_path, "wb"))

    logger.info(f"test auc: {test_auc},val acc: {test_acc}, "
                f"val rec: {test_rec}, val pre: {test_pre}, val cm: {val_cm}")
