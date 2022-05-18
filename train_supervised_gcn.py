import math
import logging
import time
import sys
import random
import argparse
import pickle
from pathlib import Path
import inspect

import torch
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score, \
    confusion_matrix

from model.tgn import TGN
from model.softDT import DTArgs, SoftDecisionTree
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
                    default='1kxCz6tt2MU_v3.10')  # 1kxCz6tt2MU_v3.10  concat_half_v3.10  concat_week_v3.10
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
parser.add_argument('--dataset_r1', type=float, default=0.50, help='Validation dataset ratio')
parser.add_argument('--dataset_r2', type=float, default=0.75, help='Test dataset ratio')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='tgn-attn-1kxCz6tt2MU_v2', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
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
parser.add_argument('--original_encoder', action='store_true', help='Use original TGN encoder')
parser.add_argument('--original_decoder', action='store_true', help='Use original TGN decoder')

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

class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = torch.nn.functional.relu(self.gc1(x, adj))
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.nn.functional.relu(self.gc2(x, adj))
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return torch.nn.functional.log_softmax(x, dim=1)




for i in range(args.n_runs):
    results_path = "results/{}_node_classification_{}.pkl".format(args.prefix,
                                                                  i) if i > 0 else "results/{}_node_classification.pkl".format(
        args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)

    # Initialize Model
    model = GCN(nfeat=node_features.shape[1], nhid=node_features.shape[1], nclass=2, dropout=0.8)

    model = model.to(device)
    node_features = torch.from_numpy(node_features).float().to(device)
    adj_train = sparse_mx_to_torch_sparse_tensor(adj_train).to(device)
    adj_val = sparse_mx_to_torch_sparse_tensor(adj_val).to(device)
    adj_test = sparse_mx_to_torch_sparse_tensor(adj_test).to(device)



    # if DECODER=='GBDT':
    #  decoder = GradientBoostingClassifier(max_depth=args.max_depth, n_estimators=args.n_estimators, learning_rate=LEARNING_RATE)
    # elif DECODER=='XGB':
    #  decoder = xgb.XGBClassifier(max_depth=args.max_depth, learning_rate=LEARNING_RATE, n_estimators=args.n_estimators,
    #                            objective='reg:logistic', use_label_encoder=False)
    # decoder = SoftDecisionTree(DTargs).to(device)


    # params = [decoder.parameters() for decoder in decoders]
    # params = params + list(tgn.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

        pred = model(node_features, adj_train)

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
        pred_val = model(node_features, adj_val)
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
    pred_test = model(node_features, adj_test)
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
