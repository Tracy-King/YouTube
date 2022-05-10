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

from gpu_mem_track import MemTracker

from xgboost import plot_importance
from matplotlib import pyplot as plt

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='1kxCz6tt2MU_v3.10')
parser.add_argument('--n_decoder', type=int, help='Number of ensemble decoder',
                    default=30)
parser.add_argument('--label', type=str, help='Label type(eg. superchat or membership)',
                    choices=['superchat', 'membership'], default='superchat')
parser.add_argument('--decoder', type=str, help='Type of decoder', choices=['GBDT', 'XGB'],
                    default='GBDT')
parser.add_argument('--n_estimators', type=int, help='Number of estimators in decoder',
                    default=3000)
parser.add_argument('--max_depth', type=int, help='Number of maximum depth in decoder',
                    default=20)
parser.add_argument('--dataset_r1', type=float, default=0.70, help='Validation dataset ratio')
parser.add_argument('--dataset_r2', type=float, default=0.85, help='Test dataset ratio')
parser.add_argument('--bs', type=int, default=2000, help='Batch_size')
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

full_data, node_features, edge_features, update_records, train_data, val_data, test_data = \
    get_data_node_classification(DATA, TAG, DATASET_R1, DATASET_R2, NODE_DIM, device,
                                 use_validation=args.use_validation)

max_idx = max(full_data.unique_nodes)

train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM, max_node_idx=max_idx)

DTargs = DTArgs(args.bs, args.node_dim, args.n_epoch, args.lr, device)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
    results_path = "results/{}_node_classification_{}.pkl".format(args.prefix,
                                                                  i) if i > 0 else "results/{}_node_classification.pkl".format(
        args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)

    # Initialize Model
    tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
              edge_features=edge_features, update_records=update_records, device=device, data=DATA,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
              message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
              memory_update_at_start=not args.memory_update_at_end,
              embedding_module_type=args.embedding_module,
              embedding_dim=NODE_DIM,
              message_function=args.message_function,
              aggregator_type=args.aggregator, memory_updater_type=args.updater_type, n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message,
              original_encoder=args.original_encoder)

    tgn = tgn.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.debug("Num of training instances: {}".format(num_instance))
    logger.debug("Num of batches per epoch: {}".format(num_batch))

    logger.info("Loading saved TGN model")
    # model_path = f"./saved_models/{args.prefix}-{DATA}.pth"
    # tgn.load_state_dict(torch.load(model_path))
    # tgn.eval()
    # logger.info("TGN models loaded")
    logger.info("Start training node classification task")

    decoders = [MLP(node_features.shape[1], drop=DROP_OUT) for _ in range(N_DECODERS)]
    decoders = [decoder.to(device) for decoder in decoders]

    # if DECODER=='GBDT':
    #  decoder = GradientBoostingClassifier(max_depth=args.max_depth, n_estimators=args.n_estimators, learning_rate=LEARNING_RATE)
    # elif DECODER=='XGB':
    #  decoder = xgb.XGBClassifier(max_depth=args.max_depth, learning_rate=LEARNING_RATE, n_estimators=args.n_estimators,
    #                            objective='reg:logistic', use_label_encoder=False)
    # decoder = SoftDecisionTree(DTargs).to(device)
    decoder = MLP(node_features.shape[1], drop=DROP_OUT).to(device)

    params = list(tgn.parameters())
    for decoder in decoders:
        params = params + list(decoder.parameters())
    # params = [decoder.parameters() for decoder in decoders]
    # params = params + list(tgn.parameters())
    optimizer = torch.optim.Adagrad(params, lr=LEARNING_RATE)
    decoder_loss_criterion = torch.nn.BCELoss()

    val_aucs = []
    train_losses = []
    val_accs = []
    val_recs = []
    val_pres = []
    val_cms = []

    early_stopper = EarlyStopMonitor(max_round=args.patience)
    for epoch in range(args.n_epoch):
        start_epoch = time.time()

        # Initialize memory of the model at each epoch
        if USE_MEMORY:
            tgn.memory.__init_memory__()

        tgn = tgn.train()
        # decoder = decoder.train()
        decoders = [decoder.train() for decoder in decoders]
        loss = 0
        train_auc = 0
        train_pre = 0
        optimizer.zero_grad()

        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance, s_idx + BATCH_SIZE)

            sources_batch = train_data.sources[s_idx: e_idx]
            destinations_batch = train_data.destinations[s_idx: e_idx]
            timestamps_batch = train_data.timestamps[s_idx: e_idx]
            edge_idxs_batch = full_data.edge_idxs[s_idx: e_idx]
            labels_batch = train_data.labels[s_idx: e_idx]

            size = len(sources_batch)
            pred_prob_num = torch.zeros((N_DECODERS, size, 2)).to(device)

            with torch.no_grad():
                source_embedding, destination_embedding = tgn.compute_temporal_embeddings_seq(sources_batch,
                                                                                          destinations_batch,
                                                                                          timestamps_batch,
                                                                                          edge_idxs_batch,
                                                                                          NUM_NEIGHBORS)

            # labels_batch_torch = torch.from_numpy(labels_batch).long().to(device)
            ones = torch.sparse.torch.eye(2)
            labels_batch_onehot = ones.index_select(0, torch.from_numpy(labels_batch)).to(device)
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
            if (torch.isfinite(source_embedding) == False).nonzero().shape[0] != 0:
                print("max and min and inf of pos_prob: ", min(source_embedding), max(source_embedding),
                      (torch.isfinite(source_embedding) == False).nonzero().shape[0])
                source_embedding = torch.nan_to_num(source_embedding, nan=0.0, posinf=1.0, neginf=0.0)
            if TAG == 'superchat':
                pos_count = np.count_nonzero(labels_batch)
                neg_count = size - pos_count
                pos_weight = neg_count / (pos_count + 1)
                # print("pos_weight:{}".format(pos_weight))
                '''
                # under sampling start
                index = list(range(size))
                sample_pos_index = []
                for i in index:
                  if labels_batch[i]==1:
                    sample_pos_index.append(i)
                if len(sample_pos_index) == 0:
                  continue
                sample_neg_index = random.sample([i for i in index if i not in sample_pos_index],
                                               min((len(sample_pos_index)+1), size-len(sample_pos_index)))
                #sample_neg_index = random.sample([i for i in index if i not in sample_pos_index], (len(sample_pos_index) + 1))
                sample_pos_index.extend(sample_neg_index)
                random.shuffle(sample_pos_index)
                sample_index = sample_pos_index
                #under sampling end
                '''
                # train_x = source_embedding[sample_index].clone().detach().cpu().numpy()
                # train_y = labels_batch[sample_index]
                # train_x = source_embedding.clone().detach().cpu().numpy()
                # train_y = labels_batch
                # print(len(sample_index))
                # under sampling end
                # pred_u = pred[sample_index].clone().detach().cpu().numpy()
                decoder_loss_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
                # if DECODER=='GBDT':
                #  decoder.fit(train_x, train_y)
                # elif DECODER=='XGB':
                #  decoder.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric=['logloss'], verbose=False)
                # decoder_loss = decoder_loss_criterion(pred, labels_batch_onehot)
                # decoder.fit(train_x, train_y)    # for GBDT
                # decoder_loss, pred = decoder.train_(source_embedding, labels_batch_torch, size)
                # print(decoder.evals_result())
                # decoder_loss = np.mean(decoder.evals_result()['validation_0']['logloss'])
                for d_idx in range(N_DECODERS):
                    pred_prob_num[d_idx] = decoders[d_idx](source_embedding)
                pred = torch.mean(pred_prob_num, 0).squeeze()
                pred_label = torch.argmax(pred, dim=1).detach().cpu().numpy()
                if len(np.unique(labels_batch)) == 2:
                    train_auc = roc_auc_score(labels_batch, pred_label)
                    train_pre = precision_score(labels_batch, pred_label)
                # print(pred.shape)
                # pred = (pred_prob_num+0.5).trunc()
                decoder_loss += decoder_loss_criterion(pred, labels_batch_onehot)
            else:
                sample_index = random.sample(list(range(size)), int(size / 10))
                # print(len(sample_index))
                random.shuffle(sample_index)
                train_x = source_embedding[sample_index].clone().detach().cpu().numpy()
                train_y = labels_batch[sample_index]
                if DECODER == 'GBDT':
                    decoder.fit(train_x, train_y)
                elif DECODER == 'XGB':
                    decoder.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric=['logloss'], verbose=False)
                # decoder.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric=['logloss'], verbose=False)
                # decoder.fit(train_x, train_y)   # for GBDT
                # print(pred[sample_index][:10], labels_batch_torch[sample_index][:10])
                # decoder_loss, pred = decoder.train_(source_embedding[sample_index], labels_batch_torch[sample_index], size)
                # decoder_loss_criterion(pred[sample_index], labels_batch_torch[sample_index])
            # decoder_loss = np.mean(decoder.evals_result()['validation_0']['logloss']) if DECODER=='XGB' else 0.0
            decoder_loss.backward()
            optimizer.step()
            loss += decoder_loss / N_DECODERS
            torch.cuda.empty_cache()
            if (k % 1000 == 0):
                print(k, loss)
                # gpu_tracker.track()
        train_losses.append(loss.item())
        # gpu_tracker.track()
        torch.cuda.empty_cache()

        val_auc, val_acc, val_rec, val_pre, val_cm = eval_node_classification(tgn, decoders, val_data,
                                                                              full_data.edge_idxs, NODE_DIM,
                                                                              BATCH_SIZE, n_neighbors=NUM_NEIGHBORS,
                                                                              device=device)

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
            f"Epoch {epoch}: train loss: {loss / num_batch}, train auc{train_auc}, train pre{train_pre},  val auc: {val_auc}, val acc: {val_acc}, "
            f"val rec: {val_rec}, val pre: {val_pre}, val cm: {val_cm}, time: {time.time() - start_epoch}")

    if args.use_validation:
        if early_stopper.early_stop_check(val_auc):
            logger.info("No improvement over {} epochs, stop training".format(early_stopper.max_round))
            break
        else:
            torch.save(decoder.state_dict(), get_checkpoint_path(epoch))

    if args.use_validation:
        logger.info(f"Loading the best model at epoch {early_stopper.best_epoch}")
        best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        decoder.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded the best model at epoch {early_stopper.best_epoch} for inference")
        decoder.eval()

        test_auc, test_acc, test_rec, test_pre, test_cm = eval_node_classification(tgn, decoder, test_data,
                                                                                   full_data.edge_idxs, BATCH_SIZE,
                                                                                   n_neighbors=NUM_NEIGHBORS,
                                                                                   device=device)
    else:
        # If we are not using a validation set, the test performance is just the performance computed
        # in the last epoch
        test_auc = val_aucs[-1]
        test_acc = val_accs[-1]
        test_rec = val_recs[-1]
        test_pre = val_pres[-1]
        test_cm = val_cms[-1]

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
    # if DECODER=='XGB':
    #  plot_importance(decoder)
    #  plt.show()
    #  plt.savefig(f'{results_path}_feature_importance.png')
