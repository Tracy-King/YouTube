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
from sklearn.metrics import roc_auc_score, precision_score
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, get_neighbor_finder, MLP
from utils.data_processing import compute_time_statistics, get_data_node_classification
from evaluation.evaluation import eval_node_classification

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='concat_week_v3.10')  # 1kxCz6tt2MU_v3.10  concat_half_v3.10  concat_week_v3.10
parser.add_argument('--n_decoder', type=int, help='Number of ensemble decoder',
                    default=1)
parser.add_argument('--n_undersample', type=int, help='Parameter for undersampling',
                    default=3)
parser.add_argument('--dataset_r1', type=float, default=0.90, help='Validation dataset ratio')
parser.add_argument('--dataset_r2', type=float, default=0.95, help='Test dataset ratio')
parser.add_argument('--bs', type=int, default=2000, help='Batch_size')
parser.add_argument('--prefix', type=str, default='1kxCz6tt2MU_v3.10', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=1, help='Number of epochs')
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
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

DATASET_R1 = args.dataset_r1
DATASET_R2 = args.dataset_r2
N_DECODERS = args.n_decoder
TAG = 'superchat'
BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim

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

full_data, node_features, edge_features, update_records, train_data, val_data, test_data = \
    get_data_node_classification(DATA, TAG, DATASET_R1, DATASET_R2, NODE_DIM, device)

max_idx = max(full_data.unique_nodes)

train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM, max_node_idx=max_idx)

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
              n_heads=NUM_HEADS, dropout=DROP_OUT,
              embedding_dim=NODE_DIM, n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst)

    tgn = tgn.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.debug("Num of training instances: {}".format(num_instance))
    logger.debug("Num of batches per epoch: {}".format(num_batch))

    logger.info("Loading saved TGN model")
    logger.info("Start training node classification task")

    decoders = [MLP(node_features.shape[1], drop=DROP_OUT) for _ in range(N_DECODERS)]
    decoders = [decoder.to(device) for decoder in decoders]

    params = list(tgn.parameters())
    for decoder in decoders:
        params = params + list(decoder.parameters())
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

        tgn = tgn.train()
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

            source_embedding = tgn.compute_temporal_embeddings(sources_batch,
                                                               destinations_batch,
                                                               timestamps_batch,
                                                               edge_idxs_batch,
                                                               NUM_NEIGHBORS)

            labels_batch_torch = torch.from_numpy(labels_batch).long().to(device)
            with torch.no_grad():
                ones = torch.sparse.torch.eye(2)
                labels_batch_onehot = ones.index_select(0, torch.from_numpy(labels_batch)).to(device)

            pos_count = np.count_nonzero(labels_batch)
            neg_count = size - pos_count
            pos_weight = neg_count / (pos_count + 1)

            # under sampling start
            index = list(range(size))
            sample_pos_index = []
            for j in index:
                if labels_batch[j] == 1:
                    sample_pos_index.append(i)
                if len(sample_pos_index) == 0:
                    continue
            sample_neg_index = random.sample([j for j in index if j not in sample_pos_index],
                                             min(args.n_undersample * (len(sample_pos_index) + 1),
                                                 size - len(sample_pos_index)))

            sample_pos_index.extend(sample_neg_index)
            random.shuffle(sample_pos_index)
            sample_index = sample_pos_index
            # under sampling end

            for d_idx in range(N_DECODERS):
                pred_prob_num[d_idx] = decoders[d_idx](source_embedding)
            pred = torch.mean(pred_prob_num, 0).squeeze()
            pred_label = torch.argmax(pred, dim=1).detach().cpu().numpy()

            if len(np.unique(labels_batch)) == 2:
                train_auc = roc_auc_score(labels_batch, pred_label)
                train_pre = precision_score(labels_batch, pred_label)
            decoder_loss = decoder_loss_criterion(pred[sample_index], labels_batch_onehot[sample_index])

            decoder_loss.backward()
            optimizer.step()
            loss += decoder_loss.item() / N_DECODERS
            torch.cuda.empty_cache()
            if k % 100 == 0:
                logger.debug("{}/{},{}".format(k, num_batch, loss))
        train_losses.append(loss / num_batch)
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

        if early_stopper.early_stop_check(val_auc):
            logger.info("No improvement over {} epochs, stop training".format(early_stopper.max_round))
            break
        else:
            torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

    logger.info(f"Loading the best model at epoch {early_stopper.best_epoch}")
    best_model_path = get_checkpoint_path(early_stopper.best_epoch)
    tgn.load_state_dict(torch.load(best_model_path))
    logger.info(f"Loaded the best model at epoch {early_stopper.best_epoch} for inference")

    test_auc, test_acc, test_rec, test_pre, test_cm = eval_node_classification(tgn, decoders, test_data,
                                                                               full_data.edge_idxs, NODE_DIM, BATCH_SIZE,
                                                                               n_neighbors=NUM_NEIGHBORS,
                                                                               device=device)

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
