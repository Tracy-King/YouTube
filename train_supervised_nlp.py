import math
import logging
import time
import sys
import random
import argparse
import pickle
from pathlib import Path
import inspect
import pandas as pd
import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertJapaneseTokenizer, BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score, \
    confusion_matrix
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
parser.add_argument('--dataset_r1', type=float, default=0.70, help='Validation dataset ratio')
parser.add_argument('--dataset_r2', type=float, default=0.85, help='Test dataset ratio')
parser.add_argument('--bs', type=int, default=100, help='Batch_size')
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

#full_data, node_features, edge_features, update_records, train_data, val_data, test_data = \
#    get_data_node_classification(DATA, TAG, DATASET_R1, DATASET_R2, NODE_DIM, device,
#                                 use_validation=args.use_validation)

single = ['1kxCz6tt2MU']
concat_week = ['ON3WijEIS1c', 'qO8Ld-qLjb0', 'k3Nzow_OqQY', 'y3DCfZmX8iA', 'qHZwDxea7fQ']#, 'cibdBr9TkEo', 'rW8jSXVsW2E', 'eIi8zCPFyng', 'wtJj3CO_YR0']
concat_half = ['ON3WijEIS1c', 'qO8Ld-qLjb0', 'k3Nzow_OqQY', 'y3DCfZmX8iA', 'qHZwDxea7fQ', 'cibdBr9TkEo', 'rW8jSXVsW2E', 'eIi8zCPFyng', 'wtJj3CO_YR0']
model_path = 'cl-tohoku/bert-base-japanese'# 'bert-base-cased' #'BERT-base_mecab-ipadic-bpe-32k'
max_length = 32
data_list = concat_week
ratio = 0.9

flag = 0
node_dict = dict()

for video in data_list:
    tmp = pd.read_csv('./embedding/UC1opHUrw8rvnsadT-iGp7Cg/{}.csv'.format(video))
    if flag == 0:
        graph_df = tmp
        flag = 1
    else:
        graph_df = pd.concat([graph_df, tmp], ignore_index=True)

nodes = graph_df['commenter_id'].drop_duplicates()
node_list = nodes.tolist()
n_nodes = len(node_list)
node_dict = dict(zip(node_list, range(n_nodes)))
labels = np.zeros(n_nodes, dtype='int32')
body = np.zeros(n_nodes, dtype='unicode')
body = body.astype('object')

# Load the pretrained Tokenizer
tokenizer = BertJapaneseTokenizer.from_pretrained(model_path, word_tokenizer_type='mecab')


for i, row in graph_df.iterrows():
    idx = node_dict[row['commenter_id']]
    labels[idx] = max(labels[idx], min(int(row['superchat']), 1))
    #print('row', row['body'])
    body[idx] = str(body[idx]) + str(row['body']) + '\t'
    #print('body', body[idx])


def encode_fn(text_list):
    all_input_ids = []
    for text in text_list:
        input_ids = tokenizer.encode(
                        text,
                        add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
                        pad_to_max_length=True,           # 设定最大文本长度
                        max_length = max_length,   # pad到最大的长度
                        return_tensors = 'pt'       # 返回的类型为pytorch tensor
                   )
        all_input_ids.append(input_ids)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids


def flat_evaluation(preds, labels):
    """A function for calculating accuracy scores"""

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()

    #print(pred_flat.shape, labels_flat.shape)
    acc = accuracy_score(labels_flat, pred_flat)
    if np.count_nonzero(labels_flat) == 0:
        print('No positive val sample')
        auc = 0.0
    else:
        auc = roc_auc_score(labels_flat, pred_flat)
        cm = confusion_matrix(labels_flat, pred_flat)
        print(cm)
    return acc, auc#, confusion_matrix(labels_flat, pred_flat)

all_input_ids = encode_fn(body)
print('labels:', np.count_nonzero(labels), 'positive in ', n_nodes)

pos_idx = np.where(labels==1)[0]
#pos_idx = np.tile(pos_idx, int(len(labels)/np.count_nonzero(labels)))
neg_idx = np.where(labels==0)[0]
#node_idx = np.hstack((np.arange(n_nodes), pos_idx))
#labels = labels[node_idx]

train_size = int(ratio * n_nodes)
pos_split = np.split(pos_idx, [int(ratio*len(pos_idx)), n_nodes])
neg_split = np.split(neg_idx, [int(ratio*len(neg_idx)), n_nodes])
print(pos_split[0].shape, pos_split[1].shape, neg_split[0].shape, neg_split[1].shape)
train_node_idx = np.hstack((neg_split[0], np.tile(pos_split[0], int(n_nodes/len(pos_idx)))))
val_node_idx = np.hstack((neg_split[1], np.tile(pos_split[1], int(n_nodes/len(pos_idx)))))

print(pos_idx.shape, neg_idx.shape, np.count_nonzero(labels[train_node_idx]), np.count_nonzero(labels[val_node_idx]))


ones = torch.sparse.torch.eye(2)
labels = ones.index_select(0, torch.from_numpy(labels)).to(device)

#print(all_input_ids.shape, labels.shape)

epochs = NUM_EPOCH
batch_size = BATCH_SIZE

# Split data into train and validation

train_dataset = TensorDataset(torch.from_numpy(train_node_idx), labels[train_node_idx])
val_dataset = TensorDataset(torch.from_numpy(val_node_idx), labels[val_node_idx])

# Create train and validation dataloaders
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

# Load the pretrained BERT model

model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2,
                                                      output_attentions=False, output_hidden_states=False)
model.cuda()

# create optimizer and learning rate schedule
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(epochs):
    model.train()
    total_loss, total_val_loss = 0, 0
    total_eval_accuracy = 0
    total_eval_auc = 0
    for step, batch in enumerate(train_dataloader):
        model.zero_grad()
        loss, logits = model(all_input_ids[batch[0]].to(device), token_type_ids=None,
                             attention_mask=(all_input_ids[batch[0]] > 0).to(device),
                             labels=batch[1].to(device), return_dict=False)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    y_pred = np.array([])
    y_true = np.array([])

    for i, batch in enumerate(val_dataloader):
        with torch.no_grad():
            loss, logits = model(all_input_ids[batch[0]].to(device), token_type_ids=None,
                                 attention_mask=(all_input_ids[batch[0]] > 0).to(device),
                                 labels=batch[1].to(device), return_dict=False)

            total_val_loss += loss.item()
            #print(y_pred.shape, logits.shape)
            logits = logits.detach().cpu().numpy()

            label_ids = batch[1].to('cpu').numpy()
            #print(y_true.shape, y_true.shape)

            if y_pred.shape[0] == 0:
                y_pred = logits
                y_true = label_ids
            else:
                y_pred = np.vstack((y_pred, logits))
                y_true = np.vstack((y_true, label_ids))

    avg_val_accuracy, avg_val_auc = flat_evaluation(y_pred, y_true)
    avg_train_loss = total_loss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(val_dataloader)

    print(f'Train loss     : {avg_train_loss}')
    print(f'Validation loss: {avg_val_loss}')
    print(f'Accuracy: {avg_val_accuracy:.2f}')
    print(f'Auc score: {avg_val_auc:.2f}')
    print('\n')


