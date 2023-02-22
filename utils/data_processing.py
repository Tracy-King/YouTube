import numpy as np
import random
import pandas as pd
from scipy import sparse
import json
from utils.utils import MergeLayer
import torch


class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)


def get_data_node_classification(dataset_name, tag, dataset_r1, dataset_r2, NODE_DIM, device):
  ### Load data and train val test split
  graph_df = pd.read_csv('./dynamicGraph/ml_{}.csv'.format(dataset_name))
  with open('./dynamicGraph/ml_{}.json'.format(dataset_name), 'r', encoding='UTF-8') as f:
    update_records = json.load(f)
  node_features = np.load('./dynamicGraph/ml_{}_node.npy'.format(dataset_name))
  graph_df = graph_df.fillna(0)


  val_time, test_time = list(np.quantile(graph_df.ts, [dataset_r1, dataset_r2]))

  if node_features.shape[1] != NODE_DIM:
    zipper = MergeLayer(node_features.shape[1], 0, NODE_DIM, NODE_DIM).to(device)
    node_features = zipper(torch.from_numpy(node_features).float().to(device), torch.tensor([]).to(device)).cpu().detach().numpy()


  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.index.values
  labels = graph_df.superchat.values  # label superchat

  labels = np.array([int((n+9)//10) for n in labels])

  weight = graph_df.weight.values
  length = graph_df.length.values
  timestamps = graph_df.ts.values

  edge_features = np.array([[w, l] for w, l in zip(weight, length)])


  random.seed(2020)

  train_mask = timestamps <= val_time

  test_mask = timestamps > test_time
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  return full_data, node_features, edge_features, update_records, train_data, val_data, test_data



def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()  # 记录src中node最后出现的ts
  last_timestamp_dst = dict()      # 记录dst中node最后出现的ts
  all_timediffs_src = []           # 记录src中edge离上一个edge的间隔时间，第一个是0
  all_timediffs_dst = []           # 记录dst中edge离上一个edge的间隔时间，第一个是0
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)   # src平均间隔时间
  std_time_shift_src = np.std(all_timediffs_src)     # src间隔时间标准差
  mean_time_shift_dst = np.mean(all_timediffs_dst)   # dst平均间隔时间
  std_time_shift_dst = np.std(all_timediffs_dst)     # dst间隔时间标准差

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
