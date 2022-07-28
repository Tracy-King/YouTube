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


def get_data_node_classification(dataset_name, tag, dataset_r1, dataset_r2, NODE_DIM, device, use_validation=False, binary=True):
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
  #labels = graph_df.label.values

  if tag == 'superchat':
    labels = graph_df.superchat.values  # label superchat
  else:
    labels = graph_df.membership.values  # label membership
  if binary:
    print('binary')
    labels = np.array([int((n+9)//10) for n in labels])
  #print(labels.shape, labels, np.unique(labels))

  weight = graph_df.weight.values
  length = graph_df.length.values
  timestamps = graph_df.ts.values

  edge_features = np.array([[w, l] for w, l in zip(weight, length)])

  #print('node_feature:', (np.isfinite(node_features)==False).nonzero())
  #print('edge_feature:', (np.isfinite(node_features)==False).nonzero())
  #print('timestamps:', (np.isfinite(timestamps)==False).nonzero())



  random.seed(2020)

  train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
  #train_mask = [int(x) for x in train_mask]
  test_mask = timestamps > test_time
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
  #print(len(train_mask), train_mask)
  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  return full_data, node_features, edge_features, update_records, train_data, val_data, test_data


def get_data(dataset_name, tag, dataset_r1, dataset_r2, different_new_nodes_between_val_and_test=False, randomize_features=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./dynamicGraph/ml_{}.csv'.format(dataset_name))
  with open('./dynamicGraph/ml_{}.json'.format(dataset_name), 'r', encoding='UTF-8') as f:
    update_records = json.load(f)
  node_features = np.load('./dynamicGraph/ml_{}_node.npy'.format(dataset_name))
  graph_df = graph_df.fillna(0)

  if randomize_features:
    node_features = np.random.rand(node_features.shape[0], node_features.shape[1])    # 随机初始node feature

  val_time, test_time = list(np.quantile(graph_df.ts, [dataset_r1, dataset_r2]))  # 取分位数0.7 0.85
  sources = graph_df.u.values         # node u
  destinations = graph_df.i.values    # node i
  edge_idxs = graph_df.index.values     # edge index
  if tag == 'superchat':
    labels = graph_df.superchat.values  # label superchat
  else:
    labels = graph_df.membership.values      # label membership

  timestamps = graph_df.ts.values     # time stamps
  weight = graph_df.weight.values
  length = graph_df.length.values

  edge_features = np.array([[w, l] for w, l in zip(weight, length)])

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  random.seed(2020)

  node_set = set(sources) | set(destinations)    # 并集
  n_total_unique_nodes = len(node_set)     # unique node set

  # Compute nodes which appear at test time
  test_node_set = set(sources[timestamps > val_time]).union(
    set(destinations[timestamps > val_time]))        # test node set. with timestamps > 70%
  # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
  # their edges from training
  new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

  # Mask saying for each source and destination whether they are new test nodes
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  # Mask which is true for edges with both destination and source not being new test nodes (because
  # we want to remove all edges involving any new test node)
  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)  # 不在test node中

  # For train we keep edges happening before the validation time which do not involve any new node
  # used for inductiveness
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)   # 不在test node 且time stamp < 70%

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  # define the new nodes sets for testing inductiveness of the model
  train_node_set = set(train_data.sources).union(train_data.destinations)
  assert len(train_node_set & new_test_node_set) == 0     # 交集 确保train set内不含test set的新node
  new_node_set = node_set - train_node_set

  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)   # 70% < ts < 85% 的edge
  test_mask = timestamps > test_time                                          # ts < 85% 的edge

  if different_new_nodes_between_val_and_test:
    n_new_nodes = len(new_test_node_set) // 2
    val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
    test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

    edge_contains_new_val_node_mask = np.array(
      [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
    edge_contains_new_test_node_mask = np.array(
      [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)


  else:
    edge_contains_new_node_mask = np.array(
      [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

  # validation and test with all edges
  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  # validation and test with edges that at least has one new node (not in training set)
  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])

  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask],
                            edge_idxs[new_node_test_mask], labels[new_node_test_mask])

  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
  print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))
  print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
  print("The new node test dataset has {} interactions, involving {} different nodes".format(
    new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
  print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
    len(new_test_node_set)))

  return node_features, edge_features, update_records, full_data, train_data, val_data, test_data, \
         new_node_val_data, new_node_test_data


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
