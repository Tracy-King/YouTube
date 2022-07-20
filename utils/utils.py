import numpy as np
import torch


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.PReLU()
    self.bn = torch.nn.BatchNorm1d(dim4)


    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    h = self.bn(self.act(self.fc2(h)))
    return h


class LinearLayer(torch.nn.Module):
  def __init__(self, dim1, dim2):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1, dim2)
    self.act = torch.nn.ReLU()
    self.bn = torch.nn.BatchNorm1d(dim2)

    torch.nn.init.xavier_normal_(self.fc1.weight)

  def forward(self, x):
    h = self.act(self.fc1(x))
    return self.bn(h)


class BlockLSTM(torch.nn.Module):
    def __init__(self, time_steps, num_variables, lstm_hs=64, dropout=0.8, attention=False):
      super().__init__()
      self.lstm = torch.nn.LSTM(input_size=time_steps, hidden_size=lstm_hs, num_layers=num_variables)
      self.dropout = torch.nn.Dropout(p=dropout)
      self.attn = attention

    def attention_net(self, lstm_output, final_state):
      # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
      # final_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

      batch_size = len(lstm_output)
      # hidden = final_state.view(batch_size,-1,1)
      hidden = final_state[0].unsqueeze(2)
      # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
      attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
      # attn_weights : [batch_size,n_step]
      soft_attn_weights = torch.nn.functional.softmax(attn_weights, 1)

      # context: [batch_size, n_hidden * num_directions(=2)]
      context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2))

      return context, soft_attn_weights

    def forward(self, x):
      # input is of the form (batch_size, num_variables, time_steps), e.g. (128, 1, 512)
      x = torch.transpose(x, 0, 1)
      # lstm layer is of the form (num_variables, batch_size, time_steps)
      output, (final_hidden_state, final_cell_state) = self.lstm(x)
      y = torch.transpose(output, 1, 2)
      #print('LSTM', x.shape)
      if self.attn:
        y, _ = self.attention_net(output, final_hidden_state)
      # dropout layer input shape:
      y = self.dropout(y)
      #print(y.shape)
      # output shape is of the form ()
      return y


class BlockFCNConv(torch.nn.Module):
    def __init__(self, in_channel=1, out_channel=64, kernel_size=8, momentum=0.99, epsilon=0.001, squeeze=False):
      super().__init__()
      self.conv = torch.nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size)
      self.batch_norm = torch.nn.BatchNorm1d(num_features=out_channel, eps=epsilon, momentum=momentum)
      self.relu = torch.nn.ReLU()

    def forward(self, x):
      # input (batch_size, num_variables, time_steps), e.g. (128, 1, 512)
      #print("BlockFCN", x.shape)
      x = self.conv(x)
      # input (batch_size, out_channel, L_out)
      x = self.batch_norm(x)
      # same shape as input
      y = self.relu(x)
      return y


class BlockFCN(torch.nn.Module):
  def __init__(self, time_steps, channels=[1, 64, 128, 64], kernels=[5, 3, 2], mom=0.99, eps=0.001):
    super().__init__()
    self.conv1 = BlockFCNConv(channels[0], channels[1], kernels[0], momentum=mom, epsilon=eps, squeeze=True)
    self.conv2 = BlockFCNConv(channels[1], channels[2], kernels[1], momentum=mom, epsilon=eps, squeeze=True)
    self.conv3 = BlockFCNConv(channels[2], channels[3], kernels[2], momentum=mom, epsilon=eps)
    output_size = time_steps - sum(kernels) + len(kernels)
    self.global_pooling = torch.nn.AvgPool1d(kernel_size=output_size)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    # apply Global Average Pooling 1D
    y = self.global_pooling(x)
    return y


class LSTMFCN(torch.nn.Module):
    def __init__(self, dim, num_variables=1, attn=False):
      super().__init__()
      self.lstm_block = BlockLSTM(dim, num_variables, attention=attn)
      self.fcn_block = BlockFCN(dim)
      self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
      # input is (batch_size, time_steps), it has to be (batch_size, 1, time_steps)
      x = x.unsqueeze(1)
      # pass input through LSTM block
      x1 = self.lstm_block(x)
      # pass input through FCN block
      x2 = self.fcn_block(x)
      #print('LSTMFCN', x1.shape, x2.shape)
      # concatenate blocks output
      x = torch.cat([x1, x2], 1)
      # pass through Softmax activation
      y = self.softmax(x)

      return y


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 64)
    self.fc_2 = torch.nn.Linear(64, 10)
    self.fc_3 = torch.nn.Linear(10, 2)
    torch.nn.init.xavier_normal_(self.fc_1.weight, gain=1)
    torch.nn.init.xavier_normal_(self.fc_2.weight, gain=1)
    torch.nn.init.xavier_normal_(self.fc_3.weight, gain=1)
    self.bn_1 = torch.nn.BatchNorm1d(64)
    self.bn_2 = torch.nn.BatchNorm1d(10)
    self.bn_3 = torch.nn.BatchNorm1d(2)
    self.act = torch.nn.PReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.bn_1(self.fc_1(x)))
    x = self.dropout(x)
    x = self.act(self.bn_2(self.fc_2(x)))
    x = self.dropout(x)
    x = self.act(self.bn_3(self.fc_3(x)))
    x = self.dropout(x)
    return torch.nn.functional.softmax(x, dim=1)


class LSTMCell(torch.nn.Module):
  def __init__(self, dim, hid):
    super().__init__()
    self.lstmcell = torch.nn.LSTMCell(dim, hid)  #  tensor of shape (seq length, batch size, input size) when batch_first=False

  def forward(self, x, h):
    x = torch.unsqueeze(x, dim=0)
    h = torch.unsqueeze(h, dim=0)
    hx, _ = self.lstmcell(x, (h, h))
    return hx


class GRUCell(torch.nn.Module):
  def __init__(self, dim, hid, drop=0.3):
    super().__init__()
    self.grucell = torch.nn.GRUCell(dim, hid)  #  tensor of shape (seq length, batch size, input size) when batch_first=False

  def forward(self, x, h):
    x = torch.unsqueeze(x, dim=0)
    h = torch.unsqueeze(h, dim=0)
    hx = self.grucell(x, h)
    return torch.squeeze(hx)

class RNNCell(torch.nn.Module):
  def __init__(self, dim, hid, drop=0.3):
    super().__init__()
    self.grucell = torch.nn.RNNCell(dim, hid)  #  tensor of shape (seq length, batch size, input size) when batch_first=False

  def forward(self, x, h):
    x = torch.unsqueeze(x, dim=0)
    h = torch.unsqueeze(h, dim=0)
    hx = self.grucell(x, h)
    return torch.squeeze(hx)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]    # 有向图邻接表
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []    # 行数i代表第i个node，第i行的内容代表node i 的neighbor 按照ts排序
    self.node_to_edge_idxs = []    # 行数i代表第i个node，第i行的内容代表node i 和neighbor的边 按照ts排序
    self.node_to_edge_timestamps = []     # 行数i代表第i个node，第i行的内容代表node i 和neighbor的ts 按ts排序

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])  # x[2]:timestamp 按ts排序
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):  # 找到cut_time 之前的node src_idx的所有neighbor, edge, index
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)  # 在有序array内 找到index 使得 a[index-1] < cut_time < a[index]

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def find_between(self, src_idx, cut_time, window=30):  # 找到cut_time 之前的node src_idx的所有neighbor, edge, index
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)  # 在有序array内 找到index 使得 a[index-1] < cut_time < a[index]
    j = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time-window)  # 在有序array内 找到index 使得 a[index-1] < cut_time - window < a[index]

    return self.node_to_neighbors[src_idx][j:i], self.node_to_edge_idxs[src_idx][j:i], self.node_to_edge_timestamps[src_idx][j:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """  # 对每一对source_node和timestamp，查找ts之前source_node所有edge，得到neighbor，从其中sampling 最近的20个（或随机）
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_between(source_node,
                                                   timestamp, window=300)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)  # 生产从0 到 len(source_ngh)的n_neighbors 个随机数

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()  # 返回的是数组值从小到大的索引值
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors    # 靠右对齐
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times