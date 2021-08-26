import logging
import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode, EdgeEncode

import os


class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, update_records, device, data, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False, original_encoder=False):
    super(TGN, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)
    self.data = data

    self.update_records = update_records
    #print('update_records length:', len(update_records))
    self.update_records_idx = [int(key) for key in update_records]

    #print('self.update_record_idx:', self.update_records_idx, type(self.update_records_idx))
    self.update_records_idx = np.array(sorted(self.update_records_idx))

    #self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    #self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)
    self.node_raw_features = node_features
    self.edge_raw_features = edge_features


    self.n_node_features = self.node_raw_features.shape[1]
    self.n_edge_features = self.n_node_features
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edges = self.edge_raw_features.shape[0]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep
    self.original_encoder = original_encoder

    self.use_memory = use_memory
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.edge_encoder = EdgeEncode(dimension=self.n_node_features)
    self.edge_encoder.to(device)

    self.edge_feature_initiate()
    #print("type of self.edge_raw_feature", type(self.edge_raw_features))
    self.embedding_dict = dict()

    self.memory = None

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst

    if self.use_memory:
      self.memory_dimension = memory_dimension
      self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                              self.time_encoder.dimension
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device)

    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 update_records=self.update_records,
                                                 memory=self.memory,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors,
                                                 original_encoder=self.original_encoder)

    # MLP to compute probability on an edge given two node embeddings
    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                     self.n_node_features,
                                     1)

  def edge_feature_initiate(self):
    if os.path.exists('./dynamicGraph/edge_feature_{}.npy'.format(self.data)):
      self.edge_raw_features = np.load('./dynamicGraph/edge_feature_{}.npy'.format(self.data))
      self.n_edge_feature = self.edge_raw_features.shape[1]
      print("Edge feature loaded!")
    else:
      print('Edge feature not founded. Computing...')
      batch_size = 100000
      result = 0
      for batch in range(0, self.n_edges, batch_size):
        print("{}/{}".format(batch, self.n_edges))
        end = min(self.n_edges, batch+batch_size)
        n_batch = end - batch
        input_data = torch.from_numpy(self.edge_raw_features[batch:end].astype(np.float32)).to(self.device)
        if batch == 0:
          result = self.edge_encoder(input_data).view(n_batch, -1)
          result = result.cpu().detach().numpy()
        else:
          tmp = self.edge_encoder(input_data).view(n_batch, -1)
          tmp = tmp.cpu().detach().numpy()
          #result = torch.cat((result, tmp), dim=0)
          result = np.vstack((result, tmp))

      self.edge_raw_features = result
      self.n_edge_features = self.edge_raw_features.shape[1]
      torch.cuda.empty_cache()
      np.save('./dynamicGraph/edge_feature_{}.npy'.format(self.data), self.edge_raw_features)
      print("Edge feature saved!")

    assert self.edge_raw_features.shape[0] == self.n_edges
    #print("n_edges:", self.n_edges, "n_edge_feature:", self.n_edge_features)

  def update_old_embeddings(self):
    nodes = []
    embeddings = 0
    flag = 1
    for k, v in self.embedding_dict.items():
      nodes.append(k)
      if flag == 1:
        embeddings = v
        flag -= 1
      else:
        #embeddings = torch.cat([embeddings, v], dim=0)
        embeddings = np.vstack((embeddings, v))

    self.embedding_module.update_old_embeddings(np.array(nodes), embeddings)
    self.embedding_dict.clear()
    return

  def update_node_features(self, node_idx, records_idx):
    self.node_raw_features[node_idx] = np.array(self.update_records[str(records_idx)]).astype(np.float32)
    return

  def compute_temporal_embeddings_origin(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs, n_neighbors=20, use_memory=True):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.
    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    memory = None
    time_diffs = None
    if use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages)
      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update

      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        negative_nodes].long()
      negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                             dim=0)
    node_raw_feature = torch.from_numpy(self.node_raw_features[nodes, :].astype(np.float32)).to(self.device)
    # Compute the embeddings using the embedding module
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             source_node_raw_features=node_raw_feature,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    if use_memory:
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)
        self.update_memory(positives, self.memory.messages)

        assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
          "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(positives)

      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      if self.memory_update_at_start:
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        negative_node_embedding = memory[negative_nodes]

    return source_node_embedding, destination_node_embedding, negative_node_embedding

  def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs, n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    if self.original_encoder:
      return self.compute_temporal_embeddings_origin(source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs, n_neighbors, use_memory=True)
    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages)
      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update

      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        negative_nodes].long()
      negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                             dim=0)

    start_idx = np.searchsorted(self.update_records_idx, edge_idxs[0])  # 第一个update record index
    end_idx = np.searchsorted(self.update_records_idx, edge_idxs[-1], side='right') - 1  # 最后一个update record index

    if start_idx <= end_idx:
      if start_idx == end_idx:
        updates_idx = [self.update_records_idx[start_idx]]
      else:
        updates_idx = self.update_records_idx[start_idx: end_idx]    # edge index
      #print('update_idx:', updates_idx)
      #print(edge_idxs)
      update_nodes_idx = [int(np.searchsorted(edge_idxs, i)) for i in updates_idx]   # edge对应的node list内的index
      #print('update_nodes_idx:', update_nodes_idx)
      n_updates = len(updates_idx)

      if n_updates == 0:
        source_node_features = self.node_raw_features[source_nodes, :]
        destination_node_features = self.node_raw_features[destination_nodes, :]
        negative_node_features = self.node_raw_features[negative_nodes, :]
      else:
        source_node_features = self.node_raw_features[source_nodes[:min(n_samples, update_nodes_idx[0])], :]
        destination_node_features = self.node_raw_features[destination_nodes[:min(n_samples, update_nodes_idx[0])], :]
        negative_node_features = self.node_raw_features[negative_nodes[:min(n_samples, update_nodes_idx[0])], :]

        for i in range(n_updates-1):
          start_slice = update_nodes_idx[i]
          end_slice = update_nodes_idx[i+1]
          self.update_node_features(source_nodes[start_slice], updates_idx[i])

          source_node_features = np.vstack(
            (source_node_features, self.node_raw_features[source_nodes[start_slice:end_slice], :]))
          destination_node_features = np.vstack(
            (destination_node_features, self.node_raw_features[destination_nodes[start_slice:end_slice], :]))
          negative_node_features = np.vstack(
            (negative_node_features, self.node_raw_features[negative_nodes[start_slice:end_slice], :]))

      #self.node_raw_features[source_nodes[update_nodes_idx[-1]]] = updates[update_nodes_idx[-1]]
      #print(source_nodes[update_nodes_idx])
          self.update_node_features(source_nodes[update_nodes_idx[-1]], updates_idx[-1])
        if update_nodes_idx[-1] < n_samples:
          start_slice = update_nodes_idx[-1]
          source_node_features = np.vstack(
            (source_node_features, self.node_raw_features[source_nodes[start_slice:], :]))
          destination_node_features = np.vstack(
            (destination_node_features, self.node_raw_features[destination_nodes[start_slice:], :]))
          negative_node_features = np.vstack(
            (negative_node_features, self.node_raw_features[negative_nodes[start_slice:], :]))
      #print(len(source_node_features))
      assert len(source_node_features) == n_samples, 'source_node_feature error'
      assert len(destination_node_features) == n_samples, 'destination_node_feature error'
      assert len(negative_node_features) == n_samples, 'negative_node_features error'

    else:
      source_node_features = self.node_raw_features[source_nodes, :]
      destination_node_features = self.node_raw_features[destination_nodes, :]
      negative_node_features = self.node_raw_features[negative_nodes, :]

    #print(type(source_node_features), type(destination_node_features), type(negative_node_features))
    source_node_features = torch.from_numpy(source_node_features.astype(np.float32)).to(self.device)
    destination_node_features = torch.from_numpy(destination_node_features.astype(np.float32)).to(self.device)
    negative_node_features = torch.from_numpy(negative_node_features.astype(np.float32)).to(self.device)
    node_features = torch.cat([source_node_features, destination_node_features, negative_node_features], 0)

    assert (node_features.shape[0] == 3*n_samples), 'node_features dimension 0 error'
    assert (node_features.shape[1] == self.n_node_features), 'node_features dimension 1 error'

    self.embedding_module.update_node_features(updated_node_raw_features=self.node_raw_features)
    #print(type(node_features))
    # Compute the embeddings using the embedding module
    node_embedding = self.embedding_module.compute_embeddingv2(memory=memory,
                                                             source_node_raw_features=node_features,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    for idx in range(len(source_nodes)-1, -1, -1):
      if source_nodes[idx] not in self.embedding_dict.keys():
        self.embedding_dict[source_nodes[idx]] = source_node_embedding[idx].clone().detach().cpu().numpy()
      if destination_nodes[idx] not in self.embedding_dict.keys():
        self.embedding_dict[destination_nodes[idx]] = destination_node_embedding[idx].clone().detach().cpu().numpy()
    #print(self.embedding_dict.keys())
    self.embedding_module.update_old_embeddings(np.unique(np.concatenate([source_nodes, destination_nodes])), self.embedding_dict)
    self.embedding_dict.clear()

    if self.use_memory:
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)
        self.update_memory(positives, self.memory.messages)

        assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
          "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(positives)
      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)


      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      if self.memory_update_at_start:
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        negative_node_embedding = memory[negative_nodes]


    return source_node_embedding, destination_node_embedding, negative_node_embedding



  def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                 edge_idxs, n_neighbors=20):
    """
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """
    n_samples = len(source_nodes)
    if self.original_encoder:
      source_node_embedding, destination_node_embedding, negative_node_embedding = \
        self.compute_temporal_embeddings_origin( source_nodes, destination_nodes, negative_nodes,
                                                 edge_times, edge_idxs, n_neighbors)
    else:
      source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
      source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

    score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                torch.cat([destination_node_embedding,
                                           negative_node_embedding])).squeeze(dim=0)
    pos_score = score[:n_samples]
    neg_score = score[n_samples:]


    return pos_score.sigmoid(), neg_score.sigmoid()

  def update_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)

  def get_updated_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_memory, updated_last_update

  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = torch.from_numpy(self.edge_raw_features[edge_idxs].astype(np.float32)).to(self.device)

    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
