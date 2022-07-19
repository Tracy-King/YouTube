import torch
from torch import nn
import numpy as np
import math
from utils.utils import LinearLayer

from model.temporal_attention import TemporalAttentionLayer, TemporalAttentionLayerOrigin


class EmbeddingModule(nn.Module):
  def __init__(self, node_features, edge_features, update_records, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout):
    super(EmbeddingModule, self).__init__()
    self.node_features = node_features
    self.device = device
    self.embedding_dimension = embedding_dimension
    self.node_old_embedding = self.node_features#torch.from_numpy(self.node_features.astype(np.float32)).to(self.device)
    self.edge_features = edge_features
    self.update_records = update_records
    self.update_records_idx = [int(i) for i in update_records.keys()]
    self.memory = None
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_node_old_embedding = embedding_dimension
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout

    self.LinearLayer = LinearLayer(embedding_dimension, embedding_dimension)




    #print(' self.node_old_embedding:', type( self.node_old_embedding))

  def update_old_embeddings(self, nodes, embeddings):
      for i in range(len(nodes)):
        self.node_old_embedding[nodes[i]] = embeddings[nodes[i]]

  def update_node_features(self, updated_node_raw_features):
    #self.node_features = torch.from_numpy(updated_node_raw_features.astype(np.float32)).to(self.device)
    self.node_features = updated_node_raw_features
    return

  def compute_embedding(self, memory, source_nodes, source_node_features, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    pass


class IdentityEmbedding(EmbeddingModule):
  def compute_embedding(self, memory, source_nodes, source_node_features, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, update_records, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
    super(TimeEmbedding, self).__init__(node_features, edge_features, update_records, memory,
                                        neighbor_finder, time_encoder, n_layers,
                                        n_node_features, n_edge_features, n_time_features,
                                        embedding_dimension, device, dropout)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.n_node_features)

  def compute_embedding(self, memory, source_nodes, source_node_features, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

    return source_embeddings


class GraphEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, update_records, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=False, original_encoder=False):
    super(GraphEmbedding, self).__init__(node_features, edge_features, update_records,
                                         neighbor_finder, time_encoder, n_layers,
                                         n_node_features, n_edge_features, n_time_features,
                                         embedding_dimension, device, dropout)

    self.use_memory = use_memory
    self.device = device


  def compute_embedding(self, source_nodes, source_node_raw_features, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True, memory=None):
    """Recursive implementation of curr_layers temporal graph attention layers.

    src_idx_l [batch_size]: users / items input ids.
    cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
    curr_layers [scalar]: number of temporal convolutional layers to stack.
    num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
    """

    assert (n_layers >= 0)

    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

    # query node always has the start time -> time span == 0
    source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
      timestamps_torch))
    #source_node_features = self.node_features[source_nodes_torch, :]
    source_node_features = source_node_raw_features

    if self.use_memory:
      source_node_features = memory[source_nodes, :] + source_node_features

    if n_layers == 0:
      return source_node_features
    else:
      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps,
        n_neighbors=n_neighbors)

      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

      #edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

      edge_deltas = timestamps[:, np.newaxis] - edge_times

      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      neighbors = neighbors.flatten()
      neighbor_embeddings = self.compute_embedding(neighbors,
                                                   torch.from_numpy(self.node_features[neighbors, :].astype(np.float32)).to(self.device),
                                                   np.repeat(timestamps, n_neighbors),
                                                   n_layers=n_layers - 1,
                                                   n_neighbors=n_neighbors)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)

      #print('type of edge_features', type(self.edge_features))
      edge_features = torch.from_numpy(self.edge_features[edge_idxs, :].astype(np.float32)).to(self.device)

      mask = neighbors_torch == 0

      source_embedding = self.aggregate_origin(n_layers, source_node_features,
                                        source_nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        edge_features,
                                        mask)

      return source_embedding

  def compute_embeddingv2(self, source_nodes, source_node_raw_features, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True, memory=None):
    """Recursive implementation of curr_layers temporal graph attention layers.

    src_idx_l [batch_size]: users / items input ids.
    cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
    curr_layers [scalar]: number of temporal convolutional layers to stack.
    num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
    """

    assert (n_layers >= 0)

    #source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

    # query node always has the start time -> time span == 0
    source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
      timestamps_torch))
    #source_node_features = self.node_features[source_nodes_torch, :]
    source_node_old_embedding = torch.from_numpy(self.node_old_embedding[source_nodes, :].astype(np.float32)).to(self.device)
    source_node_features = source_node_raw_features
    '''
    if self.use_memory:
      source_node_features = memory[source_nodes, :] + source_node_raw_features
    '''
    if n_layers == 0:
      return source_node_old_embedding
    else:
      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps,
        n_neighbors=n_neighbors)

      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

      #edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

      edge_deltas = timestamps[:, np.newaxis] - edge_times

      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      neighbors = neighbors.flatten()
      neighbor_embeddings = self.compute_embeddingv2(source_nodes=neighbors,
                                                   source_node_raw_features=torch.from_numpy(self.node_features[neighbors, :].astype(np.float32)).to(self.device),
                                                   timestamps=np.repeat(timestamps, n_neighbors),
                                                   n_layers=n_layers - 1,
                                                   n_neighbors=n_neighbors)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)

      #print(neighbor_embeddings.shape, source_node_features.shape)
      #for i in range(neighbor_embeddings.shape[0]):
      #  neighbor_embeddings[i] = torch.sub(neighbor_embeddings[i], source_node_features[i])

      if (torch.isfinite(neighbor_embeddings) == False).nonzero().shape[0] != 0:
          print("inf detected", neighbor_embeddings, (torch.isfinite(neighbor_embeddings) == False).nonzero().shape[0])
          neighbor_embeddings = torch.nan_to_num(neighbor_embeddings, nan=0.0, posinf=1.0, neginf=0.0)




      #print('type of edge_features', type(self.edge_features))
      edge_features = torch.from_numpy(self.edge_features[edge_idxs, :].astype(np.float32)).to(self.device)

      mask = neighbors_torch == 0

      source_embedding = self.aggregate(n_layers,
                                        source_node_features,
                                        source_node_old_embedding,
                                        source_nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        edge_features,
                                        mask)

      return source_embedding

  def aggregate(self, n_layers, source_node_features, source_node_old_embedding, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    return None

  def aggregate_origin(self, n_layers, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings, edge_time_embeddings, edge_features, mask):
    return None


class GraphSumEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, update_records, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                            edge_features=edge_features,
                                            update_records= update_records,
                                            neighbor_finder=neighbor_finder,
                                            time_encoder=time_encoder, n_layers=n_layers,
                                            n_node_features=n_node_features,
                                            n_edge_features=n_edge_features,
                                            n_time_features=n_time_features,
                                            embedding_dimension=embedding_dimension,
                                            device=device,
                                            n_heads=n_heads, dropout=dropout,
                                            use_memory=use_memory)
    self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +
                                                         n_edge_features, embedding_dimension)
                                         for _ in range(n_layers)])
    self.linear_2 = torch.nn.ModuleList(
      [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                       embedding_dimension) for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_node_old_embedding, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                   dim=2)
    neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
    neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

    source_features = torch.cat([source_node_features,
                                 source_nodes_time_embedding.squeeze()], dim=1)
    source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
    source_embedding = self.linear_2[n_layer - 1](source_embedding)

    return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, update_records, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, original_encoder=False):
    super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, update_records,
                                                  neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout,
                                                  use_memory)


    if original_encoder:
      self.attention_models = torch.nn.ModuleList([TemporalAttentionLayerOrigin(
        n_node_features=n_node_features,
        n_neighbors_features=n_node_features,
        n_edge_features=n_edge_features,
        time_dim=n_time_features,
        n_head=n_heads,
        dropout=dropout,
        output_dimension=embedding_dimension)
        for _ in range(n_layers)])
    else:
      self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
        n_node_features=n_node_features,
        n_node_old_embedding=self.n_node_old_embedding,
        n_neighbors_features=n_node_features,
        n_edge_features=n_edge_features,
        time_dim=n_time_features,
        n_head=n_heads,
        dropout=dropout,
        output_dimension=embedding_dimension)
        for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_node_old_embedding, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    attention_model = self.attention_models[n_layer - 1]

    source_embedding, _ = attention_model(source_node_features,
                                          source_node_old_embedding,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask)

    return source_embedding

  def aggregate_origin(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
      attention_model = self.attention_models[n_layer - 1]

      source_embedding, _ = attention_model(source_node_features,
                                            source_nodes_time_embedding,
                                            neighbor_embeddings,
                                            edge_time_embeddings,
                                            edge_features,
                                            mask)

      return source_embedding

def get_embedding_module(module_type, node_features, edge_features, update_records, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=False, original_encoder=False):
  if module_type == "graph_attention":
    return GraphAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    update_records=update_records,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory, original_encoder=original_encoder)
  elif module_type == "graph_sum":
    return GraphSumEmbedding(node_features=node_features,
                              edge_features=edge_features,
                              update_records=update_records,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory)

  elif module_type == "identity":
    return IdentityEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             update_records=update_records,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout)
  elif module_type == "time":
    return TimeEmbedding(node_features=node_features,
                         edge_features=edge_features,
                         update_records=update_records,
                         neighbor_finder=neighbor_finder,
                         time_encoder=time_encoder,
                         n_layers=n_layers,
                         n_node_features=n_node_features,
                         n_edge_features=n_edge_features,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         dropout=dropout,
                         n_neighbors=n_neighbors)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))


