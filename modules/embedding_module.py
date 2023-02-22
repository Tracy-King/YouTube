import torch
from torch import nn
import numpy as np
from utils.utils import LinearLayer

from model.temporal_attention import TemporalAttentionLayer


class EmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, update_records, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 dropout):
        super(EmbeddingModule, self).__init__()
        self.node_features = node_features
        self.device = device
        self.embedding_dimension = embedding_dimension
        self.node_old_embedding = self.node_features
        self.edge_features = edge_features
        self.update_records = update_records
        self.update_records_idx = [int(i) for i in update_records.keys()]
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_node_old_embedding = embedding_dimension
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dict = dict()

        self.LinearLayer = LinearLayer(embedding_dimension, embedding_dimension)

    def update_old_embeddings(self, nodes, embeddings):
        for i in range(len(nodes)):
            self.node_old_embedding[nodes[i]] = embeddings[nodes[i]]

    def update_node_features(self, updated_node_raw_features):
        self.node_features = updated_node_raw_features
        return

    def compute_embedding(self, source_nodes, source_node_features, timestamps, n_layers, n_neighbors=20,
                          time_diffs=None,
                          use_time_proj=True):
        pass


class GraphEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, update_records, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1):
        super(GraphEmbedding, self).__init__(node_features, edge_features, update_records,
                                             neighbor_finder, time_encoder, n_layers,
                                             n_node_features, n_edge_features, n_time_features,
                                             embedding_dimension, device, dropout)

        self.device = device

    def compute_embedding(self, source_nodes, source_node_raw_features, timestamps, n_layers, n_neighbors=20,
                          time_diffs=None,
                          use_time_proj=True):
        """Recursive implementation of curr_layers temporal graph attention layers.

    src_idx_l [batch_size]: users / items input ids.
    cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
    curr_layers [scalar]: number of temporal convolutional layers to stack.
    num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
    """

        assert (n_layers >= 0)

        timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

        # query node always has the start time -> time span == 0
        source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
            timestamps_torch))

        source_node_old_embedding = self.node_old_embedding[source_nodes, :]
        source_node_features = source_node_raw_features

        if n_layers == 0:
            return source_node_old_embedding
        else:
            neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
                source_nodes,
                timestamps,
                n_neighbors=n_neighbors)

            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
            edge_features = self.edge_features[edge_idxs, :]
            edge_features = torch.mul(edge_features[:, :, 0], edge_features[:, :, 1])
            edge_weight = torch.tile(torch.unsqueeze(edge_features, dim=2), (1, 1, self.n_node_features))

            edge_deltas = timestamps[:, np.newaxis] - edge_times

            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

            neighbors = neighbors.flatten()

            neighbor_embeddings = self.compute_embedding(source_nodes=neighbors,
                                                         source_node_raw_features=self.node_features[neighbors, :],
                                                         timestamps=np.repeat(timestamps, n_neighbors),
                                                         n_layers=n_layers - 1,
                                                         n_neighbors=n_neighbors)

            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)

            for i in range(neighbor_embeddings.shape[0]):
                neighbor_embeddings[i] = torch.mul(torch.sub(neighbor_embeddings[i], source_node_old_embedding[i]),
                                                   edge_weight[i])
                # neighbor_embeddings[i] = torch.sub(neighbor_embeddings[i], source_node_old_embedding[i])

            edge_time_embeddings = self.time_encoder(edge_deltas_torch)

            mask = neighbors_torch == 0
            torch.cuda.empty_cache()
            source_embedding = self.aggregate(n_layers,
                                              source_node_features,
                                              source_node_old_embedding,
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              mask)

            return source_embedding

    def aggregate(self, n_layers, source_node_features, source_node_old_embedding, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, mask):
        return None


class GraphAttentionEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, update_records, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1):
        super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, update_records,
                                                      neighbor_finder, time_encoder, n_layers,
                                                      n_node_features, n_edge_features,
                                                      n_time_features,
                                                      embedding_dimension, device,
                                                      n_heads, dropout)

        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            n_node_features=n_node_features,
            n_node_old_embedding=self.n_node_old_embedding,
            n_neighbors_features=n_node_features,
            time_dim=n_time_features,
            n_head=n_heads,
            dropout=dropout,
            output_dimension=embedding_dimension)
            for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_node_old_embedding, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, mask):
        attention_model = self.attention_models[n_layer - 1]

        source_embedding = attention_model(source_node_features,
                                              source_node_old_embedding,
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              mask)

        return source_embedding


def get_embedding_module(node_features, edge_features, update_records, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1):
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
                                       n_heads=n_heads, dropout=dropout)
