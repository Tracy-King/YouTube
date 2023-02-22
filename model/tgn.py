import logging
import numpy as np
import torch

from utils.utils import MergeLayer
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode

import os


class TGN(torch.nn.Module):
    def __init__(self, neighbor_finder, node_features, edge_features, update_records, device, data, n_layers=2,
                 n_heads=2, dropout=0.1, use_memory=False, embedding_dim=128,
                 mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
                 std_time_shift_dst=1, n_neighbors=None, dyrep=False):
        super(TGN, self).__init__()

        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.data = data

        self.update_records = update_records
        self.update_records_idx = [int(key) for key in update_records]
        self.update_records_idx = np.array(sorted(self.update_records_idx))

        self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

        self.n_node_features = self.node_raw_features.shape[1]
        self.n_edge_features = self.n_node_features
        self.n_nodes = self.node_raw_features.shape[0]
        self.n_edges = self.edge_raw_features.shape[0]
        self.embedding_dimension = embedding_dim
        self.n_neighbors = n_neighbors
        self.dyrep = dyrep
        self.last_updated = np.zeros(self.n_nodes)

        self.use_memory = use_memory
        self.time_encoder = TimeEncode(dimension=self.n_node_features)

        self.embedding_dict = dict()
        self.last_updated_dict = np.zeros(self.n_nodes)

        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst


        self.embedding_module = get_embedding_module(node_features=self.node_raw_features,
                                                     edge_features=self.edge_raw_features,
                                                     update_records=self.update_records,
                                                     neighbor_finder=self.neighbor_finder,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     n_node_features=self.n_node_features,
                                                     n_edge_features=self.n_edge_features,
                                                     n_time_features=self.n_node_features,
                                                     embedding_dimension=self.embedding_dimension,
                                                     device=self.device,
                                                     n_heads=n_heads, dropout=dropout,
                                                     n_neighbors=self.n_neighbors)

        # MLP to compute probability on an edge given two node embeddings
        self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                         self.n_node_features,
                                         1)

    def edge_feature_initiate(self):
        if os.path.exists('./dynamicGraph/edge_feature_{}.npy'.format(self.data)):
            self.edge_raw_features = np.load('./dynamicGraph/edge_feature_{}.npy'.format(self.data))
            torch.from_numpy(self.edge_raw_features.astype(np.float32)).to(self.device)
            self.n_edge_feature = self.edge_raw_features.shape[1]
            print("Edge feature loaded!")
        else:
            print('Edge feature not founded. Computing...')
            batch_size = 100000
            result = 0
            for batch in range(0, self.n_edges, batch_size):
                print("{}/{}".format(batch, self.n_edges))
                end = min(self.n_edges, batch + batch_size)
                n_batch = end - batch
                input_data = torch.from_numpy(self.edge_raw_features[batch:end].astype(np.float32)).to(self.device)
                if batch == 0:
                    result = self.edge_encoder(input_data).view(n_batch, -1)
                    # result = result.cpu().detach().numpy()
                else:
                    tmp = self.edge_encoder(input_data).view(n_batch, -1)
                    result = torch.cat((result, tmp), dim=0)
                    # result = np.vstack((result, tmp))

            self.edge_raw_features = result  # torch.from_numpy(result.astype(np.float32)).to(self.device)
            # self.n_edge_features = self.edge_raw_features.shape[1]
            torch.cuda.empty_cache()
            np.save('./dynamicGraph/edge_feature_{}.npy'.format(self.data), result)
            print("Edge feature saved!")

        assert self.edge_raw_features.shape[0] == self.n_edges

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
                embeddings = np.vstack((embeddings, v))

        self.embedding_module.update_old_embeddings(np.array(nodes), embeddings)
        self.embedding_dict.clear()
        return

    def update_node_features(self, node_idx, records_idx):
        self.node_raw_features[node_idx] = torch.from_numpy(
            np.array(self.update_records[str(records_idx)]).astype(np.float32)).to(self.device)
        return

    def compute_temporal_embeddings(self, source_nodes, destination_nodes, edge_times,
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

        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes])
        positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times])

        ### Compute differences between the time the memory of a node was last updated,
        ### and the time for which we want to compute the embedding of a node
        source_time_diffs = torch.tensor(edge_times - self.last_updated_dict[
            source_nodes]).to(self.device)
        destination_time_diffs = torch.tensor(edge_times - self.last_updated_dict[
            destination_nodes]).to(self.device)

        time_diffs = torch.cat([source_time_diffs, destination_time_diffs], dim=0)

        start_idx = np.searchsorted(self.update_records_idx, edge_idxs[0])  # The first update record index
        end_idx = np.searchsorted(self.update_records_idx, edge_idxs[-1],
                                  side='right') - 1  # The last update record index

        if start_idx <= end_idx:
            if start_idx == end_idx:
                updates_idx = [self.update_records_idx[start_idx]]
            else:
                updates_idx = self.update_records_idx[start_idx: end_idx]  # edge index

            update_nodes_idx = [int(np.searchsorted(edge_idxs, i)) for i in
                                updates_idx]  # node index of the corrsponding edge
            n_updates = len(updates_idx)

            if n_updates == 0:
                source_node_features = self.node_raw_features[source_nodes, :]
                destination_node_features = self.node_raw_features[destination_nodes, :]
            else:
                source_node_features = self.node_raw_features[source_nodes[:min(n_samples, update_nodes_idx[0])], :]
                destination_node_features = self.node_raw_features[
                                            destination_nodes[:min(n_samples, update_nodes_idx[0])], :]

                for i in range(n_updates - 1):
                    start_slice = update_nodes_idx[i]
                    end_slice = update_nodes_idx[i + 1]
                    self.update_node_features(source_nodes[start_slice], updates_idx[i])

                    source_node_features = torch.cat(
                        (source_node_features, self.node_raw_features[source_nodes[start_slice:end_slice], :]), 0)
                    destination_node_features = torch.cat(
                        (
                        destination_node_features, self.node_raw_features[destination_nodes[start_slice:end_slice], :]),
                        0)

                self.update_node_features(source_nodes[update_nodes_idx[-1]], updates_idx[-1])
                if update_nodes_idx[-1] < n_samples:
                    start_slice = update_nodes_idx[-1]
                    source_node_features = torch.cat(
                        (source_node_features, self.node_raw_features[source_nodes[start_slice:], :]), 0)
                    destination_node_features = torch.cat(
                        (destination_node_features, self.node_raw_features[destination_nodes[start_slice:], :]), 0)

            assert len(source_node_features) == n_samples, 'source_node_feature error'
            assert len(destination_node_features) == n_samples, 'destination_node_feature error'

        else:
            source_node_features = self.node_raw_features[source_nodes, :]
            destination_node_features = self.node_raw_features[destination_nodes, :]

        node_features = torch.cat((source_node_features, destination_node_features), 0)

        assert (node_features.shape[0] == 2 * n_samples), 'node_features dimension 0 error'
        assert (node_features.shape[1] == self.n_node_features), 'node_features dimension 1 error'

        self.embedding_module.update_node_features(updated_node_raw_features=self.node_raw_features)

        # Compute the embeddings using the embedding module

        node_embedding = self.embedding_module.compute_embedding(source_node_raw_features=node_features,
                                                                 source_nodes=nodes,
                                                                 timestamps=timestamps,
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=n_neighbors,
                                                                 time_diffs=time_diffs)

        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples:]
        with torch.no_grad():
            for idx in range(len(source_nodes) - 1, -1, -1):
                if source_nodes[idx] not in self.embedding_dict.keys():
                    self.embedding_dict[source_nodes[idx]] = source_node_embedding[idx]
                if destination_nodes[idx] not in self.embedding_dict.keys():
                    self.embedding_dict[destination_nodes[idx]] = destination_node_embedding[idx]

            self.embedding_module.update_old_embeddings(np.unique(np.concatenate((source_nodes, destination_nodes))),
                                                        self.embedding_dict)
            self.embedding_dict.clear()

        for i, ts in zip(nodes, timestamps):
            self.last_updated_dict[i] = ts

        return source_node_embedding

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder
