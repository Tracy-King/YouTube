import torch
from torch import nn

from utils.utils import MergeLayer, LinearLayer


class TemporalAttentionLayer(torch.nn.Module):
    """
  Temporal attention layer. Return the temporal embedding of a node given the node itself,
   its neighbors and the edge timestamps.
  """

    def __init__(self, n_node_features, n_node_old_embedding, n_neighbors_features, time_dim,
                 output_dimension, n_head=2,
                 dropout=0.1):
        super(TemporalAttentionLayer, self).__init__()

        self.n_head = n_head

        self.feat_dim = n_node_features
        self.embed_dim = n_node_old_embedding
        self.time_dim = time_dim

        self.query_dim = n_node_old_embedding + time_dim
        self.key_dim = n_neighbors_features + time_dim

        self.merger = MergeLayer(self.query_dim, n_node_old_embedding, n_node_features, output_dimension)

        self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim,
                                                       kdim=self.key_dim,
                                                       vdim=self.key_dim,
                                                       num_heads=n_head,
                                                       dropout=dropout)

    def forward(self, src_node_features, src_node_old_embedding, src_time_features, neighbors_features,
                neighbors_time_features, neighbors_padding_mask):
        """
    "Temporal attention model
    :param src_node_features: float Tensor of shape [batch_size, n_node_features]
    :param src_time_features: float Tensor of shape [batch_size, 1, time_dim]
    :param neighbors_features: float Tensor of shape [batch_size, n_neighbors, n_node_features]
    :param neighbors_time_features: float Tensor of shape [batch_size, n_neighbors,
    time_dim]
    :param neighbors_padding_mask: float Tensor of shape [batch_size, n_neighbors]
    :return:
    attn_output: float Tensor of shape [1, batch_size, n_node_features]
    attn_output_weights: [batch_size, 1, n_neighbors]
    """

        src_node_features_unrolled = torch.unsqueeze(src_node_features, dim=1)
        query = torch.cat([src_node_features_unrolled, src_time_features], dim=2)
        key = torch.cat([neighbors_features, neighbors_time_features], dim=2)

        # print(neighbors_features.shape, edge_features.shape, neighbors_time_features.shape)
        # Reshape tensors so to expected shape by multi head attention
        query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]
        key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]

        # Compute mask of which source nodes have no valid neighbors
        invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
        # If a source node has no valid neighbor, set it's first neighbor to be valid. This will
        # force the attention to just 'attend' on this neighbor (which has the same features as all
        # the others since they are fake neighbors) and will produce an equivalent result to the
        # original tgat paper which was forcing fake neighbors to all have same attention of 1e-10
        neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False

        attn_output, _ = self.multi_head_target(query=query, key=key, value=key,
                                                                  key_padding_mask=neighbors_padding_mask)

        attn_output = attn_output.squeeze()

        # Source nodes with no neighbors have an all zero attention output. The attention output is
        # then added or concatenated to the original source node features and then fed into an MLP.
        # This means that an all zero vector is not used.
        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)

        # Skip connection with temporal attention over neighborhood and the features of the node itself
        torch.cuda.empty_cache()
        attn_output = self.merger(attn_output, src_node_old_embedding)

        return attn_output
