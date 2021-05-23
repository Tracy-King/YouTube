import torch
import numpy as np


class TimeEncode(torch.nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(1, dimension)

    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                       .float().reshape(dimension, -1))
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    t = t.unsqueeze(dim=2)

    # output has shape [batch_size, seq_len, dimension]
    output = torch.cos(self.w(t))

    return output



class EdgeEncode(torch.nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(EdgeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(2, dimension, bias=False)

    self.w.weight = torch.nn.Parameter((torch.from_numpy(np.tile((1 / 10 ** np.linspace(0, 9, dimension)), (2, 1)).T))
                                       .float())

  def forward(self, t):
    output = torch.cos(self.w(t))

    return output
