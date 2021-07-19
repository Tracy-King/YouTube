import pickle
import json
import codecs
import pandas as pd
import torch
import numpy as np

pd.set_option('display.max_columns', None)

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

a = [0, 1, 2]

print(np.tile(a, (2, 1)))


a = torch.tensor([0, float('nan'), float('inf'), float('-inf')])
b = torch.tensor([0, 1, 2, 3])

t = torch.isfinite(b)

print(t)

print(torch.nonzero(t))

print((t==False).nonzero().shape[0])

#data = pd.read_csv('src/sc_data3.csv')

#data.info()

#print(data[:10])
