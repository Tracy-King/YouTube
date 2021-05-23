import pickle
import json
import codecs
import pandas as pd
import torch

pd.set_option('display.max_columns', None)

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

a = [0, 1, 2]

print(a[1:1])

root = 'tgn-attn-97DWg8tqo4M_node_classification'

data = pickle.load(open('results/{}.pkl'.format(root), 'rb'))


print(json.dumps(data, sort_keys=True, indent=4))

#data = pd.read_csv('src/sc_data3.csv')

#data.info()

#print(data[:10])
