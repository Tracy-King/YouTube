import pickle
import json
import codecs
import pandas as pd
import torch
import numpy as np
import re
import difflib
from torch.autograd import Variable
import datetime

pd.set_option('display.max_columns', None)

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

dataset_name = 'concat_full_v3.10'
dataset_name2 = 'concat_week_v3.10'

size = 5
labels_batch_torch = torch.tensor([0, 1, 1, 0, 1])
ones = torch.sparse.torch.eye(2)
labels_batch_onehot = ones.index_select(0, labels_batch_torch)
print(labels_batch_onehot)
'''
a = [0, 1, 2]

print(np.tile(a, (2, 1)))

b = np.linspace(0, 9, 20)
print('b1:{}'.format(b))
b = 1 / 10 ** b
print('b2:{}'.format(b))

b = np.tile(b, (2, 1)).T

print('b3:{}'.format(b))

b = np.cos(b)

print('b4:{}'.format(b))

video_id = '97DWg8tqo4M'
new_data = pd.read_pickle('dynamicGraph/{}_dynamic_graph.pkl'.format(video_id))
test = new_data
#end_time = new_data['Offset'].values[-1]
#print(end_time, type(end_time))

for (i, j), k in zip(new_data[10:20].iterrows(), range(10)):
  print(i, j, k)


class args():

    def __init__(self):
        self.dim = 0
        self.length = 1

args = args()
print(args.dim)

history_list = list(range(10,0,-1))
delete_list = [2, 3, 5]
print(history_list)
history_list = [history_list[idx] for idx in range(len(history_list)) if idx not in delete_list]
print(history_list)
'''
'''

dataset_name = 'concat_v2'
graph_df = pd.read_csv('./dynamicGraph/ml_{}.csv'.format(dataset_name))
print(graph_df['superchat'].value_counts())
print(graph_df['membership'].value_counts())



print(0>=0.0)
tst = 'なるはや待機� aqua ❤ エペかな、モンハンかな、🥳'
a = '許されたｗ'
b = '許されてるｗ'
c = '許された'
a = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a\uAC00-\uD7AF\u3040-\u31FF])","",a)
b = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a\uAC00-\uD7AF\u3040-\u31FF])","",b)
c = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a\uAC00-\uD7AF\u3040-\u31FF])","",c)

r1 = difflib.SequenceMatcher(None, a, b).real_quick_ratio()
r2 = difflib.SequenceMatcher(None, a, c).real_quick_ratio()

print(a, b, c)
print(r1, r2)
#print(subtst)
'''
#data = pd.read_csv('src/sc_data3.csv')

#data.info()

#print(data[:10])
