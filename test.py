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
import pandas as pd
from collections import Counter


pd.set_option('display.max_columns', None)

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


source_embedding = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])

b = (torch.isfinite(source_embedding) == False).nonzero().shape[0]
print(b)

print((torch.isfinite(source_embedding) == False).nonzero().shape[0] != 0)






concat_list = ['ON3WijEIS1c', 'qO8Ld-qLjb0', 'k3Nzow_OqQY', 'y3DCfZmX8iA',
               'qHZwDxea7fQ']  # , 'cibdBr9TkEo', 'rW8jSXVsW2E', 'eIi8zCPFyng', 'wtJj3CO_YR0']
#                ['97DWg8tqo4M', 'sXnTgUkXqEE', 'zl5P5lAvLwM', 'GsgbCSC6d50', 'TDXBiMKQZpI', 'fkWB_8Yyt0A', '8QEhoC-DOjM', 'DaT7j74W7zw', '1kxCz6tt2MU']
# concat_list = ['fkWB_8Yyt0A', '8QEhoC-DOjM', 'DaT7j74W7zw', '1kxCz6tt2MU']

# concat_list = ['ON3WijEIS1c', 'qO8Ld-qLjb0', 'k3Nzow_OqQY', 'y3DCfZmX8iA', 'qHZwDxea7fQ']#, 'cibdBr9TkEo', 'rW8jSXVsW2E', 'eIi8zCPFyng', 'wtJj3CO_YR0']
#                ['97DWg8tqo4M', 'sXnTgUkXqEE', 'zl5P5lAvLwM', 'GsgbCSC6d50', 'TDXBiMKQZpI', 'fkWB_8Yyt0A', '8QEhoC-DOjM', 'DaT7j74W7zw', '1kxCz6tt2MU']
# concat_list = ['fkWB_8Yyt0A', '8QEhoC-DOjM', 'DaT7j74W7zw', '1kxCz6tt2MU']
'''
data = 0  # pd.read_pickle('../dynamicGraph/concat_full_v3_tmp.pkl')
cnt = 1
end_time = 0  # data['Offset'].iat[-1].to_numpy()
for id in concat_list:
    new_data = pd.read_csv('./embedding/UC1opHUrw8rvnsadT-iGp7Cg/{}.csv'.format(id))
    if cnt == 1:
        print('first:{}-{}'.format(cnt, id))
        cnt += 1
        data = new_data
        end_time = data['offset'].iat[-1]
        # print(new_data['Offset'].to_numpy()[:10])
    else:
        print('next:{}-{}'.format(cnt, id))
        cnt += 1
        new_data['offset'] = new_data['offset'].add(end_time + 3600)
        # print(new_data['Offset'].to_numpy()[:10])
        data = data.append(new_data, ignore_index=True)
        end_time = data['offset'].iat[-1]
        # print(end_time)

print(data.info())
data.to_csv('./concat_week_v3.10_tmp3.csv')


dataset_name = 'concat_week_v3.10_tmp3'
graph_df = pd.read_csv('./{}.csv'.format(dataset_name))
#print(graph_df.info())
#graph_df = pd.read_csv('./dynamicGraph/ml_{}.csv'.format(dataset_name))
#print(graph_df.info())

with open('val.pkl', 'rb') as file:
    val_data = pickle.loads(file.read())
pred_labels = np.load('pred_label.npy')


superchats = graph_df[(graph_df['superchat']>0) & (graph_df['offset']>=val_data.timestamps[0])]
#print(superchats.info(), superchats)
#superchats.to_csv('./{}-superchats.csv'.format(dataset_name))


#pos_idx = np.nonzero(np.logical_and(pred_labels, val_data.labels))
pos_idx = np.nonzero(val_data.labels)
pos_id = []
for idx in pos_idx[0]:
    if val_data.sources[idx] == val_data.destinations[idx]:
        pos_id.append(val_data.sources[idx])
        #print(val_data.sources[idx], val_data.destinations[idx], val_data.edge_idxs[idx], val_data.labels[idx],
        #      pred_labels[idx], val_data.timestamps[idx])
pos_id = np.array(pos_id)  # superchat node id
pos_edge_id = val_data.edge_idxs[pos_idx]

print(pos_id.shape)
print('pos_edge_id:', val_data.edge_idxs[pos_idx])
print('counter pos_id:', Counter(pos_id))
print('counter all_id:', Counter(val_data.sources))


idxs = np.where(val_data.sources == 962  )
surroundings = []
print(len(idxs))
for idx in idxs[0]:
    if val_data.labels[idx] == 1 and pred_labels[idx] == 1 and val_data.timestamps[idx] > 140211: #
        #print(pred_labels[idx], val_data.labels[idx], val_data.edge_idxs[idx])
        print(val_data.sources[idx], val_data.destinations[idx], val_data.edge_idxs[idx], val_data.labels[idx],
              pred_labels[idx], val_data.timestamps[idx])
        #surroundings.append()







print(a)
npy = np.load('embedding/UC1opHUrw8rvnsadT-iGp7Cg/97DWg8tqo4M_aug.npy')
data = pd.read_csv('embedding/UC1opHUrw8rvnsadT-iGp7Cg/97DWg8tqo4M_aug.csv')
print(data.shape)
print(npy.shape)

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




concat_list = ['ON3WijEIS1c', 'qO8Ld-qLjb0', 'k3Nzow_OqQY', 'y3DCfZmX8iA', 'qHZwDxea7fQ', 'cibdBr9TkEo', 'rW8jSXVsW2E', 'eIi8zCPFyng', 'wtJj3CO_YR0', '1kxCz6tt2MU']

for video_id in concat_list:
    #video_id = i #'97DWg8tqo4M'
    print('Video {} augmentation start'.format(video_id))
    old_data = pd.read_csv('embedding/UC1opHUrw8rvnsadT-iGp7Cg/{}.csv'.format(video_id))
    old_emb = np.load('embedding/UC1opHUrw8rvnsadT-iGp7Cg/{}.npy'.format(video_id))
    new_data = old_data.copy(deep=True)
    new_data = new_data[0:0]
    new_emb = np.zeros((1, old_emb.shape[1]))
    #print(new_data)
    for idx, line in old_data.iterrows():
        if idx%1000 == 0:
            print(idx)
    #print(idx, line)
        if int(line['superchat']) > 0:
            for i in range(10):
                new_data = new_data.append(line, ignore_index=True)# = pd.concat([new_data, line], ignore_index=True)
                new_emb = np.append(new_emb, np.expand_dims(old_emb[idx], axis=0), axis=0)
        else:
            new_data = new_data.append(line, ignore_index=True)#new_data = pd.concat([new_data, line], ignore_index=True)
            new_emb = np.append(new_emb, np.expand_dims(old_emb[idx], axis=0), axis=0)

    new_data = new_data.drop(columns=['Unnamed: 0'])
    print('old:', old_data.info())
    print('new:', new_data.info(), new_emb.shape)


    new_data.to_csv('embedding/UC1opHUrw8rvnsadT-iGp7Cg/{}_aug_10.csv'.format(video_id))
    np.save('embedding/UC1opHUrw8rvnsadT-iGp7Cg/{}_aug_10.npy'.format(video_id), new_emb)
    print('Video {} augmentation finished'.format(video_id))



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
dataset_name = '1kxCz6tt2MU_v3.10_dynamic_graph'
graph_df = pd.read_pickle('./dynamicGraph/{}.pkl'.format(dataset_name))
#graph_df = pd.read_csv('./dynamicGraph/ml_{}.csv'.format(dataset_name))
print(graph_df.info())
print(graph_df[:10])

'''
'''
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
