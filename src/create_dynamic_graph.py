import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
import math
from pathlib import Path
import codecs
import json
import os

pd.set_option('display.width', None)


def video_separate(data):
    video_list = data['video_id'].drop_duplicates().values.tolist()
    groups = data.groupby(data.video_id)
    tmp = groups.get_group(video_list[0])
    #print(tmp, len(groups))

    return video_list, groups


def time_window_separate(series, emb, window, thrs):
    dynamic_graph = []
    end_time = int(series['offset'].iloc[-1]) + 1
    history_list = []
    node_list = {}
    history_edge_list = []
    node_feature = {}

    batch_list = list(range(0, end_time, window))
    batch_list.append(end_time)

    #print(batch_list, len(batch_list))

    for i in range(len(batch_list) - 1):  #
        tmp_series = series[(series['offset'] > batch_list[i])
                            & (series['offset'] <= batch_list[i + 1])]
        for j, cur_row in tmp_series.iterrows():
            #print(type(cur_row), cur_row)
            #print(cur_row['body'])
            body_length = len(str(cur_row['body']))
            commenter_id = cur_row['commenter_id']
            offset = cur_row['offset']
            s_label = cur_row['superchat']
            m_label = cur_row['membership']
            value = emb[j]


            while len(history_list) != 0 and history_list[0][0]['offset'] + window < offset:
                history_list.pop(0)

            while len(history_edge_list) != 0 and history_edge_list[0]['offset'] + window < offset:
                history_edge_list.pop(0)
                #print("Remove")

            if commenter_id not in node_list:
                node_list[commenter_id] = 1
                node_feature[commenter_id] = value
                dynamic_graph.append(['CREATE', commenter_id, '0', value, i, offset, s_label, m_label, 1.0, body_length])
                #print("Create")
            else:
                node_list[commenter_id] += 1
                node_feature[commenter_id] = value
                dynamic_graph.append(['UPDATE', commenter_id, '0', value, i, offset, s_label, m_label, 1.0, body_length])
                #print("Update")
            for n in history_list:
                v1 = np.array(emb[n[1]])
                v2 = np.array(emb[i])
                cos_sim = 1 - pdist(np.vstack([v1, v2]), 'cosine')
                if math.isnan(cos_sim) or n[0]['commenter_id'] == commenter_id:
                    continue
                if cos_sim > thrs:
                    history_edge_list.append({'node1': n[0]['commenter_id'],
                                              'node2': commenter_id,
                                              'offset': offset,
                                              'superchat':n[0]['superchat'],
                                              'membership':n[0]['membership']})

                    dynamic_graph.append(['EDGE', n[0]['commenter_id'], commenter_id, [0], i, offset, n[0]['superchat'], n[0]['membership'], cos_sim, body_length])
                    #print("Edge")
            history_list.append([cur_row, i])
        if i % 50 == 0:
             print('Batch:{}/{}'.format(i, len(batch_list)))
            #print("edge:", len(history_edge_list))
            #print("node:", len(history_list))
            #print(dynamic_graph)

    return pd.DataFrame(dynamic_graph, columns=['Inst', 'Node1', 'Node2', 'Value', 'Batch', 'Offset', 'Superchat', 'Membership', 'Weight', 'Length'])

'''
def dynamic_graph_create(video_list, video_groups, window, thrs):
    cnt = 0
    for video_id in video_list:
        print('Video {} start ({}/{})'.format(video_id, cnt, len(video_list)))
        dynamic_graph = time_window_separate(video_groups.get_group(video_id), window, thrs)
        dir = Path('{}_w{}_t{}'.format(root, window, thrs))
        dir.mkdir(parents=True, exist_ok=True)
        dynamic_graph.to_pickle('{}_w{}_t{}/{}_dynamic_graph.pkl'.format(root, window, thrs, video_id))
        cnt += 1
'''

def dynamic_graph_create(series, emb, window, thrs):
    dynamic_graph = time_window_separate(series, emb, window, thrs)
    dir = Path('{}_w{}_t{}'.format(root, window, thrs))
    dir.mkdir(parents=True, exist_ok=True)
    dynamic_graph.to_pickle('{}_w{}_t{}/{}_dynamic_graph.pkl'.format(root, window, thrs, video_id))

def preprocess(data, video_id, channel_id):
    u_list, i_list, ts_list, superchat_list, membership_list, weight_list, length_list = [], [], [], [], [], [], []
    idx_list = []
    idx_cnt = 1
    update_records = {}
    unique_node = data['Node1'].drop_duplicates().values.tolist()
    n_unique_node = len(unique_node)
    print('unique node:', n_unique_node)
    node_dict = dict(zip(unique_node, list(range(1, n_unique_node+1))))

    data['Superchat'] = data['Superchat'].fillna(0)
    data['Membership'] = data['Membership'].fillna(0)
    data['Superchat'] = data['Superchat'].replace('', 0)
    data['Membership'] = data['Membership'].replace('', 0)

    feat_n = [0 for _ in range(n_unique_node+1)]

    for _, cur_row in data[data['Inst']=='CREATE'].iterrows():
        feat_n[node_dict[cur_row['Node1']]] = np.array([float(x) for x in cur_row['Value']])

    for idx, cur_row in data[(data['Inst']=='UPDATE') | (data['Inst']=='EDGE')].iterrows():
        u = int(node_dict[cur_row['Node1']])  ## user id
        if cur_row['Inst'] == 'UPDATE':
            i = u  ## item id
            update_records[idx_cnt] = [float(x) for x in cur_row['Value']]
        else:
            i = int(node_dict[cur_row['Node2']])  ## item id

        ts = float(cur_row['Offset'])  ## time stamp
        superchat = int(cur_row['Superchat'])  # int(e[3])   # state label
        membership = int(float(cur_row['Membership']))  # int(e[3])   # state label
        weight = cur_row['Weight']
        length = cur_row['Length']

        u_list.append(u)
        i_list.append(i)
        ts_list.append(ts)
        weight_list.append(float(weight))
        length_list.append(int(length))
        superchat_list.append(superchat)
        membership_list.append(membership)
        idx_list.append(idx_cnt)
        idx_cnt += 1


    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'weight': weight_list,
                         'length': length_list,
                         'superchat': superchat_list,
                         'membership': membership_list,
                         'idx': idx_list}), np.array(feat_n[1:]), update_records, node_dict


def save_file(df, video_id, update_records, feat_n):
    Path("data/").mkdir(parents=True, exist_ok=True)
    PATH = '../dynamicGraph/{}.csv'.format(video_id)
    OUT_DF = '../dynamicGraph/ml_{}.csv'.format(video_id)
    OUT_FEAT = '../dynamicGraph/ml_{}.json'.format(video_id)
    OUT_NODE_FEAT = '../dynamicGraph/ml_{}_node.npy'.format(video_id)

    #print(feat_l.shape, feat_n.shape)
    #print(feat_l[:5], feat_n[:5])

    empty = np.zeros(feat_n[0].shape[0])[np.newaxis, :]
    feat_n = np.vstack([empty, feat_n])


    #print(feat_l.shape, feat_n.shape)
    #print(feat_l[:5], feat_n[:5])

    df.to_csv(OUT_DF)

    with open(OUT_FEAT, 'w') as f:
        json.dump(update_records, f)

    np.save(OUT_NODE_FEAT, feat_n)


if __name__ == '__main__':
    root = '..\embedding'
    #video_id = '97DWg8tqo4M'
    video_id_list = []
    channel_id = 'UC1opHUrw8rvnsadT-iGp7Cg'
    g = os.walk('{}\{}'.format(root, channel_id))
    for path, dir_list, file_list in g:
      for file_name in file_list:
        print(file_name[:-4])
        if file_name[-3:] == 'csv':
          video_id_list.append(file_name[:-4])
    print("videos num:", len(video_id_list))
    #savemodel = 'train.model'
    window = 30
    thrs = math.cos(math.pi/6)

    for video_id in video_id_list:
        data = pd.read_csv('..\embedding\{}\{}.csv'.format(channel_id, video_id), na_values='0.0', keep_default_na=False)
        emb = np.load('..\embedding\{}\{}.npy'.format(channel_id, video_id))
        print(data.info())

    #video_list, video_groups = video_separate(data)
    #dynamic_graph_create(video_list, video_groups, window, thrs)
    #data = pd.read_pickle('{}/277076677_dynamic_graph.pkl'.format(root))

    #print(data['superchat'].drop_duplicates().tolist())

        data = time_window_separate(data, emb, window, thrs)
    #data.to_csv('../dynamicGraph/{}_dynamic_graph.csv'.format(video_id))
        data.to_pickle('../dynamicGraph/{}_dynamic_graph.pkl'.format(video_id))
        new_data = pd.read_pickle('../dynamicGraph/{}_dynamic_graph.pkl'.format(video_id))
        print(new_data.info())
    #print(data)
    #print(data[(data['video_id']=='277076677') & (data['commenter_id']=='113567493')])
        df, feat_n, update_records, node_dict = preprocess(new_data, video_id, channel_id)

    #print(df['superchat'].drop_duplicates())
    #print(df['membership'].drop_duplicates())

        save_file(df, video_id, update_records, feat_n)

        with open('..\dynamicGraph\ml_{}.json'.format(video_id), 'r', encoding='UTF-8') as f:
            update_records = json.load(f)

        print(len(update_records))







