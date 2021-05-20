from datetime import datetime
import pandas as pd
import numpy as np
import torch
import os




def superchat(data):
    data.insert(data.shape[1], 'isSuperchat', 0)
    #sc_data = pd.read_csv('../superchats.csv')

    #sc_data = originChannelId(sc_data)
    #sc_data.to_csv('sc_data1.csv', encoding='utf-8')

    #sc_data = pd.read_csv('sc_data1.csv')
    #sc_data['timestamp'] = pd.to_datetime(sc_data['timestamp'])
    #sc_data = sc_data[sc_data['originVideoId']=='97DWg8tqo4M']
    #sc_data = commenters(data, sc_data)
    #sc_data.to_csv('sc_data2.csv', encoding='utf-8')

    sc_data = pd.read_csv('sc_data2.csv')

    result = pd.concat([data, sc_data], join='outer')
    result['timestamp'] = pd.to_datetime(result['timestamp'])
    result.sort_values('timestamp', inplace=True, ignore_index=True)
    del sc_data
    result.to_csv('sc_data3.csv', encoding='utf-8')
    result = membership(result)

    result.info()
    #print(result.describe())

    return result


def membership(data):
    status = {'unknown':0, 'non-member':0, 'less than 1 month':1, '1 month':2, '2 months':3, '6 months':4, '1 year':5, '2 years':6}

    data['membership'].replace(status, inplace=True)


    print('result:', data.info())

    return data

def originChannelId(sc_data):
    channels = pd.read_csv('../channels.csv')
    channels = channels[['channelId', 'name.en']]
    channels.columns = ['originChannelId', 'originChannel']
    sc_data = sc_data[['timestamp', 'significance', 'body', 'id', 'channelId', 'originVideoId',
             'originChannel']]
    sc_data = pd.merge(sc_data, channels, how='left', on=['originChannel'])
    sc_data.drop(['originChannel'], axis=1, inplace=True)
    sc_data.info()

    return sc_data

def commenters(data, sc_data):
    membership_l = np.array([])
    moderator_l = np.array([])
    verified_l = np.array([])
    dic = {}
    chats_n = sc_data.shape[0]
    for index, row in sc_data.iterrows():
        commenter_id = row['channelId']
        if commenter_id in dic:
            mem = dic[commenter_id][0]
            mod = dic[commenter_id][1]
            ver = dic[commenter_id][2]
        else:
            cur = data[data['channelId']==commenter_id]
            #print(cur)
            mem = cur['membership'].max()
            mod = cur['isModerator'].max()
            ver = cur['isVerified'].max()
            dic[commenter_id] = [mem, mod, ver]
        membership_l = np.append(membership_l, mem)
        moderator_l = np.append(moderator_l, mod)
        verified_l = np.append(verified_l, ver)
        if index % 100 == 0:
            print('{}/{}'.format(index, chats_n))
        if index % 10000 == 0:
            np.save('membership_l_{}.npy'.format(index), membership_l)
            np.save('moderator_l_{}.npy'.format(index), moderator_l)
            np.save('verified_l{}.npy'.format(index), verified_l)


    sc_data.rename(columns={'significance': 'isSuperchat'}, inplace=True)
    sc_data= sc_data[['timestamp', 'body', 'isSuperchat', 'id', 'channelId', 'originVideoId', 'originChannelId']]
    sc_data.insert(sc_data.shape[1], 'isVerified', verified_l)
    sc_data.insert(sc_data.shape[1], 'isModerator', moderator_l)
    sc_data.insert(sc_data.shape[1], 'membership', membership_l)

    print('commenter finished!')
    return sc_data


def main():
    ''''
    chat_3 = pd.read_csv('../chats_2021-03.csv')
    chat_4 = pd.read_csv('../chats_2021-04.csv')
    print('chat3:', chat_3.info())
    print('chat4:', chat_4.info())
    chat_3['timestamp'] = pd.to_datetime(chat_3['timestamp'])
    chat_4['timestamp'] = pd.to_datetime(chat_4['timestamp'])

    chat = pd.concat([chat_3[chat_3['timestamp'] > '2021-03-15T23:19:38.000000+00:00'], chat_4])
    chat.to_csv('../chat3&4.csv')
    '''
    chat = pd.read_csv('../chat3&4.csv')
    result = superchat(chat)
    result.to_csv('../result2.csv', encoding='utf-8')



if __name__ == '__main__':
    main()