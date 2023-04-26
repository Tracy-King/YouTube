import pandas as pd
import numpy as np


import os


def splitData():
    df = pd.read_csv("../result_all.csv", index_col="timestamp", parse_dates=True, na_values='', keep_default_na=False)

    # Convert UTC to JST
    df.index = df.index.tz_convert('Asia/Tokyo')

    # body length
    df["bodylength"] = df["body"].str.len().fillna(0).astype("int")
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df.drop(['Unnamed: 0.1'], axis=1, inplace=True)
    df.info()


    channel_list = df['originChannelId'].drop_duplicates().values.tolist()

    channel_cnt = 0
    channel_list_n = len(channel_list)

    for channel in channel_list:
        video_cnt = 0
        path = '../channels/{}'.format(channel)
        if not os.path.exists(path):
            os.mkdir(path)
        channel_df = df[df['originChannelId'] == channel]
        video_list = channel_df['originVideoId'].drop_duplicates().values.tolist()
        video_list_n = len(video_list)
        for video in video_list:
            video_df = channel_df[channel_df['originVideoId'] == video]
            video_df.to_csv(path + '/{}.csv'.format(video), encoding='utf-8')
            print('Video {}/{}'.format(video_cnt, video_list_n))
            video_cnt += 1

        print('Channel {}/{}'.format(channel_cnt, channel_list_n))
        channel_cnt += 1


if __name__ == '__main__':
    splitData()
