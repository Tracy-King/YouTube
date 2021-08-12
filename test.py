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

video_id = '97DWg8tqo4M'
new_data = pd.read_pickle('dynamicGraph/{}_dynamic_graph.pkl'.format(video_id))
end_time = new_data['Offset'].values[-1]
print(end_time, type(end_time))

#data = pd.read_csv('src/sc_data3.csv')

#data.info()

#print(data[:10])
