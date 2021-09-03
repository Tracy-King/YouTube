import pickle
import json
import codecs
import pandas as pd
import torch
import numpy as np
import re
import difflib

pd.set_option('display.max_columns', None)

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

a = [0, 1, 2]

print(np.tile(a, (2, 1)))
'''
video_id = '97DWg8tqo4M'
new_data = pd.read_pickle('dynamicGraph/{}_dynamic_graph.pkl'.format(video_id))
test = new_data
#end_time = new_data['Offset'].values[-1]
#print(end_time, type(end_time))

for (i, j), k in zip(new_data[10:20].iterrows(), range(10)):
  print(i, j, k)
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

#data = pd.read_csv('src/sc_data3.csv')

#data.info()

#print(data[:10])
