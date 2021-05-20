import pickle
import json
import codecs


root = 'tgn-attn-277076677-2'

data = pickle.load(open('results/{}.pkl'.format(root), 'rb'))


print(json.dumps(data, sort_keys=True, indent=4))