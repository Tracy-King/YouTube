from sklearn import manifold, datasets
import matplotlib.pyplot as plt
import time
import numpy as np




def TSNE(data, label):
  start = time.time()
  tsne = manifold.TSNE(n_components=2, init='pca', n_iter=3000, perplexity=50, random_state=114, learning_rate=100)
  datat_tsne = tsne.fit_transform(data)
  # 归一化
  x_min, x_max = datat_tsne.min(0), datat_tsne.max(0)
  X_norm = (datat_tsne - x_min) / (x_max - x_min)  # 归一化
  print("Org data dimension is {}.TSNE data dimension is {}. time:{:.3f}s"
        .format(data.shape[-1], datat_tsne.shape[-1], time.time() - start))

  plt.figure(figsize=(8, 8))
  #for i in range(X_norm.shape[0]):
    # cm: color map
  fontsize = 20
  pos_mask = np.where(label==1)[0].tolist()
  neg_mask = np.where(label==0)[0].tolist()
  print(len(pos_mask), len(neg_mask))
  print(type(X_norm))
  plt.scatter(X_norm[neg_mask, 0], X_norm[neg_mask, 1], s=2*fontsize, marker='.', label=label[neg_mask], color=plt.cm.Set1(label[neg_mask]))
  plt.scatter(X_norm[pos_mask, 0], X_norm[pos_mask, 1], s=fontsize, marker='x', label=label[pos_mask], color=plt.cm.Set1(label[pos_mask]))
  plt.xticks(fontsize=fontsize)
  plt.yticks(fontsize=fontsize)

  plt.legend(('Normal', 'Superchat'), markerscale=2, fontsize=fontsize, frameon=False)
  plt.savefig('./distribution.jpg', dpi=500)
  plt.show()

  return


dataset = 'week_'  # week_
n = 4
pred_embedding = np.load('{}pred_embedding.npy'.format(dataset))


pred_label = np.load('{}pred_label.npy'.format(dataset))
data_labels = np.load('{}data_labels.npy'.format(dataset))

#print(data_labels[:10])

pos_mask_1 = np.where(data_labels == 1)[0]
neg_mask_1 = np.where(data_labels == 0)[0]

pos_mask_2 = np.where(pred_label == 1)[0]
neg_mask_2 = np.where(pred_label == 0)[0]

pos_mask = np.intersect1d(pos_mask_1, pos_mask_2)
neg_mask = np.intersect1d(neg_mask_1, neg_mask_2)

print(len(pos_mask), len(neg_mask))
neg_mask = np.random.choice(neg_mask, n*len(pos_mask))
mask = np.hstack((pos_mask, neg_mask))
np.random.shuffle(mask)
print(len(mask))
np.save('{}mask.npy'.format(dataset), mask)

mask = np.load('{}mask.npy'.format(dataset))

TSNE(pred_embedding[mask], data_labels[mask])
print('graph generated.')