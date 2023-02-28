import math
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score, \
    confusion_matrix
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
import time


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc, val_acc, val_rec, val_pre = [], [], [], [], []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

            size = len(sources_batch)
            _, negative_samples = negative_edge_sampler.sample(size)

            pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                                  negative_samples, timestamps_batch,
                                                                  edge_idxs_batch, n_neighbors)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])


            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

            pred_label = [int(n + 0.5) for n in pred_score]
            acc = accuracy_score(true_label, pred_label)
            pre = precision_score(true_label, pred_label)
            rec = recall_score(true_label, pred_label)
            val_acc.append(acc)
            val_pre.append(pre)
            val_rec.append(rec)

    return np.mean(val_ap), np.mean(val_auc), np.mean(val_acc), np.mean(val_rec), np.mean(val_pre)


def eval_node_classification(tgn, decoders, data, edge_idxs, node_dim, batch_size, n_neighbors, device):
    pred = torch.zeros((len(data.sources), node_dim)).to(device)
    pred_prob_num = torch.zeros((len(decoders), len(data.sources))).to(device)
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        decoders = [decoder.eval() for decoder in decoders]
        tgn.eval()
        # decoder.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx: e_idx]
            destinations_batch = data.destinations[s_idx: e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx: e_idx]

            source_embedding = tgn.compute_temporal_embeddings(sources_batch,
                                                               destinations_batch,
                                                               timestamps_batch,
                                                               edge_idxs_batch,
                                                               n_neighbors)

            pred[s_idx: e_idx] = source_embedding

    embedding = pred.cpu().numpy()

    for d_idx in range(len(decoders)):
        pred_prob = decoders[d_idx](pred)
        pred_prob_num[d_idx] = torch.argmax(pred_prob, dim=1)

    pred_prob_num = torch.mean(pred_prob_num, 0).squeeze()
    pred_label = (pred_prob_num + 0.5).trunc().detach().cpu().numpy()
    acc = accuracy_score(data.labels, pred_label)
    pre = precision_score(data.labels, pred_label)
    rec = recall_score(data.labels, pred_label)
    cm = confusion_matrix(data.labels, pred_label)
    auc_roc = roc_auc_score(data.labels, pred_label)
    torch.cuda.empty_cache()

    np.save('week_pred_label.npy', np.array(pred_label))
    np.save('week_pred_embedding.npy', np.array(embedding))
    np.save('week_data_labels.npy', np.array(data.labels))
    print('pred_label saved')

    return auc_roc, acc, pre, rec, cm



def TSNE(data, label):
    start = time.time()
    tsne = manifold.TSNE(n_components=2, init='pca', n_iter=2000, perplexity=50, random_state=501)
    datat_tsne = tsne.fit_transform(data)
    x_min, x_max = datat_tsne.min(0), datat_tsne.max(0)
    X_norm = (datat_tsne - x_min) / (x_max - x_min)  # 归一化
    print("Org data dimension is {}.TSNE data dimension is {}. time:{:.3f}s"
          .format(data.shape[-1], datat_tsne.shape[-1], time.time() - start))

    plt.figure(figsize=(8, 8))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], color=plt.cm.Set1(label))
    plt.xticks([])
    plt.yticks([])

    plt.savefig('./distribution.jpg', dpi=500)
    plt.show()

    return
