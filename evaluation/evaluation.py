import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix


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

      if (np.isfinite(pred_score) == False).nonzero()[0].shape[0] != 0:
        pred_score = np.nan_to_num(pred_score, nan=0.0, posinf=1.0, neginf=0.0)

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

      pred_label = [int(n+0.5) for n in pred_score]
      acc = accuracy_score(true_label, pred_label)
      pre = precision_score(true_label, pred_label)
      rec = recall_score(true_label, pred_label)
      val_acc.append(acc)
      val_pre.append(pre)
      val_rec.append(rec)


      #print('acc:{}, pre:{}, rec:{}'.format(acc, pre, rec))

  return np.mean(val_ap), np.mean(val_auc), np.mean(val_acc), np.mean(val_rec), np.mean(val_pre)


def eval_node_classification(tgn, decoder, data, edge_idxs, node_dim, batch_size, n_neighbors):
  pred = torch.zeros((len(data.sources), node_dim))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    # [decoder.eval() for decoder in decoders]
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      # pred_prob_batch = decoder(source_embedding).sigmoid()

      pred[s_idx: e_idx] = source_embedding

      #pred_label = [int(n + 0.5) for n in pred_prob]


      # pred_prob_batch = decoder.test_(source_embedding).sigmoid().cpu().numpy()
      #pred_prob_num[s_idx: e_idx] = np.sum(pred_prob_batch >= 0.5, axis=0)
      # pred_prob[s_idx: e_idx] = np.mean(pred_prob_batch, axis=0)
      '''    # rank start
  n_decoder = len(decoders)

  pred_rank_index = pred_prob.argsort()[-min(num_instance, 2*data.n_pos):]
  true_rank_index = data.labels.argsort()[-min(num_instance, 2*data.n_pos):]
  #pred_rank_label = [1 if n > (n_decoder / 2) else 0 for n in pred_prob_num[pred_rank_index]]
  pred_rank_label = [0]*num_instance
  for i in pred_rank_index:
    pred_rank_label[i] = 1

  print(pred_prob[pred_rank_index])
  print("True pos probability:", pred_prob[true_rank_index])
  

  acc = accuracy_score(data.labels, pred_rank_label)
  pre = precision_score(data.labels, pred_rank_label)
  rec = recall_score(data.labels, pred_rank_label)
  #print('acc:{}, pre:{}, rec:{}'.format(acc, pre, rec))
  cm = confusion_matrix(data.labels, pred_rank_label)
  #print('confusion matrix', cm)
  #print(data.labels.shape, pred_prob.shape, pred_prob)
  try:
    auc_roc = roc_auc_score(data.labels, pred_prob[pred_rank_index])
  except ValueError:
    auc_roc = 0.0
    # rank end

  np.savetxt("./pred_prob.csv", pred_prob, delimiter=' ')
  np.savetxt("./true_label.csv", data.labels, delimiter=' ')
  '''

  if (torch.isfinite(pred)==False).nonzero().shape[0] != 0:
    pred_prob = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
  # pred_label = [1 if n>(n_decoder/2) else 0 for n in pred_prob_num]
  # pred_label = [int(n+0.5) for n in pred_prob]
  # print(set(pred_label))
  # print(set(data.labels))
  # print(pred_prob[10:])
  target = torch.from_numpy(data.labels).long().to(tgn.device)
  pred_label, pred_prob = decoder.test_(pred, target, len(data.sources))

  print(set(pred_label))

  acc = accuracy_score(data.labels, pred_label)
  pre = precision_score(data.labels, pred_label)
  rec = recall_score(data.labels, pred_label)
  #print('acc:{}, pre:{}, rec:{}'.format(acc, pre, rec))
  cm = confusion_matrix(data.labels, pred_label)
  #print('confusion matrix', cm)
  #print(data.labels.shape, pred_prob.shape, pred_prob)
  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc, acc, pre, rec, cm
