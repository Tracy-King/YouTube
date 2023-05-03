# Predicting Potential Real-time Donations in YouTube Live Streaming Services via Continuous-time Dynamic Graph


## Introduction

Online live streaming platforms, such as YouTube Live and Twitch, have seen a surge in popularity in recent years. These platforms allow viewers to send real-time gifts to streamers, which can bring significant profits and fame. However, there has been little research on the donation system used on live streaming platforms. 

This work aims to fill this gap by building a continuous-time dynamic graph to model the interactions among viewers based on real-time chat messages and predict the donations on live streaming platforms. To achieve this, we propose a novel model called the Temporal Difference Graph Neural Network (TDGNN) that incorporates imbalanced learning strategies to identify potential donors during live streaming. Our model can predict the exact time when donations appear. 

We conduct extensive experiments on three live streaming video datasets and demonstrate that our proposed model is more effective and robust than other baseline methods from other fields.

## Keywords
Online Live Streaming, Real-time Donation, Continuous-time Dynamic Graph, Dynamic Node Label Prediction



## Running the experiments

### Requirements

Dependencies (with python >= 3.7):

```{bash}
pandas==1.1.0
torch==1.6.0
scikit_learn==0.23.1
pyarrow==11.0.0 (only for reading *.parquet files)
glob
sentence-transformers
```

### Dataset and Preprocessing

#### Download the public data
Download the Vtuber 1B datasets from
[here](https://github.com/sigvt/vtuber-livechat-dataset) and store the files in the `root` folder.

Update: The dataset author changed the dataset content to a simple version. Please contact the author to obtain the VTuber 1B Complete version dataset. Dataset file format are changed from *.csv to *.parquet. Pyarrow library is needed for reading the *.parquet files.


#### Preprocess the data
A pretrained BERT model is necessary for embedding. Download the pretrained model from [here](https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models) and store the files in the `root` folder. `paraphrase-xlm-r-multilingual-v1` is used as the default. You can try other pretrained multilingual models :).

Preprocess initialization is necessary for the first time.
```{bash}
# Initializing the dataset:
python preprocess_data.py --initial
```

After initialization, you can check the channels IDs and video IDs in the '/channel' folder. To create the 'short/mid/long' dynamic graph for a specific channel `channelId`:

```{bash}
# Create mid dynamic graph for channelId:
python preprocess_data.py --channel channelId --type mid
```
#### General flags
```{txt}
optional arguments:
  --channel channelId        Channel ID (refer to dir '/channel/')
  --type length              Length of the dynamic graph (select from short, mid, long)
  --graph                    Whether to create dynamic graph
  --initial                  Whether to initialize
```

The name format will be `concat_channelId_length` (e.g. concat_UC1opHUrw8rvnsadT-iGp7Cg_mid). The generated file will be stored in folder `/dynamicGraph/`. For example:
```{txt}
  './dynamicGraph/ml_concat_UC1opHUrw8rvnsadT-iGp7Cg_mid.csv'        dynamic graph interactions
  './dynamicGraph/ml_concat_UC1opHUrw8rvnsadT-iGp7Cg_mid.json'       dynamic graph node updating records
  './dynamicGraph/ml_concat_UC1opHUrw8rvnsadT-iGp7Cg_mid_node.npy'   node initial features
```



### Model Training

Supervised learning on generated dynamic node classification:
```{bash}
# Train the model on mid dynamic graph of channel UC1opHUrw8rvnsadT-iGp7Cg_mid:
python3 train_supervised.py -d concat_UC1opHUrw8rvnsadT-iGp7Cg_mid --dataset_r1 0.7 --dataset_r2 0.85 --prefix tgn-attn-UC1opHUrw8rvnsadT-iGp7Cg_mid
```


### Ablation Study
```{bash}
# Without the temporal difference module:
python train_supervised.py -d concat_UC1opHUrw8rvnsadT-iGp7Cg_mid --dataset_r1 0.7 --dataset_r2 0.85 --prefix tgn-attn-UC1opHUrw8rvnsadT-iGp7Cg_mid --without_difference
```


#### General flags

```{txt}
optional arguments:
  -d DATA, --data DATA         Data sources to use (wikipedia or reddit)
  --n_decoder N_DECODER        Number of ensemble decoders
  --n_undersample N            Parameter for undersampling
  --dataset_r1  R1             Validation dataset ratio
  --dataset_r2  R2             Test dataset ratio
  --bs BS                      Batch size
  --prefix PREFIX              Prefix to name checkpoints and results
  --n_degree N_DEGREE          Number of neighbors to sample at each layer
  --n_head N_HEAD              Number of heads used in the attention layer
  --n_epoch N_EPOCH            Number of epochs
  --n_layer N_LAYER            Number of graph attention layers
  --lr LR                      Learning rate
  --patience                   Patience of the early stopping strategy
  --n_runs                     Number of runs (compute mean and std of results)
  --drop_out DROP_OUT          Dropout probability
  --gpu GPU                    Idx for the gpu to use
  --node_dim NODE_DIM          Dimensions of the node embedding
  --time_dim TIME_DIM          Dimensions of the time embedding
  --backprop_every             Number of batches to process before performing backpropagation
  --uniform                    Whether to sample the temporal neighbors uniformly (or instead take the most recent ones)
  --cost                       Whether to use the cost-sensitivity loss function
  --without_difference         Whether to not use temporal difference module
```

