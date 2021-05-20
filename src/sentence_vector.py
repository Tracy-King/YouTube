from sentence_transformers import SentenceTransformer, models
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import os
import re
from sklearn.decomposition import PCA



stop_words = {"i'm", "i’m", 'im', "i've", "won't", "wouldn't", 'wasn', "wasn't",
              'while', 'who', 'than', 'aren', 'ourselves', "mightn't", 'at',
              'himself', 'same', 'once', 'it', 'a', 'both', 'no', 'me', 'shan',
              'do', 'myself', 'those', 'to', 'from', "don't", 'other', 'o', 'haven',
              'whom', 'theirs', 'what', "you'd", 'most', 'hasn', 'its', "you'll",
              'each', 'through', 'just', "didn't", "mustn't", 'how''her', 'am',
              "you've", 'be', 'over', 'didn', 'about', 'below', "hadn't", 'mightn',
              'being', 'they', 'he', 'yourself', 'there', 'or', 'wouldn', 'is',
              'where', 'so', 'if', 'him', 'you', "needn't", 'couldn', "couldn't",
              'with', "she's", 'them', 'been', 'had', 'on', 'up', 'off', 's', 'have',
              'itself', 'are', 'by', "that'll", 't', 'll', 'has', 'their', 'did', 'as',
              'all', 'not', 'only', 'y', 'your', 'before', 'we''having', 'don', 'needn',
              'these', "you're", 're', 'm', 'after', 'hers', 'she', 'until', 'of',
              'again', 'any', 'out', 'against', 'does', 'more', "haven't", 'down',
              'hadn', 'because', 'why', 've', 'my', 'here', 'the', 'his', 'now', 'ma',
              'yourselves', "aren't", 'such', "doesn't", 'and', 'our', 'themselves',
              'should', 'this', 'weren', 'under', 'but', 'can', 'isn', "weren't",
              'between', 'herself', 'own', 'shouldn', 'doing', 'i', 'few', "shouldn't",
              'for', 'then', 'which', 'in', 'ours', 'd', 'an', 'into', "hasn't", 'very',
              "shan't", 'were', 'too', 'nor', 'will', "should've", 'some', 'was', 'ain',
              'above', 'yours', 'during', 'doesn', "isn't", 'mustn', "it's", 'when',
              'further', 'that',
              }

def split_message(message):
    pattern = re.compile(r'[\t\r\n\f]+')
    message = message.strip(',')
    message = message.strip('')
    message = message.replace('.', ' ')
    message = message.replace('⠀', ' ')
    message = message.replace(' ', ' ')
    message = message.replace("’", "'")
    message = message.replace("::", ": :")
    message = message.lower()
    message = re.sub(pattern, ' ', message)
    return message.split(' ')


def clean_punct(word):
    word = word.strip('?')
    word = word.strip(".")
    word = word.strip('"')
    word = word.strip("!")
    word = word.strip(",")
    word = word.strip("'")
    word = word.strip(")")
    word = word.strip("(")
    word = word.strip("|")
    return word



def is_number(s):
    if re.match("^\d+?\.\d+?$", s) is None:
        return s.isdigit()
    return True


def is_URL(word):
    return "http" in word


def is_valid_word(word, stop_words_set):
    if(is_URL(word)):
        return False
    if(word in stop_words_set):
        return False
    if len(word) <= 1:
        return False
    if word == '"':
        return False
    if is_number(word):
        return False
    return True

# 1. stop words   2. private messages, ‘bot’ messages and generally malformed and unusual tokens
def clean_words(sentence):
    words = split_message(sentence)
    result = []
    for word in words:
        if not is_valid_word(word, stop_words):
            continue
        word = word.strip(' \t\n\r')
        word = clean_punct(word)

        result.append(word)

    return result


def embedding(df, output_d=128):
    model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1',
                                device='cuda')
    #df.info()
    #print(df.describe())
    #print(df[:10])
    chats = df['body'].astype(str).values.tolist()
    #sentence = clean_words(chats)


    pca_train_sentences = chats[0:2000]
    train_embeddings = model.encode(pca_train_sentences, convert_to_numpy=True)

    pca = PCA(n_components=output_d)
    pca.fit(train_embeddings)
    pca_comp = np.asarray(pca.components_)

    dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=output_d, bias=False,
                         activation_function=torch.nn.Identity())
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    model.add_module('dense', dense)

    emb = model.encode(chats, convert_to_numpy=True)

    #print(emb[:10])
    #print(emb.shape)

    return np.array(emb)

def new_df(df, emb):
    body = df['body'].values.tolist()
    commenter_id = df['channelId'].values.tolist()
    timestamp = df['timestamp']
    superchat = df['isSuperchat']
    moderator = df['isModerator']
    verified = df['isVerified']
    member = df['membership']

    offset = []
    try:
        start = datetime.strptime(timestamp[0], "%Y-%m-%d %H:%M:%S.%f%z")
    except ValueError:
        start = datetime.strptime(timestamp[0], "%Y-%m-%d %H:%M:%S%z")
    for time in timestamp:
        try:
            end = datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f%z")
        except ValueError:
            end = datetime.strptime(time, "%Y-%m-%d %H:%M:%S%z")

        offset_t = (end - start).seconds + (end - start).microseconds/10e5
        offset.append(offset_t)

    #print(len(body), len(commenter_id), len(offset))

    data = {
        'body':body,
        'commenter_id':commenter_id,
        'offset':offset,
        'superchat':superchat,
        'moderator':moderator,
        'verified':verified,
        'membership':member
    }
    new_data = pd.DataFrame(data)

    #print(new_data[:10])
    return new_data



def main(root, channel_id, video_id):
    target_path = '../embedding/{}/{}.csv'.format(channel_id, video_id)
    if os.path.exists(target_path):
        return
    df = pd.read_csv(root)
    if df.shape[0] <= 128:
        return
    emb = embedding(df)
    new_data = new_df(df, emb)
    path = '../embedding/{}'.format(channel_id, video_id)
    if not os.path.exists(path):
        os.mkdir(path)

    new_data.to_csv(target_path, encoding='utf-8')
    np.save('../embedding/{}/{}.npy'.format(channel_id, video_id), emb)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    channel_id = 'UC1opHUrw8rvnsadT-iGp7Cg'
    #video_id = '97DWg8tqo4M'
    #root = '../channels/{}/{}.csv'.format(channel_id, video_id)

    g = os.walk(r"..\channels")
    for path, dir_list, file_list in g:
        channel_id = path[12:]
        #print(channel_id)
        for file_name in file_list:
            video_id = file_name[:-4]
            #print(video_id)
            print(os.path.join(path, file_name))
            main(os.path.join(path, file_name), channel_id, video_id)



    #main(root, channel_id, video_id)


    #t1 = '2021-03-27 18:20:07.070000+09:00'
    #t2 = '2021-03-27 18:20:05.665000+09:00'
    #d1 = datetime.strptime(t1, "%Y-%m-%d %H:%M:%S.%f%z")
    #d2 = datetime.strptime(t2, "%Y-%m-%d %H:%M:%S.%f%z")
    #print((d1-d2), (d1-d2).seconds, (d1-d2).microseconds, (d1-d2).seconds+(d1-d2).microseconds/10e5)

    #superchat = df[df['isSuperchat'] == 1]['channelId']
    #print(superchat)
    #print(len(superchat.drop_duplicates().values.tolist()))
    #print(len(df['channelId'].drop_duplicates().values.tolist()))

    # toxic = deleted_chat.sample(n_sample)['body'].to_list()
    # safe  = chat_df.dropna().sample(n_sample)['body'].to_list()

    # toxic_embeds = model.encode(toxic)
    # safe_embeds = model.encode(safe)