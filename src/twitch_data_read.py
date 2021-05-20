import pandas as pd
from collections import Counter
from lemmatisation_tools import lemmatise
import re
import codecs
import math
import json
import numpy as np

from gensim.models import word2vec

HIDDEN = 300

pd.set_option('display.width', None)

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


def is_bot(word):
    if "nightbot" in word:
        return True
    if "mmrbot" in word:
        return True
    if "streamelements" in word:
        return True
    if "moobot" in word:
        return True
    if "deepbot" in word:
        return True
    if "wizebot" in word:
        return True
    if "phantombot" in word:
        return True
    if "72hrsbot" in word:
        return True
    if "stay_hydrated_bot" in word:
        return True
    return False


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


def is_bot_command(word):
    return word[0] == '/'


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
    if is_bot_command(word):
        return False
    return True

# 1. stop words   2. private messages, ‘bot’ messages and generally malformed and unusual tokens
def clean_words(sentence):
    words = split_message(sentence)
    result = []
    for word in words:
        if not is_valid_word(word, stop_words):
            continue
        word = lemmatise(word)
        word = word.strip(' \t\n\r')
        word = clean_punct(word)

        result.append(word)

    return result

def tfidf(word, count, count_list):
    tf = count[word] / sum(count.values())
    idf = math.log(len(count_list) / (1 + sum(1 for count in count_list if word in count)))
    return tf * idf

def pkl2txt(root):
    f = codecs.open('{}.txt'.format(root), 'w', "utf-8")
    data = pd.read_pickle('ICWSM19_data/{}.pkl'.format(root))
    body = data['body'].values.tolist()

    word_list = []
    for row in body:
        sentence = clean_words(row)
        word_list.append(sentence)
        f.write(' '.join(sentence)+'\n')

    f.close()



def w2vmodel(corpus, savemodel):
    sentences = word2vec.Text8Corpus(corpus)
    train_model = word2vec.Word2Vec(sentences,
                                    sg=0,  # 0为CBOW  1为skip-gram
                                    size=HIDDEN,  # 特征向量的维度
                                    window=2,  # 表示当前词与预测词在一个句子中的最大距离是多少
                                    min_count=5,  # 词频少于min_count次数的单词会被
                                    sample=1e-3,  # 高频词汇的随机降采样的配置阈值
                                    iter=23,  # 训练的次数
                                    hs=1,  # 为 1 用hierarchical softmax   0 negative sampling
                                    workers=8  # 开启线程个数
                                    )

    train_model.save(savemodel)


def sentence_embed(savemodel, root):
    data = pd.read_pickle('ICWSM19_data/{}.pkl'.format(root))
    w2vmodel = word2vec.Word2Vec.load(savemodel)

    #print(data.iloc[-1] , data.shape)
    word_list = []

    with codecs.open('{}.txt'.format(root), 'r', "utf-8") as f:
        for line in f.readlines():
            word_list.append(line.strip().split(' '))

    countlist = []

    for i in range(len(word_list)):
        count = Counter(word_list[i])
        countlist.append(count)

    print(countlist[:10])

    video_list = data['video_id'].drop_duplicates().values.tolist()

    new_data = data[['body', 'commenter_id', 'offset', 'video_id']]
    new_data.insert(new_data.shape[1], 'sentence_embed', [np.zeros(HIDDEN) for _ in range(new_data.shape[0])])

    for i in range(new_data.shape[0]):
        sentence = clean_words(new_data['body'].iloc[i])
        sentence_embed = np.zeros(HIDDEN)
        count = Counter(sentence)
        #tfidf_dict = {word: tfidf(word, count, countlist) for word in count}
        cnt = 0
        for word in sentence:
            if word in w2vmodel.wv:
                tmp = np.array(w2vmodel.wv[word]) #*tfidf_dict[word]
                cnt += 1
                sentence_embed = sentence_embed + tmp
        if cnt != 0:
            sentence_embed = sentence_embed / cnt
        new_data['sentence_embed'].iat[i] = sentence_embed
        if i%50000 == 0:
            print(i, '/', new_data.shape[0])

    new_data.to_pickle('{}_sentence_embedding.pkl'.format(root))


def main():
    load_model = word2vec.Word2Vec.load(savemodel)

    print("Total vocabulary number:", len(load_model.wv.vocab))

    s_query_word1 = 'lul*'
    s_query_word2 = 'haha*'
    s_query_word3 = 'missed'

    f_word_sim1 = load_model.wv.similarity(s_query_word1, s_query_word2)
    f_word_sim2 = load_model.wv.similarity(s_query_word2, s_query_word3)
    f_word_sim3 = load_model.wv.similarity(s_query_word1, s_query_word3)
    np_word1 = load_model.wv[s_query_word1]
    np_word2 = load_model.wv[s_query_word2]
    np_word3 = load_model.wv[s_query_word3]

    print(s_query_word1, s_query_word2, s_query_word3)
    print("Similarity between 1 & 2", f_word_sim1)
    print("Similarity between 2 & 3", f_word_sim2)
    print("Similarity between 1 & 3", f_word_sim3)

    print(np_word1, np_word2, np_word3)

    print("Similar words of 1")
    for s_word, f_sim in load_model.wv.most_similar(s_query_word1):
        print(s_word, f_sim)

    print("Similar words of 2")
    for s_word, f_sim in load_model.wv.most_similar(s_query_word2):
        print(s_word, f_sim)

    print("Similar words of 3")
    for s_word, f_sim in load_model.wv.most_similar(s_query_word3):
        print(s_word, f_sim)

if __name__ == '__main__':
    root = 'admiralbulldog'
    #savemodel = 'train.model'
    #pkl2txt(root)
    data = pd.read_pickle('ICWSM19_data/{}.pkl'.format(root))
    print(data[:10])
    print(data['commenter_type'].drop_duplicates().values.tolist())
    #w2vmodel('{}.txt'.format(root), savemodel)
    #main()
    #sentence_embed(savemodel, root)
    #data = pd.read_pickle('{}_sentence_embedding.pkl'.format(root))
    print(data[:10])



'''
print(data[['body', 'fragments']][:100].values.tolist())
print(data['fragments'][:100].values.tolist())

commenter_list = data['commenter_id'].values.tolist()

video_list = data['video_id'].drop_duplicates().values.tolist()

print('video list:', len(video_list), video_list)


d = dict(Counter(commenter_list))

print(len(d))

d1={k:v for k,v in d.items()  if v>1000}

print(">1000:", len(d1))

d2={k:v for k,v in d.items()  if v>100}

print(">100:", len(d2))


audience_list = []
for video_id in video_list:
    audience = data[data['video_id']==video_id]['commenter_id'].drop_duplicates().values.tolist()
    audience_list.append(audience)

result = set(audience_list[0])
for i in range(len(audience_list)-1):
    result = set(result.intersection(set(audience_list[i+1])))

print("result:", len(result), result)
'''