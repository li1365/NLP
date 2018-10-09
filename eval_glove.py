

# import tensorflow as tf
# tf.test.gpu_device_name()

import pickle
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# !pip install -U -q PyDrive
 
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials

# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)

import random, string, copy
import numpy as np
import scipy.spatial as sp
import pandas as pd

# train_trump = drive.CreateFile({'id': '14dLJCHpBZsFDASs5DIp0WZuzOnKMxwik'})
# train_trump.GetContentFile('train_trump.txt')
# train_obama = drive.CreateFile({'id': '1633StDSVWWMnlduee8QycuhSLxluT3E0'})
# train_obama.GetContentFile('train_obama.txt')
# dev_trump = drive.CreateFile({'id': '17hJRp9B432ugnbcKcDt38KJQAdcSV6AF'})
# dev_trump.GetContentFile('dev_trump.txt')
# dev_obama = drive.CreateFile({'id': '16-afZ200h1KwglqFwCgygFDYAy6E53dG'})
# dev_obama.GetContentFile('dev_obama.txt')
# test = drive.CreateFile({'id': '1RJbTRfma8tyJyMUGbsyy9QRVpazJN7c0'})
# test.GetContentFile('test.txt')

# analogy_test = drive.CreateFile({'id': '1GSuLyBxru2S4jyJpwtdJSSHTw961Lab4'})
# analogy_test.GetContentFile('analogy_test.txt')


# train_trump = remove_quate(open('train_trump.txt', 'r').read()).split()
# train_obama = remove_quate(open('train_obama.txt', 'r').read()).split()
# dev_trump = remove_quate(open('dev_trump.txt', 'r').read()).split()
# dev_obama = remove_quate(open('dev_obama.txt', 'r').read()).split()
# test = remove_quate(open('test.txt', 'r').read()).split()

import nltk
nltk.download('punkt')
# from nltk.tokenize import word_tokenize
# tokens = word_tokenize(train_trump)
# print(tokens[:10])


def uniGram(tokens):
  unigram_dict = {}
  for token in tokens:
    if token in unigram_dict:
      unigram_dict[token] += 1
    else:
      unigram_dict[token] = 1
  return unigram_dict, len(tokens)

def biGram(tokens):
  bigram_dict = {}
  bigram_num = 0
  for i in range(len(tokens)-1):
    if tokens[i] in bigram_dict:
      if tokens[i+1] in bigram_dict[tokens[i]]:
        bigram_dict[tokens[i]][tokens[i+1]] += 1
      else:
        bigram_dict[tokens[i]][tokens[i+1]] = 1
        bigram_num += 1
    else:
      bigram_dict[tokens[i]] = {}
      bigram_dict[tokens[i]][tokens[i+1]] = 1
      bigram_num += 1
  return bigram_dict, bigram_num

# bitable, counts = biGram(train_trump)
# print(bitable['yes']['.'], counts, "prob: " , bitable["yes"]['.'] / counts, )
# print(sum(bitable['yes'].values()))

def random_bigramNext(bigram_dict):
  return random.choices(list(bigram_dict.keys()), weights=list(bigram_dict.values()))[0]

# print(random_bigramNext(bitable['you']))

def merge_bigramDict(dict1, dict2):
  dict_merged = {}
  for key in dict1:
    if key in dict2:
      dict_merged[key] = dict1[key] + dict2[key]
    else:
      dict_merged[key] = dict1[key]
  for key in dict2:
    if key not in dict_merged:
      dict_merged[key] = dict2[key]
  return dict_merged

# print(len(bitable['.']))
# print(len(merge_bigramDict(bitable['.'], bitable['?'])))

def merge_EoSP(bigram_dict): #End of sentence punctuation
  return merge_bigramDict(merge_bigramDict(bigram_dict['.'], bigram_dict['!']), bigram_dict['?'])

def bisentenceGenerator(tokens, sentence = '', max_length = 60):
  print("computing bigrams and generating random sentence: ")
  bigram_dict,_ = biGram(tokens)
  if len(sentence) == 0:
    last = random_bigramNext(merge_EoSP(bigram_dict))
  else:
    sentence = nltk.word_tokenize(sentence)
    last = sentence[-1]
    temp = ''
    for x in sentence:
      temp = temp + ' ' + x
    sentence = temp
  for i in range(max_length):  
    if last in bigram_dict:
      next = random_bigramNext(bigram_dict[last])
    else:
      next = random.choice(list(bigram_dict.keys()))[0]
    if next == ('.' or '!' or '?'):
      return sentence + ' ' + next
    sentence = sentence + ' ' + next
    last = next
  return sentence

# print(bisentenceGenerator(train_trump, "you"))

def unisentenceGenerator(tokens):
  unigram_dict,_ = uniGram(tokens)
  return random.choice(list(unigram_dict.keys()))[0]

"""## SMOOTHED N-GRAMS"""

def tokensWithUnk(tokens):
  tokens_used = set()
  tokens_unk = copy.deepcopy(tokens)
  for i in range(len(tokens_unk)):
    if tokens_unk[i] not in tokens_used:
      tokens_used.add(tokens_unk[i])
      tokens_unk[i] = u'UNK'
  return tokens_unk

"""### Good Turing for count less than 6 (count_default) in bigram dict"""

def GT(tokens, count_default = 6):
  num_uni = len(tokens)
  bigram_dict, num_bi = biGram(tokens)
  num_bi_all = num_uni ** 2
  num_bi_unseen = num_bi_all - num_bi
  count_list = [0] * (count_default + 1)
  for word1 in bigram_dict:
    for word2 in bigram_dict[word1]:
      if bigram_dict[word1][word2] <= count_default:
        count_list[bigram_dict[word1][word2]] += 1
  count_list[0] = num_bi_unseen
  GT_list = []
  for i in range(count_default):
    GT_list.append((i + 1) * float(count_list[i+1]) / float(count_list[i]))
  return GT_list

# print(GT(train_trump))

def get_GT(word1, word2, GT_list, unigram_dict, bigram_dict):
  count = 0
  count = bigram_dict[word1][word2] if word2 in bigram_dict[word1] else count
  count = GT_list[count] if count < len(GT_list) else count
  return float(count) / float(unigram_dict[word1])

import math

def get_perplexity(train_tokens, dev_tokens, GT_list, unigram_dict, bigram_dict):
  dev_tokens[0] = u'UNK' if dev_tokens[0] not in unigram_dict else dev_tokens[0]
  prob = math.log(float(unigram_dict[dev_tokens[0]]) / float(len(train_tokens)))
  for i in range(len(dev_tokens)-1):
    dev_tokens[i+1] = u'UNK' if dev_tokens[i+1] not in unigram_dict else dev_tokens[i+1]
    prob += get_GT(dev_tokens[i], dev_tokens[i+1], GT_list, unigram_dict, bigram_dict) * (-1)
  return math.exp(prob / len(dev_tokens))



"""## TEXT CLASSIFICATION - SECTION 6"""

def token_lines(filename):
  lines = remove_quate(open(filename, 'r').read()).split("\n")
  res = []
  for line in lines:
    if len(line) > 0:
      res.append(line.split())
  return res

# #tokens with unk
# train_trump_unk = tokensWithUnk(train_trump)
# train_obama_unk = tokensWithUnk(train_obama)

# #combined tokens for both genre
# train_tokens = [train_trump, train_obama]
# unigram_dict_trump,_ = uniGram(train_trump_unk)
# unigram_dict_obama,_ = uniGram(train_obama_unk)
# unigram_dicts = [unigram_dict_trump, unigram_dict_obama]
# bigram_dict_trump,_ = biGram(train_trump_unk)
# bigram_dict_obama,_ = biGram(train_obama_unk)
# bigram_dicts = [bigram_dict_trump, bigram_dict_obama]
# GT_lists = [GT(train_trump), GT(train_obama)]
# dev_lines = [token_lines('dev_trump.txt'), token_lines('dev_obama.txt')]
# test = token_lines('test.txt')

def simpleClassification(dev_tokens, tarin_tokens, GT_lists, unigram_dicts, bigram_dicts):
  score = {}
  score['trump'] = get_perplexity(train_tokens[0], dev_tokens, GT_lists[0], unigram_dicts[0], bigram_dicts[0])
  score['obama'] = get_perplexity(train_tokens[1], dev_tokens, GT_lists[1], unigram_dicts[1], bigram_dicts[1])
  return min(score, key = score.get)

# def devAccuracy(lines, label, train_tokens, GT_lists, unigram_dicts, bigram_dicts):
#   correct = 0
#   for line in lines:
#     if simpleClassification(line, train_tokens, GT_lists, unigram_dicts, bigram_dicts) == label:
#       correct += 1
#   print(str(correct) +'/' + str(len(lines)))
#   return float(correct) / float(len(lines))

# print(dev_lines[0])

# accuracy_trump = devAccuracy(dev_lines[0], 'trump', train_tokens, GT_lists, unigram_dicts, bigram_dicts)
# accuracy_obama = devAccuracy(dev_lines[1], 'obama', train_tokens, GT_lists, unigram_dicts, bigram_dicts)

def testClassify(test, train_tokens, GT_lists, unigram_dicts, bigram_dicts):
  res = np.zeros((len(test),2),dtype='int32')
  index = 0
  for line in test:
    res[index,0] = index
    res[index,1] = 0 if simpleClassification(line, train_tokens, GT_lists, unigram_dicts, bigram_dicts) == 'obama' else 1
    index += 1
  df = pd.DataFrame(res, columns=['Id','Prediction'])
  df.to_csv('submission.csv', index = False)

# testClassify(test, train_tokens, GT_lists, unigram_dicts, bigram_dicts)

# from google.colab import files
# files.download('submission.csv')

# listx = [1,2,3]
# def m(listx):
#   listx[1] = 6
#   return listx
# listy = m(listx)
# print(listx)
# print(listy)

"""## WORD EMBEDDING - SECTION 7"""

analogy_test = open("Assignment1_resources/analogy_test.txt", "r").read().split("\n") 
analogy_test = analogy_test[:-1] # last line is empty


"""- google word2vec doc link: https://radimrehurek.com/gensim/test/utils.html
- http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
"""

import gensim

# gg_file = drive.CreateFile({'id': '1bqNbACJmiNlfQhu77mKCdPLEDrccgoOE'})
# gg_file.GetContentFile('gg_model.bin')

from gensim.models import KeyedVectors
# gg_model = KeyedVectors.load_word2vec_format("gg_model.bin", binary=True)  # C binary format

# uses gensim pkg's "similar_by_vector" method
# the embedding model has to be generated by gensim
# test file is a list whose item consist of one line in the raw txt file
def eval_embedding(test_file, embedding):
  acc = []
  sim_score = []
  for i in range(len(test_file)):
    words = test_file[i].split() # word a, b, c, d
    tmp_input = embedding[words[1]] - embedding[words[0]] + embedding[words[2]] # word b-a+c
    tmp_sims = embedding.similar_by_vector(tmp_input)
    for sim in tmp_sims:
      curr_word, curr_score = sim[0], sim[1]
      if (curr_word) not in words[0:3]: # the most similar word is not in the original three words
        pred = curr_word
        sim_score.append(curr_score)
        break
    if i%500 == 0:
          print("current i is", i)
    if (pred == words[3]): 
      acc.append(1)
    else:
      acc.append(0)    
  return acc, sim_score

def segment_test(test_file, embedding):
  inputs = np.zeros([len(test_file),300])
  target = np.zeros([len(test_file),300])
  for i in range(len(test_file)):
    words = test_file[i].split() # word a, b, c, d
    inputs[i,:] = embedding[words[1]] - embedding[words[0]] + embedding[words[2]] # word b-a+c
    target[i,:] = embedding[words[3]]
  return inputs, target

# gg_eval.to_csv("google_evaluation.csv", encoding = "utf-8")

# from google.colab import files
# files.download("google_evaluation.csv")

"""- glove model load reference: https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python"""

# glove_file = drive.CreateFile({'id': '1W6uTrRdI72PEKGneh2y0-pqbRGjvH2k2'})
# glove_file.GetContentFile('glove_file.txt')

## load glove model using gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

print("before loading glove model")
tmp_file = get_tmpfile("test_word2vec.txt")

# call glove2word2vec script
# default way (through CLI): python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec("glove.840B.300d.txt", tmp_file)
print("glove model converted to word2vec format")

model = KeyedVectors.load_word2vec_format(tmp_file)

print("glove model loaded successfully")
glove_acc, glove_sim = eval_embedding(analogy_test, model)

print(np.mean(glove_acc), np.mean(glove_sim))

glove_eval = pd.DataFrame({'test':analogy_test, "acc": glove_acc, "similarity_score": glove_sim})
glove_eval.to_csv("glove_evaluation.csv")

# import numpy as np

# from numpy import linalg as LA
# def cosine_similarity(arr1, arr2):
#   return np.dot(arr1, arr2)/(LA.norm(arr1)*LA.norm(arr2))


"""## GET A SMALLER WORD EMBEDDING DICTIONARY"""

# raw_train_trump = open('train_trump.txt', 'r').read().split()
# raw_train_obama = open('train_obama.txt', 'r').read().split()
# raw_dev_trump = open('dev_trump.txt', 'r').read().split()
# raw_dev_obama = open('dev_obama.txt', 'r').read().split()
# raw_test = open('test.txt', 'r').read().split()

# google_dict = dict()
# def get_embedding(text_list, dic, embedding):
#   for text in text_list:
# #     print(text)
#     for word in text:
#       if word not in dic.keys():
#         try:
#           dic[word] = embedding[word]
#         except:
#           pass
#   return dic

# glove_dict = dict()

# doc_list = [raw_train_trump, raw_train_obama, raw_dev_trump, raw_dev_obama, raw_test]
# google_dict = get_embedding(doc_list, glove_dict, model)

# len(google_dict.keys())

# len(google_dict.keys())

# pkl_gg = drive.CreateFile({'id': '134Vx8qPBn9hapaK_-RQyi-5Bn1dx4hI4'})
# pkl_gg.GetContentFile('glove.pkl')

# test = load_obj("gg.pkl")
# len(test.keys())

# save_obj(google_dict, "glove_small")

# from google.colab import files
# files.download('glove_small.pkl')

# from google.colab import files

# with open('example.txt', 'w') as f:
#   f.write('some content')

# files.download('example.txt')

"""### other analogy test"""

# TASK 1: get cosine_similarity(d, b-a+c)
# get a similarity np array that uses embedding dictionary
# def analogy_task1(test_file, embedding):
# #   res = np.zeros(len(test_file), dtype = float)
#   res = dict()
#   for i in range(len(test_file)):
#     words = test_file[i].split() # word a, b, c, d
#     if len(words) == 4:
#       # cosine sim = cosine(d, b - a + c)
#       try:
#         res[words[0]] = cosine_similarity(embedding[words[3]], embedding[words[1]] - embedding[words[0]] + embedding[words[2]])
#       except:
#         print(words)
#         res[words[0]] = -2
#   return res

# TASK 2: get cosine_similarity
# get a similarity np array that uses embedding dictionary
# def get_similarity_score(test_file, embedding):
# #   res = np.zeros(len(test_file), dtype = float)
#   res = dict()
#   for i in range(len(test_file)):
#     words = test_file[i].split() # word a, b, c, d
#     if len(words) == 4:
#       # cosine sim = cosine(d, b - a + c)
#       try:
#         res[words[0]] = cosine_similarity(embedding[words[3]], embedding[words[1]] - embedding[words[0]] + embedding[words[2]])
#       except:
#         print(words)
#         res[words[0]] = -2
#   return res

# """## CLASSIFICATION WITH WORD EMBEDDING - Section **8**"""

# gg_small = drive.CreateFile({'id':"134Vx8qPBn9hapaK_-RQyi-5Bn1dx4hI4"})
# gg_small.GetContentFile('gg_small.pkl')
# gg_model = load_obj("gg_small.pkl")

# def text2vec(speech, embedding):
#   res = np.zeros([1,300]);
#   for word in speech:
#     try:
#       res = np.append(res, embedding[word].reshape([1,300]))
#     except:
#       pass
#   res = res[1:, :]
#   return np.mean(res, axis = 0)

# def token_lines(filename):
#   lines = remove_quate(open(filename, 'r').read()).split("\n")
#   res = []
#   for line in lines:
#     if len(line) > 0:
#       res.append(line.split())
#   return res

# """## FREE STYLE CLASSIFICATION"""

# np.ones([2,3])[1:,:]

