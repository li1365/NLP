import pandas as pd 
import numpy as np 
import pickle
from model import read_data, strip_bio, load_glove, uniGram

EMBEDDING_FILE = "glove.840B.300d.txt"

raw = read_data("train.txt")
raw_withoutBIO = strip_bio(raw, "ner") # ner, pos, tokens, short_ner
raw_withoutBIO = raw_withoutBIO.drop(columns = ["short_ner"])
extra_data = pd.read_pickle("conll2003_combined.pkl")
test_file = read_data("test.txt")
raw_withoutBIO = pd.concat([raw_withoutBIO, extra_data, test_file], ignore_index=True)

all_words = uniGram(raw_withoutBIO, "tokens").keys()
all_words = [x.lower() for x in all_words]
print(len(all_words))
all_words = list(set(all_words))
print(len(all_words))
embedding = load_glove(EMBEDDING_FILE)

word_dict = dict()
matrix = np.zeros([len(all_words), 300])
print("start scanning the smaller words")
unknown_count = 0
for i in range(len(all_words)):
    word_dict[all_words[i]] = i
    try:
        matrix[i] = embedding[all_words[i]]
    except:
        unknown_count += 1
        matrix[i] = np.ones(300)*1/2

print("total unknown word ratio is", unknown_count/len(all_words))
f = open("embedding_matrix.pickle", "wb")
pickle.dump(matrix , f)
f.close()

f = open("word_index.pickle", "wb")
pickle.dump(word_dict , f)
f.close()


