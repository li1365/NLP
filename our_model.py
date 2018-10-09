import numpy as np 
import pandas as pd 
import math, scipy
import random, string, copy

#take the txt file, return dictionary with three keys: tokens, POS tags and NER tags
def read_data(filename):
    lines = open(filename, "r").read().split("\r\n")
    tokens = []
    pos = []
    ner = []
    try:
        assert len(lines)%3 == 0
    except:
        print("length of txt file can't be divided by 3, plz check file")
        return
    for i in range(len(lines)//3):
        tokens.append(lines[3*i].split("\t"))
        pos.append(lines[3*i+1].split("\t"))
        ner.append(lines[3*i+2].split("\t"))
    assert len(tokens) == len(pos) and len(pos) == len(ner)
    res = pd.DataFrame({"tokens": tokens, "pos": pos, "ner": ner})
    return res

# get the bigram dictionary for one column (named col_name) in the dataframe df
def get_bigram(df, col_name):
    bigram_dict = {}
    bigram_num = 0
    for tmp in range(len(df)):
        sentence = df.iloc[tmp, col_name]
        for i in range(len(sentence)-1):
            if sentence[i] in bigram_dict:
                if sentence[i+1] in bigram_dict[sentence[i]]:
                    bigram_dict[sentence[i]][sentence[i+1]] += 1
                else:
                    bigram_dict[sentence[i]][sentence[i+1]] = 1
                    bigram_num += 1
            else:
                bigram_dict[sentence[i]] = {}
                bigram_dict[sentence[i]][sentence[i+1]] = 1
                bigram_num += 1
    return bigram_dict, bigram_num

# deal with unknown words, use the first occurances of all words as UNKNOWN
# create a new column called "col_name_unknown" that is processed tokens
# return the new dataframe
def tokensWithUnk(df, col_name):
    tokens_used = set()
    # tokens_unk = copy.deepcopy(tokens)
    new_col = col_name + "_unknown"
    df[new_col] = df[col_name]
    for i in range(len(df)):
        curr = df.iloc[i, new_col]
        for j in range(len(curr)):
            if curr[j] not in tokens_used:
                tokens_used.add(curr[j])
                curr[j] = u'UNK'
        df.iloc[i, new_col] = curr
    return df

# add-k smoothing, return smoothed dictionary
def convert_add_k(bigram_with_unknown, k=0.1):
    for key in bigram_with_unknown.keys():
        for key2 in bigram_with_unknown[key].keys():
            bigram_with_unknown[key][key2] = bigram_with_unknown[key][key2] + k
    return bigram_with_unknown

"""
get P(word|tag) dictionary, tag can be POS or NER tags, with default add-k smoothing k - 0.01
return two dictionaries:
    one with key of tags, and value being dictionary of word and their counts (similar to bigram structure)
    one with key of tags and values being the occurances of that tag
TODO: clarify the definition of tags (whether to include BIO?????)
"""
def get_word_tag_prob(df, tag = "ner", k = 0.01):
    word_tag_dict = dict()
    total_count = dict()
    for doc_idx in range(len(df)):
        curr_tags = df.iloc[doc_idx, tag]
        curr_words = df.iloc[doc_idx, "tokens"]
        assert len(curr_tags) == len(curr_words)
        for i in range(len(curr_tags)):
            tag = curr_tags[i]
            word = curr_words[i]
            if tag not in word_tag_dict: # new tag 
                word_tag_dict[tag] = dict()
                total_count[tag] = k 
            # dic already has tag
            if word not in word_tag_dict[tag]: # new word for current tag
                word_tag_dict[tag][word] = k # apply add-k smoothing
            word_tag_dict[tag][word] += 1
            total_count[tag] += 1
    return word_tag_dict, total_count

"""
use the viterbi algorithm for predicting NER tag for sequence of tokens
inputs: test_seq (list of tokens), word_tag_prob(dict), tag_bigram (dict)
output: list of tags
"""
def viterbi_hmm(test_seq, training_df, tag_bigram):
    word_tag_prob, tag_counts = get_word_tag_prob(training_df)
    tags = tag_counts.keys()
    n = len(test_seq)
    scores = np.zeros([len(tag_counts), n]) # dp table
    backpointers = np.zeros([len(tag_counts), n]) 
    for i in range(len(tags)): # initialization
        scores[i][0] = tag_counts[tags[i]]/ np.sum(tag_counts.values())
        backpointers[i][0] = 0 
    for word_idx in range(1, n): # dp step
        for tag_idx in range(len(tags)):
            tmp_max = 0
            max_idx = -1
            for prev_tag in range(len(tags)):
                transition = tag_bigram[tags[prev_tag]][tags[tag_idx]] #FIXME: fix the bigram to return prob. 
                lexical = word_tag_prob[tags[tag_idx]][test_seq[word_idx]] / tag_counts[tags[tag_idx]]
                curr = scores[prev_tag][word_idx-1]*transition*lexical
                if (curr > tmp_max):
                    tmp_max = curr
                    max_idx = tag_idx
            scores[tag_idx][word_idx] = tmp_max
            assert max_idx != -1
            backpointers[tag_idx][word_idx] = max_idx
    # now figure out the maximum path (i.e. predicted tags)
    tag_preds = ["" for i in range(n)]
    max_tag_idx = np.argmax(scores[n-1])
    while (n > 0): 
        tag_preds[n-1] = tags[max_tag_idx]
        max_tag_idx = backpointers[n-1][max_tag_idx]
        n -= 1
    return tag_preds

"""
use MEMM for predicting tag labels
"""
def viterbi_memm(test_seq, training_df, tag_bigram):
    n = len(test_seq)


    tag_preds = ["" for i in range(n)]
    return tag_preds

#TODO: split data into 80% training and 20% validation
def split_training(tokens, pos, ner):
    ratio = 0.8
    # indices = []
    
    return ratio

raw = read_data("train.txt")
print("tokens")
print(type(raw.loc[0, "tokens"]))
print("POS")
print(raw.loc[0:10, "pos"])
print("NER")
print(raw.loc[0:10, "ner"])

#TODO: write the test case for HMM predictions
