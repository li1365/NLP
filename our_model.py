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

# deal with unknown words, use the first occurances of all words as UNKNOWN
# create a new column called "col_name_unknown" that is processed tokens
# return the new dataframe
def tokensWithUnk(df, col_names):
    tokens_used = set()
    # tokens_unk = copy.deepcopy(tokens)
    for col_name in col_names:
        new_col = col_name + "_unknown"
        df[new_col] = df[col_name]
        for i in range(len(df)):
            curr = df.loc[i, new_col]
            for j in range(len(curr)):
                if curr[j] not in tokens_used:
                    tokens_used.add(curr[j])
                    curr[j] = u'UNK'
            df.loc[i, new_col] = curr
    return df.reset_index()

# remove the BIOs from NER tags, adds "short_ner" column to the dataframe
def strip_bio(df, col_name):
    res = df.copy()
    df["short_ner"] = df[col_name]
    for i in range(len(df)):
        curr = df.loc[i, "short_ner"]
        for i in range(len(curr)): 
            if (len(curr[i]) > 3): # only keep the part after "-"
                try:
                    sep = curr[i].index("-")
                except:
                    print(curr[i])
                    return
                curr[i] = curr[i][sep+1:len(curr)-1]
        print(curr)
        df.loc[i, "short_ner"] = curr
    # print(len(df))
    return df

# get the bigram dictionary for one column (named col_name) in the dataframe df, with add-k smoothing
# return a nested dictionary of probabilities
def get_bigram(df, col_name, k = 0.01):
    bigram_dict = {}
    # scan the entire text and build the raw dictionary
    for tmp in range(len(df)):
        sentence = df.loc[tmp, col_name]
        for i in range(len(sentence)-1):
            if sentence[i] in bigram_dict:
                if sentence[i+1] in bigram_dict[sentence[i]]:
                    bigram_dict[sentence[i]][sentence[i+1]] += 1
                else:
                    bigram_dict[sentence[i]][sentence[i+1]] = 1 + k # apply smoothing
            else:
                bigram_dict[sentence[i]] = {}
                bigram_dict[sentence[i]][sentence[i+1]] = 1 + k
    print("keys are ", bigram_dict.keys())
    for key in bigram_dict.keys():
        # print(bigram_dict[key])
        curr_total = np.sum(bigram_dict[key].values())
        bigram_dict[key]["UNK"] = k # deal with unseen bigrams
        # compute the conditional prob. 
        for word in bigram_dict[key].keys(): 
            bigram_dict[key][word] = bigram_dict[key][word]/curr_total
    return bigram_dict


"""
get P(word|tag) dictionary, tag can be POS or NER tags, with default add-k smoothing k - 0.01
return two dictionaries:
    one with key of tags, and value being dictionary of word and their counts (similar to bigram structure)
    one with key of tags and values being the occurances of that tag
TODO: clarify the definition of tags (whether to include BIO?????)
"""
def get_word_tag_prob(df, tag_col = "ner_unknown", k = 0.01):
    word_tag_dict = dict()
    total_count = dict()
    for doc_idx in range(len(df)):
        curr_tags = df.loc[doc_idx, tag_col]
        curr_words = df.loc[doc_idx, "tokens_unknown"]
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
    assert word_tag_dict.keys() == total_count.keys()
    # deal with unseen word, use the same k to smooth
    for key in word_tag_dict.keys():
        if "UNK" not in word_tag_dict[key].keys():
            word_tag_dict[key]["UNK"] = k
            total_count[key] += k
    print("finished calculating dictionary for P(word|tag)")
    return word_tag_dict, total_count

"""
use the viterbi algorithm for predicting NER tag for sequence of tokens
inputs: test_seq (list of tokens), word_tag_prob(dict), tag_bigram (dict)
output: list of tags
"""
def viterbi_hmm(test_seq, training_df, tag_bigram):
    word_tag_prob, tag_counts = get_word_tag_prob(training_df)
    tags = list(tag_counts.keys())
    n = len(test_seq)
    print("n is ", n)
    scores = np.zeros([len(tags), n]) # dp table
    backpointers = np.zeros([len(tags), n]) 
    for i in range(len(tags)): # initialization
        scores[i][0] = tag_counts[tags[i]]/ np.sum(tag_counts.values())
        backpointers[i][0] = 0 
    for word_idx in range(1, n): # dp step
        for tag_idx in range(len(tags)):
            tmp_max = 0
            max_idx = -1
            for prev_tag in range(len(tags)):
                if (tags[tag_idx] in tag_bigram[tags[prev_tag]].keys()):
                    transition = tag_bigram[tags[prev_tag]][tags[tag_idx]] 
                else: # deal with unseen tag bigrams
                    transition = tag_bigram[tags[prev_tag]]["UNK"] 
                if test_seq[word_idx] in word_tag_prob[tags[tag_idx]]:
                    lexical = word_tag_prob[tags[tag_idx]][test_seq[word_idx]] / tag_counts[tags[tag_idx]]
                else: # FIXME: distinguish between unseen pairs and unknown words???
                    lexical = word_tag_prob[tags[tag_idx]]["UNK"] / tag_counts[tags[tag_idx]]
                curr = scores[prev_tag][word_idx-1]*transition*lexical
                if (curr > tmp_max):
                    tmp_max = curr
                    max_idx = tag_idx
            scores[tag_idx][word_idx] = tmp_max
            assert max_idx != -1
            backpointers[tag_idx][word_idx] = max_idx
    # now figure out the maximum path (i.e. predicted tags)
    tag_preds = ["" for i in range(n)]
    # print(scores)
    # print(last_scores.shape)
    # print(np.argmax(scores, axis = 0))
    # print("back pointers")
    # print(backpointers)
    max_tag_idx = np.argmax(scores, axis = 0)[-1]
    while (n > 0): 
        # print("max tag", max_tag_idx)
        tag_preds[n-1] = tags[int(max_tag_idx)]
        max_tag_idx = backpointers[int(max_tag_idx)][n-1]
        n -= 1
    return tag_preds

"""
TODO: use MEMM for predicting tag labels
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

print("converting txt file to dataframe......")
raw = read_data("train.txt")
# print("tokens")
# print(type(raw.loc[0, "tokens"]))
# print("POS")
# print(raw.loc[0:10, "pos"])
# print("NER")
# print(raw.loc[0:10, "ner"])

unknown_cols = ["ner", "tokens"]
raw_with_unknown = tokensWithUnk(raw, unknown_cols)
print(type(raw_with_unknown.loc[0, "tokens_unknown"]))
raw_with_unknown.to_pickle("raw_with_unknown.pkl")

# raw_with_unknown = strip_bio(raw, "ner")

# print(raw_with_unknown.head())
print("dealt with unknown words in", unknown_cols, " columns and the current columns in the dataframe are:", raw_with_unknown.columns)

# print("now getting the bigram for NER tags...")
# ner_bigrams = get_bigram(raw_with_unknown, "ner_unknown")

# test_tokens = "South	African	fast	bowler	Shaun	Pollock	concluded	his	Warwickshire	career	with	a	flourish".split("\t")
# test2 = "Yorkshire	captain	David	Byas	completed	his	third	Sunday	league	century	as	his	side	swept	clear	at	the	top	of	the	table	,	reaching	a	career	best	111	not	out	against	Lancashire	.".split("\t")
# print("the test tokens are ")
# print(test2)

# print("now predict with viterbi hmm......")
# test_res = viterbi_hmm("China and Spain", raw_with_unknown, ner_bigrams)
# print(viterbi_hmm("Donald Trump", raw_with_unknown, ner_bigrams))
# print("the predicted NER tags are")
# print(test_res)

# import numpy as np
# a = np.array([1,2,3,4,5,6]).reshape([2,3])
# b = np.zeros([2,3])
# print(a)
# print(b)
# print(a[:,2])
# print(np.argmax(a[:,2]))
