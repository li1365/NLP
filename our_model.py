import numpy as np 
import pandas as pd 
import math, scipy
import random, string, copy

# take the txt file, return dictionary with three keys: tokens, POS tags and NER tags
def read_data(filename):
    with open(filename, "r") as f:
        lines = f.read().split("\n")
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
    res = df.copy()
    tokens_used = set()
    # tokens_unk = copy.deepcopy(tokens)
    for col_name in col_names:
        if "ner" not in col_name:
            new_col = col_name + "_unknown"
            res[new_col] = res[col_name]
            for i in range(len(res)):
                curr = res.loc[i, new_col]
                for j in range(len(curr)):
                    if curr[j] not in tokens_used:
                        tokens_used.add(curr[j])
                        curr[j] = 'UNK'
                res.loc[i, new_col] = curr
    return res.reset_index()

# remove the BIOs from NER tags, adds "short_ner" column to the dataframe
def strip_bio(df, col_name):
    res = df.copy()
    res["short_ner"] = res[col_name]
    for i in range(len(res)):
        curr = res.loc[i, "short_ner"]
        for j in range(len(curr)): 
            if "-" in curr[j]:
                sep = curr[j].index("-")
                curr[j] = curr[j][sep+1:len(curr[j])]
        res.at[i, "short_ner"] = curr
    return res

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
    # print("keys are ", bigram_dict.keys())
    for key in bigram_dict.keys():
        curr_total = sum(bigram_dict[key].values())
        bigram_dict[key]["UNK"] = k # deal with unseen bigrams
        # compute the conditional prob. 
        for word in bigram_dict[key].keys(): 
            bigram_dict[key][word] = bigram_dict[key][word]/curr_total
    return bigram_dict

# a = 

"""
get P(word|tag) dictionary, tag can be POS or NER tags, with default add-k smoothing k - 0.01
return two dictionaries:
    one with key of tags, and value being dictionary of word and their counts (similar to bigram structure)
    one with key of tags and values being the occurances of that tag
TODO: clarify the definition of tags (whether to include BIO?????)
"""
def get_word_tag_prob(df, tag_col = "short_ner", k = 0.01):
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
# def viterbi_hmm(test_seq, training_df, tag_bigram):
def viterbi_hmm(test_seq, word_tag_prob, tag_counts, tag_bigram):
    # word_tag_prob, tag_counts = get_word_tag_prob(training_df)
    tags = list(tag_counts.keys())
    # print("tags are ", tags)
    n = len(test_seq)
    scores = np.zeros([len(tags), n]) # dp table
    backpointers = np.zeros([len(tags), n]) 
    # print("tag counts are ")
    # print(tag_counts)
    for i in range(len(tags)): # initialization
        # tag_prob = tag_counts[tags[i]] / sum(tag_counts.values())
        tag_prob = i + 1
        print("current tag prob", tag_prob)
        if test_seq[0] in word_tag_prob[tags[i]]:
            scores[i][0] = tag_prob * word_tag_prob[tags[i]][test_seq[0]]/ sum(word_tag_prob[tags[i]].values())
            # print("seen score")
            # print(scores[i][0])
        else: 
            scores[i][0] = tag_prob * word_tag_prob[tags[i]]["UNK"]/sum(word_tag_prob[tags[i]].values())
            # print("unknown score")
            # print(scores[i][0])
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
                    lexical = word_tag_prob[tags[tag_idx]][test_seq[word_idx]] / sum(word_tag_prob[tags[tag_idx]].values())
                else: # FIXME: distinguish between unseen pairs and unknown words???
                    lexical = word_tag_prob[tags[tag_idx]]["UNK"] / sum(word_tag_prob[tags[tag_idx]].values())
                curr = scores[prev_tag][word_idx-1]*transition*lexical
                if (curr > tmp_max):
                    tmp_max = curr
                    max_idx = prev_tag
            scores[tag_idx][word_idx] = tmp_max
            assert max_idx != -1
            backpointers[tag_idx][word_idx] = max_idx
    # now figure out the maximum path (i.e. predicted tags)
    tag_preds = ["" for i in range(n)]
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

# take the file name and return a df with raw and unknown tokens and pos tags
def read_test(test_filename):
    print("reading test file...")
    raw_df = read_data(test_filename)
    unknown_cols = ["tokens", "pos"]
    res_df = tokensWithUnk(raw_df, unknown_cols)
    return res_df

"""
returns a dictionary that can be used to generate the submission file
"""
def predict_test(model, training_df, test_filename):
    test = read_test(test_filename)
    ner_bigram = get_bigram(training_df, "short_ner")
    word_tag_prob, tag_counts = get_word_tag_prob(training_df) # for hmm
    word_tag_dict = word_MLE(raw_with_unknown) # for baseline
    indices = []
    preds = []
    print("getting predictions......")
    for i in range(len(test)):
        test_tokens = test.loc[i, "tokens_unknown"]
        indices += test.loc[i, "ner"][0].split(" ")
        if (model == "hmm"):
            curr_preds = viterbi_hmm(test_tokens, word_tag_prob, tag_counts, ner_bigram)
        elif model == "baseline": #TODO: fill in other options for modeling
            curr_preds = baseline_predict(test_tokens, word_tag_dict)
        # remove the BIOs
        for i in range(len(curr_preds)):
            if "-" in curr_preds[i]:
                tmp = curr_preds[i].index("-")
                curr_preds[i] = curr_preds[i][(tmp+1):len(curr_preds[i])]
        preds += curr_preds
    # reformat the preds for submission
    assert len(indices) == len(preds)
    submission = {"LOC":[], "PER":[], "ORG":[], "MISC":[]}
    i = 0
    while i < len(indices):
        curr_pred = preds[i]
        if curr_pred in submission.keys(): # NER tag is not O
            start = i
            while (i+1) < len(indices) and preds[i] == preds[i+1]:
                i += 1
            end = i
            tmp = str(start) + "-" + str(end)
            submission[curr_pred].append(tmp)
        i += 1
    return submission

# take the submission dictionary and convert it to a txt file with the inputted filename
def get_submission(submission, filename):
    print("generating submission file....")
    file = open("/Users/JanicaTang/Desktop/NLP/" + filename, "w")
    tags = ["ORG", "MISC", "PER", "LOC"]
    file.write("Type,Prediction\n")
    for i in range(len(tags)):
        tmp = " ".join(submission[tags[i]])
        file.write(tags[i] + "," + tmp + "\n")
    file.close()

# file = open("/Users/JanicaTang/Desktop/NLP/" + "submission.txt", "w")
#TODO: split data into 80% training and 20% validation
def split_training(tokens, pos, ner):
    return ratio

print("converting txt file to dataframe......")

# returns the tag counts for each word
def word_MLE(df):
    word_tags = dict()
    for i in range(len(df)):
        curr_tags = df.loc[i, "short_ner"]
        curr_words = df.loc[i, "tokens_unknown"]
        for j in range(len(curr_tags)):
            if curr_words[j] in word_tags.keys():
                if curr_tags[j] in word_tags[curr_words[j]]:
                    word_tags[curr_words[j]][curr_tags[j]] += 1
                else: 
                    word_tags[curr_words[j]][curr_tags[j]] = 1
            else: 
                word_tags[curr_words[j]] = dict()
                word_tags[curr_words[j]][curr_tags[j]] = 1
    return word_tags

def baseline_predict(test_seq, word_tags):
    preds = []
    for word in test_seq:
        if word in word_tags:
            curr = max(word_tags[word], key=word_tags[word].get)
        else:
            curr = max(word_tags["UNK"], key = word_tags["UNK"].get)
        preds.append(curr)
    assert len(preds) == len(test_seq)
    return preds

"""
preprocessing units...
"""
# raw = read_data("train.txt")
# print("raw training ", raw.columns)
# raw_with_unknown = strip_bio(raw, "ner")
# print("after removing BIOs ", raw_with_unknown.columns)
# raw_with_unknown.to_csv("tmp.csv", index = False)
# unknown_cols = ["tokens"]
# raw_with_unknown = tokensWithUnk(raw_with_unknown, unknown_cols)
# print("after dealing with unknown")
# print(raw_with_unknown.columns)
# raw_with_unknown.to_csv("/Users/JanicaTang/Desktop/NLP/train_data.csv", index = False)
# raw_with_unknown.to_pickle("raw_with_unknown.pkl")

# print("now getting the bigram for NER tags...")
raw_with_unknown = pd.read_pickle("raw_with_unknown.pkl")
word_tag, tag_counts = get_word_tag_prob(raw_with_unknown, "short_ner")
ner_bigrams = get_bigram(raw_with_unknown, "short_ner")

print("the bigram is")
print(ner_bigrams)
print("word tag prob is")

test_tokens = "South	African	fast	bowler	Shaun	Pollock	concluded	his	Warwickshire	career	with	a	flourish".split("\t")
test2 = "Yorkshire	captain	David	Byas	completed	his	third	Sunday	league	century	as	his	side	swept	clear	at	the	top	of	the	table	,	reaching	a	career	best	111	not	out	against	Lancashire	.".split("\t")
pos_test = "NNP	NN	NNP	NNP	VBD	PRP$	JJ	NNP	NN	NN	IN	PRP$	NN	VBD	JJ	IN	DT	NN	IN	DT	NN	,	VBG	DT	NN	JJS	CD	RB	RP	IN	NNP	.".split("\t")
print("the test tokens are ")
print(test_tokens)

print("now predict with viterbi hmm......")
test_res = viterbi_hmm(test_tokens, word_tag, tag_counts, ner_bigrams)
print("the predicted NER tags are")
print(test_res)

print("THE BASE LINEEEEEEEEEE")
word_tag_dict = word_MLE(raw_with_unknown)
base_res = baseline_predict(test_tokens, word_tag_dict)
print(base_res)

# import numpy as np
# a = np.array([1,2,3,4,5,6]).reshape([2,3])
# b = np.zeros([2,3])
# print(a)
# print(b)
# print(a[:,2])
# print(np.argmax(a[:,2]))
"""
generate test submission files
"""
# submission_dict = predict_test("hmm", raw_with_unknown, "test.txt")
baselien_submission = predict_test("baseline", raw_with_unknown, "test.txt")

get_submission(baselien_submission, "baseline_submission.txt")
print("current submission file is completed, saved in submission.txt")