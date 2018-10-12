import numpy as np
import pandas as pd
from nltk import ngrams
from nltk.classify import MaxentClassifier
import pickle

# take the txt file, return dictionary with three keys: tokens, POS tags and NER tags
def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
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
    #print("keys are ", bigram_dict.keys())
    for key in bigram_dict.keys():
        curr_total = sum(bigram_dict[key].values())
        #print(curr_total)
        #bigram_dict[key]["UNK"] = k # deal with unseen bigrams
        # compute the conditional prob.
        for word in bigram_dict[key].keys():
            #print(bigram_dict[key][word])
            bigram_dict[key][word] = bigram_dict[key][word]/curr_total
    return bigram_dict

def get_word_tag_prob(df, tag_col = "ner", k = 0.01):
    word_tag_dict = dict()
    total_count = dict()
    for doc_idx in range(len(df)):
        curr_tags = df.loc[doc_idx, tag_col]
        curr_words = df.loc[doc_idx, "tokens"]
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
    print("finished calculating dictionary for P(word|tag)")
    return word_tag_dict, total_count

def viterbi_hmm(test_seq, word_tag_prob, tag_counts, tag_bigram):
    # word_tag_prob, tag_counts = get_word_tag_prob(training_df)
    tags = list(tag_counts.keys())
    #print("tags are ", tags)
    n = len(test_seq)
    scores = np.zeros([len(tags), n]) # dp table
    backpointers = np.zeros([len(tags), n])
    #print("tag counts are ")
    #print(tag_counts)
    for i in range(len(tags)): # initialization
        tag_prob = tag_counts[tags[i]] / sum(tag_counts.values())
        #print("current tag prob", tag_prob)
        if test_seq[0] in word_tag_prob[tags[i]]:
            scores[i][0] = tag_prob * word_tag_prob[tags[i]][test_seq[0]]/ sum(word_tag_prob[tags[i]].values())
            #print("seen score")
            #print(scores[i][0])
        else:
            scores[i][0] = tag_prob * 0.000000001
            #print("unknown score")
            #print(scores[i][0])
        backpointers[i][0] = 0
    for word_idx in range(1, n): # dp step
        for tag_idx in range(len(tags)):
            tmp_max = 0
            max_idx = -1
            for prev_tag in range(len(tags)):
                if (tags[tag_idx] in tag_bigram[tags[prev_tag]].keys()):
                    transition = tag_bigram[tags[prev_tag]][tags[tag_idx]]
                else: # deal with unseen tag bigrams
                    transition = 0.000000001
                if test_seq[word_idx] in word_tag_prob[tags[tag_idx]]:
                    lexical = word_tag_prob[tags[tag_idx]][test_seq[word_idx]] / sum(word_tag_prob[tags[tag_idx]].values())
                else: # FIXME: distinguish between unseen pairs and unknown words???
                    lexical = 0.000000001
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

# take the file name and return a df with raw and unknown tokens and pos tags
def read_test(test_filename):
    print("reading test file...")
    raw_df = read_data(test_filename)
    #unknown_cols = ["tokens", "pos"]
    #res_df = tokensWithUnk(raw_df, unknown_cols)
    return raw_df

"""
returns a dictionary that can be used to generate the submission file
"""
def predict_test(model, training_df, test_filename):
    test = read_test(test_filename)
    ner_bigram = get_bigram(training_df, "ner")
    word_tag_prob, tag_counts = get_word_tag_prob(training_df)
    indices = []
    preds = []
    print("getting predictions......")
    for i in range(len(test)):
        word_tokens = test.loc[i, "tokens"]
        pos_tokens = test.loc[i, "pos"]
        indices += test.loc[i, "ner"][0].split(" ")
        if (model == "hmm"):
            curr_preds = viterbi_hmm(word_tokens, word_tag_prob, tag_counts, ner_bigram)
        else: #TODO: fill in other options for modeling
            curr_preds = viterbi_memm(word_tokens, pos_tokens)
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
    file = open(filename, "w")
    tags = ["ORG", "MISC", "PER", "LOC"]
    file.write("Type,Prediction\n")
    for i in range(len(tags)):
        tmp = " ".join(submission[tags[i]])
        file.write(tags[i] + "," + tmp + "\n")
    file.close()

def get_memm_train(df):
    trainX = list()
    for doc_idx in range(len(df)):
        curr_poss = df.loc[doc_idx, "pos"]
        curr_words = df.loc[doc_idx, "tokens"]
        curr_ner = df.loc[doc_idx, "ner"]
        assert len(curr_poss) == len(curr_words)
        for i in range(len(curr_poss)):
            features = dict()
            features['curr_pos'] = curr_poss[i]
            features['curr_word'] = curr_words[i]
            ner = curr_ner[i]
            trainX.append((features, ner))
    return trainX

def get_memm_features(word, pos, ner):
    features = dict()
    features['curr_word'] = word
    features['curr_pos'] = pos
    features['prev_ner'] = ner
    return features

def viterbi_memm(word_seq, pos_seq):
    assert len(word_seq) == len(pos_seq)
    tags = ['O', 'ORG', 'PER', 'LOC', 'MISC']
    n = len(word_seq)
    scores = np.zeros([len(tags), n]) # dp table
    backpointers = np.zeros([len(tags), n])
    w1 = word_seq[0]
    p1 = pos_seq[0]
    for i in range(len(tags)): # initialization
        probability = maxent_classifier.prob_classify(get_memm_features(w1,p1, "init" ))
        posterior = float(probability.prob(tags[i]))
        scores[i][0] = posterior
        backpointers[i][0] = 0
    for word_idx in range(1, n): # dp step
        for tag_idx in range(len(tags)):
            tmp_max = 0
            max_idx = -1
            for prev_tag in range(len(tags)):
                probability = maxent_classifier.prob_classify(get_memm_features(word_seq[word_idx], pos_seq[word_idx], tags[prev_tag]))
                posterior = float(probability.prob(tags[tag_idx]))
                curr = scores[prev_tag][word_idx-1] * posterior
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
        tag_preds[n-1] = tags[int(max_tag_idx)]
        max_tag_idx = backpointers[int(max_tag_idx)][n-1]
        n -= 1
    return tag_preds

if __name__ == "__main__":
    raw = read_data("train.txt")
    print("raw training ", raw.columns)

    raw_withoutBIO = strip_bio(raw, "ner") # ner, pos, tokens, short_ner
    print("after removing BIOs ", raw_withoutBIO.columns)

    ner_bigrams = get_bigram(raw_withoutBIO, "ner")

    word_tag, tag_counts = get_word_tag_prob(raw_withoutBIO, "ner")

    test_tokens = "South	African	fast	bowler	Shaun	Pollock	concluded	his	Warwickshire	career	with	a	flourish".split("\t")
    print("now predict with viterbi hmm......")
    test_res = viterbi_hmm(test_tokens, word_tag, tag_counts, ner_bigrams)
    print("the predicted NER tags are")
    print(test_res)


    #submission_dict = predict_test("hmm", raw_withoutBIO, "test.txt")
    #get_submission(submission_dict, "submission.txt")
    #print("current submission file is completed, saved in submission.txt")

    ###################################
    #           MEMM                  #
    ###################################

    ### train maxent classifier     ###
    #f = open("my_classifier.pickle", "wb")
    #trainX = get_memm_train(raw_withoutBIO)
    #maxent_classifier = MaxentClassifier.train(trainX, max_iter=10)
    #pickle.dump(maxent_classifier , f)
    #f.close()
    #print("classifier saved in picklefile")

    ### load saved classifier from pickle
    f = open('my_classifier.pickle', 'rb')
    maxent_classifier = pickle.load(f)
    f.close()
    print("saved classifier has been loaded")

    word_tokens = "South	African	fast	bowler	Shaun	Pollock	concluded	his	Warwickshire	career	with	a	flourish	on	Sunday	by	taking	the	final	three	wickets	during	the	county	's	Sunday	league	victory	over	Worcestershire	.".split("\t")
    pos_tokens = "JJ	JJ	JJ	NN	NNP	NNP	VBD	PRP$	NNP	NN	IN	DT	VB	IN	NNP	IN	VBG	DT	JJ	CD	NNS	IN	DT	NN	POS	NNP	NN	NN	IN	NNP	.".split("\t")
    print("now predict with viterbi hmm......")
    test_res = viterbi_memm(word_tokens, pos_tokens)
    print("the predicted NER tags are")
    print(test_res)

    submission_dict = predict_test("memm", raw_withoutBIO, "test.txt")
    get_submission(submission_dict, "submission_memm.txt")
    print("current submission file is completed, saved in submission_memm.txt")