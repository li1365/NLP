cimport numpy as np
import pandas as pd
from nltk import ngrams
from nltk.classify import MaxentClassifier
import pickle

from model import get_bigram

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.ensemble import VotingClassifier
# from sklearn import preprocessing

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
    if model == 'svm':
        with open("classifier_svm.pickle", 'rb') as f:  
            classifier = pickle.loads(f.read())
    elif model == 'glm':
        with open("classifier_glm.pickle", 'rb') as f:  
            classifier = pickle.loads(f.read())
    else:
        print("Model Loading Error")
        exit(1)
    test = read_test(test_filename)
    indices = []
    preds = []
    print("getting predictions......")
    for i in range(len(test)):
        word_tokens = test.loc[i, "tokens"]
        pos_tokens = test.loc[i, "pos"]
        indices += test.loc[i, "ner"][0].split(" ")
        curr_preds = viterbi_memm(word_tokens, pos_tokens, classifier)
        for i in range(len(curr_preds)):
            if "-" in curr_preds[i]:
                tmp = curr_preds[i].index("-")
                curr_preds[i] = curr_preds[i][(tmp+1):len(curr_preds[i])]
        preds += curr_preds
    # reformat the preds for submission
    print("predicting finished")
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

def viterbi_memm(word_seq, pos_seq, classifier):
    assert len(word_seq) == len(pos_seq)
    tags = ['LOC', 'MISC', 'O', 'ORG', 'PER']
    n = len(word_seq)
    scores = np.zeros([len(tags), n]) # dp table
    backpointers = np.zeros([len(tags), n])
    for i in range(len(tags)): # initialization
        init_features = get_memm_features(word_seq, pos_seq, 0, 'init')
        predictions = classifier.predict_proba(init_features)
        posterior = float(predictions[0][i])
        scores[i][0] = posterior
        backpointers[i][0] = 0
    for word_idx in range(1, n): # dp step
        for tag_idx in range(len(tags)):
            tmp_max = 0
            max_idx = -1
            for prev_tag in range(len(tags)):
                curr_features = get_memm_features(word_seq, pos_seq, word_idx, tags[prev_tag])
                predictions = classifier.predict_proba(curr_features)
                posterior = float(predictions[0][tag_idx])
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


# take the submission dictionary and convert it to a txt file with the inputted filename
def get_submission(submission, filename):
    print("generating submission file....")
    file = open(filename, "w", encoding='utf8')
    tags = ["ORG", "MISC", "PER", "LOC"]
    file.write("Type,Prediction\n")
    for i in range(len(tags)):
        tmp = " ".join(submission[tags[i]])
        print(tmp)
        print(tags[i])
        file.write(tags[i] + "," + tmp + "\n")
    file.close()

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    print("Model training")
    classifier.fit(feature_vector_train, label)
    f = open("classifier_glm.pickle", "wb")
    pickle.dump(classifier , f)
    f.close()

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    print(predictions.shape)
    print(predictions)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)

def dummy(tokens):
    return tokens

def all_tokens(df):
    tokens = []
    for i in range(len(df)):
        for token in df[i]:
            tokens.append(token)
    return tokens

def concat_rows(df):
    cols = list(df.columns)
    res = dict()
    for col in cols:
        res[col] = []
    for i in range(len(df)):
        for col in cols:
            res[col] += df.loc[i, col]
    return pd.DataFrame(res)

def pos(tag):
    one_hot = np.zeros(7)
    if tag == 'NN' or tag == 'NNS':
        one_hot[0] = 1
    elif tag == 'FW':
        one_hot[1] = 1
    elif tag == 'NNP' or tag == 'NNPS':
        one_hot[2] = 1
    elif 'VB' in tag:
        one_hot[3] = 1
    elif tag == "init":
        one_hot[4] = 1
    elif tag == "last":
        one_hot[5] = 1
    else:
        one_hot[6] = 1
    return one_hot

def ner(tag):
    one_hot = np.zeros(6)
    if tag == 'PER':
        one_hot[0] = 1
    elif tag == 'LOC':
        one_hot[1] = 1
    elif tag == "MISC":
        one_hot[2] = 1
    elif tag == "ORG":
        one_hot[3] = 1
    elif tag == "init":
        one_hot[4] = 1
    else:
        one_hot[5] = 1
    return one_hot

def get_memm_features(curr_words, curr_poss, word_idx, prev_ner):
    assert len(curr_poss) == len(curr_words)
    trainX = []
    features = dict()
    features["position"] = word_idx
    # regarding previous word
    if word_idx == 0:
        features['curr_pos'] = pos(curr_poss[word_idx])
        features['curr_word'] = embedding[word_dict[curr_words[word_idx].lower()]]
        features['prev_word'] = np.zeros(300)
        features['prev_pos'] = pos("init")
        features['prev_ner'] = ner("init")
    else:
        features['curr_pos'] = pos(curr_poss[word_idx])
        features['prev_pos'] = pos(curr_poss[word_idx-1])
        features['prev_ner'] = ner(prev_ner)
        features['curr_word'] = embedding[word_dict[curr_words[word_idx].lower()]]
        features['prev_word'] = embedding[word_dict[curr_words[word_idx-1].lower()]]
    # add in n-grams
    if word_idx == 0 or (curr_words[word_idx-1] not in word_bigram.keys()) or (curr_words[word_idx] not in word_bigram[curr_words[word_idx-1]].keys()):
        features["bigram"] = 0.000001
    else:
        features["bigram"] = word_bigram[curr_words[word_idx-1]][curr_words[word_idx]]
    features["capital"] = 1 if curr_words[word_idx][0].isupper() else -1
    # regarding next word
    if word_idx == len(curr_words) -1:
        features["next_word"] = np.ones(300)
        features["next_pos"] = pos("last")
    else:
        features["next_word"] = embedding[word_dict[curr_words[word_idx+1].lower()]]
        features['next_pos'] = pos(curr_poss[word_idx+1])
    tmp_words = np.hstack((features["prev_word"], features["curr_word"], features["next_word"]))
    tmp_words = np.hstack((tmp_words, features["curr_pos"], features["prev_pos"], features["next_pos"]))
    tmp_words = np.hstack((tmp_words, features["prev_ner"], features["bigram"], features["capital"]))
    trainX.append(tmp_words)
    return np.array(trainX)

def get_memm_train(df, bigrams, embedding, word_dict):
    print("generating training features")
    word_bigram = bigrams
    trainX = []
    trainY = []
    for doc_idx in range(len(df)):
        curr_poss = df.loc[doc_idx, "pos"]
        curr_words = df.loc[doc_idx, "tokens"]
        curr_ner = df.loc[doc_idx, "ner"]
        assert len(curr_poss) == len(curr_words)
        for i in range(len(curr_poss)):
            features = dict()
            features["position"] = i
            # regarding previous word
            if i == 0:
                features['curr_pos'] = pos(curr_poss[i])
                features['curr_word'] = embedding[word_dict[curr_words[i].lower()]]
                features['prev_word'] = np.zeros(300)
                features['prev_pos'] = pos("init")
                features['prev_ner'] = ner("init")
            else:
                features['curr_pos'] = pos(curr_poss[i])
                features['prev_pos'] = pos(curr_poss[i-1])
                features['prev_ner'] = ner(curr_ner[i-1])
                features['curr_word'] = embedding[word_dict[curr_words[i].lower()]]
                features['prev_word'] = embedding[word_dict[curr_words[i-1].lower()]]
            # add in n-grams
            if i == 0 or (curr_words[i-1] not in word_bigram.keys()) or (curr_words[i] not in word_bigram[curr_words[i-1]].keys()):
                features["bigram"] = 0.000001
            else:
                features["bigram"] = word_bigram[curr_words[i-1]][curr_words[i]]
            features["capital"] = 1 if curr_words[i][0].isupper() else -1
            # regarding next word
            if i == len(curr_words) -1:
                features["next_word"] = np.ones(300)
                features["next_pos"] = pos("last")
            else:
                features["next_word"] = embedding[word_dict[curr_words[i+1].lower()]]
                features['next_pos'] = pos(curr_poss[i+1])
            tmp_words = np.hstack((features["prev_word"], features["curr_word"], features["next_word"]))
            tmp_words = np.hstack((tmp_words, features["curr_pos"], features["prev_pos"], features["next_pos"]))
            tmp_words = np.hstack((tmp_words, features["prev_ner"], features["bigram"], features["capital"]))
            trainX.append(tmp_words)
            trainY.append(curr_ner[i])
    print(np.array(trainX).shape)
    print(np.array(trainY).shape)
    return np.array(trainX), np.array(trainY)

if __name__ == "__main__":
    raw = read_data("train.txt")
    raw_withoutBIO = strip_bio(raw, "ner") # ner, pos, tokens, short_ner
    raw_withoutBIO = raw_withoutBIO.drop(columns = ["short_ner"])
    extra_data = pd.read_pickle("conll2003_combined.pkl")
    raw_withoutBIO = pd.concat([raw_withoutBIO, extra_data], ignore_index=True)

    ### load saved classifier from pickle
    with open("embedding_matrix.pickle", 'rb') as f:  
        embedding = pickle.loads(f.read())
    with open("word_index.pickle", 'rb') as f:  
        word_dict = pickle.loads(f.read())

    word_bigram = get_bigram(raw_withoutBIO, "tokens")
    trainX, trainY = get_memm_train(raw_withoutBIO, word_bigram, embedding, word_dict)
    # split the dataset into training and validation datasets
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainX, trainY)
    #print(train_y[:20])
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)
    #print(train_x[:10])
    #print(train_y[:20])

    # SVM on Ngram Level TF IDF Vectors
    #accuracy = train_model(svm.SVC(), train_x, train_y, valid_x)
    #print("SVM: ", accuracy)

    # Linear Classifier on Character Level TF IDF Vectors
    #accuracy = train_model(linear_model.LogisticRegression(), train_x, train_y, valid_x)
    #print("LR, CharLevel Vectors: ", accuracy)

    submission_dict = predict_test("glm", raw_withoutBIO, "test.txt")
    get_submission(submission_dict, "submission_glm.txt")
    print("current submission file is completed, saved in submission_glm.txt")