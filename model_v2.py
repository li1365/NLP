import numpy as np
import pandas as pd
from nltk import ngrams
from nltk.classify import MaxentClassifier
import pickle
import category_encoders as ce

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
    test = read_test(test_filename)
    xtest = all_tokens(test['tokens'])
    xtest_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(xtest)
    indices = []
    preds = testClassify(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xtest_tfidf_ngram_chars)
    # reformat the preds for submission
    #assert len(indices) == len(preds)
    indices = preds
    invert_index = {0:'LOC', 1:'MISC', 2:'O', 3:'ORG', 4:'PER'}
    res = []
    for element in preds:
        res.append(invert_index[element])
    print(res[:10])
    submission = {"LOC":[], "PER":[], "ORG":[], "MISC":[]}
    i = 0
    while i < len(indices):
        curr_pred = res[i]
        if curr_pred in submission.keys(): # NER tag is not O
            start = i
            while (i+1) < len(indices) and res[i] == res[i+1]:
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

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)

def testClassify(classifier, feature_vector_train, label, feature_vector_test, is_neural_net=False):
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_test)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return predictions

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

def get_memm_train(df, bigrams, embedding, word_dict):
    print("generating training features for MaxEntrophy Classifier")
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
    print(train_x[:10])
    print(train_y[:10])

    # SVM on Ngram Level TF IDF Vectors
    accuracy = train_model(svm.LinearSVC(), train_x, train_y, valid_x)
    print("SVM: ", accuracy)

    # Naive Bayes on Character Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), train_x, train_y, valid_x)
    print("NB, CharLevel Vectors: ", accuracy)

    # Linear Classifier on Character Level TF IDF Vectors
    accuracy = train_model(linear_model.LogisticRegression(), train_x, train_y, valid_x)
    print("LR, CharLevel Vectors: ", accuracy)

    # submission_dict = predict_test("svm", raw_withoutBIO, "test.txt")
    # get_submission(submission_dict, "submission_svm.txt")
    # print("current submission file is completed, saved in submission_svm.txt")