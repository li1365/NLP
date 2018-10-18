import numpy as np
import pandas as pd
from nltk import ngrams
from nltk.classify import MaxentClassifier
import pickle

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.ensemble import VotingClassifier

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


if __name__ == "__main__":
    print("reading training file....")
    raw = read_data("train.txt")
    raw_withoutBIO = strip_bio(raw, "ner") # ner, pos, tokens, short_ner
    raw_df = concat_rows(raw_withoutBIO)
    raw_df = raw_df.drop('short_ner', axis=1)


    words = all_tokens(raw_withoutBIO['tokens'])
    print(len(words))
    poss = all_tokens(raw_withoutBIO['pos'])
    ners = all_tokens(raw_withoutBIO['ner'])
    X = []
    for i in range(len(words)):
        X.append([words[i], poss[i]])

    # split the dataset into training and validation datasets
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(words, ners)
    print(train_x[:10])
    print(train_y[:10])
    # label encode the target variable
    print(valid_y[:20])
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)
    print(valid_y[:20])
    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', tokenizer=dummy, preprocessor=dummy, token_pattern=None)
    count_vect.fit(words)

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)

    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', tokenizer=dummy, preprocessor=dummy, token_pattern=None, max_features=5000)
    tfidf_vect.fit(words)
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)

    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', tokenizer=dummy, preprocessor=dummy, token_pattern=None, ngram_range=(1,3), max_features=5000)
    tfidf_vect_ngram.fit(words)
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='word', tokenizer=dummy, preprocessor=dummy, token_pattern=None, ngram_range=(1,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(words)
    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)
    xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)

    # # # Naive Bayes on Count Vectors
    # accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
    # print("NB, Count Vectors: ", accuracy)
    #
    # # Naive Bayes on Word Level TF IDF Vectors
    # accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
    # print("NB, WordLevel TF-IDF: ", accuracy)
    #
    # # Naive Bayes on Ngram Level TF IDF Vectors
    # accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    # print("NB, N-Gram Vectors: ", accuracy)
    #
    # # Linear Classifier on Count Vectors
    # accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
    # print("LR, Count Vectors: ", accuracy)
    #
    # # Linear Classifier on Word Level TF IDF Vectors
    # accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
    # print("LR, WordLevel TF-IDF: ", accuracy)
    #
    # # Linear Classifier on Ngram Level TF IDF Vectors
    # accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    # print("LR, N-Gram Vectors: ", accuracy)
    #
    # SVM on Ngram Level TF IDF Vectors
    accuracy = train_model(svm.LinearSVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    print("SVM, N-Gram Vectors: ", accuracy)
    #
    # # SVM on Count Vectors
    # accuracy = train_model(svm.LinearSVC(), xtrain_count, train_y, xvalid_count)
    # print("SVM, Count Vectors: ", accuracy)

    # Naive Bayes on Character Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    print("NB, CharLevel Vectors: ", accuracy)

    # Linear Classifier on Character Level TF IDF Vectors
    accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    print("LR, CharLevel Vectors: ", accuracy)

    submission_dict = predict_test("svm", raw_withoutBIO, "test.txt")
    get_submission(submission_dict, "submission_svm.txt")
    print("current submission file is completed, saved in submission_svm.txt")