import numpy as np 
import pandas as pd 
import math, scipy

#take the txt file, return dictionary with three keys: tokens, POS tags and NER tags
def read_data(filename):
    lines = open(filename, "r").read().split("\n")
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
    res = {"tokens": tokens, "pos": pos, "ner": ner}
    # return tokens, pos, ner 
    return res

# TODO: get the bigram dictionary for tokens
# copy pasted from P1
# MAY NEED REVISING DUE TO CURRENT STRUCTURE OF TOKENS
def get_bigram(tokens):
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

#TODO: add-k smoothing, return smoothed dictionary
def add_k(bigram_dict, k = 0.01):
    return bigram_dict

#TODO: split data into 80% training and 20% validation
def split_training(tokens, pos, ner):
    ratio = 0.8
    indices = []
    
    return ratio



raw = read_data("train.txt")
print(raw["tokens"])