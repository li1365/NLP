import pandas as pd
from model import strip_bio

def read_txt(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    tokens = []
    pos = []
    ner = []
    i = 1
    curr_word = []
    curr_pos = []
    curr_ner = []
    unique_ner = []
    while i < len(lines):
        curr = lines[i].split(" ")
        if len(curr) > 1:
            curr_word.append(curr[0])
            curr_pos.append(curr[1])
            curr_ner.append(curr[-1])
        else:
            tokens.append(curr_word)
            pos.append(curr_pos)
            ner.append(curr_ner)
            curr_ner = []
            curr_pos = []
            curr_word = []
        i+= 1
    return pd.DataFrame({"tokens": tokens, "pos": pos, "ner": ner})
    
train = read_txt("train_conll2003.txt")
validate = read_txt("valid_conll2003.txt")
test = read_txt("test_conll2003.txt")
print(len(train), len(validate), len(test))
combined = pd.concat([train, validate, test], ignore_index=True)
print(len(combined))

combined = strip_bio(combined, "ner")
combined = combined.drop(columns = ["short_ner"])
combined.to_pickle("conll2003_combined.pkl")
# validate = strip_bio(validate, "ner")
# train = strip_bio(train, "ner")

# print(a,b,c)
# print(validate.head())
# print("=========================")
# print(test.head())