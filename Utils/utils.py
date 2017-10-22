import pandas as pd
import numpy as np
from dateutil.parser import parse
import os.path
import codecs
from gensim.models.keyedvectors import KeyedVectors

# create dictionary from list
# with key = <list_value> and value = <frequency>
def dict_from_list(lst):
    assert type(lst) is list
    dict = {}

    for elems in lst:
        for elem in elems:
            if elem in dict:
                dict[elem] += 1
            else:
                dict[elem] = 1

    return dict

# create a value-to-id and id-to-value from dictionary
def mapping(dico):
    assert type(dico) is dict

    # sort item by descending its value (frequency of word)
    sorted_items = sorted(dico.items(),key=lambda elem: -elem[1])
    id2item = {i:v[0] for i,v in enumerate(sorted_items)}
    item2id = {v[0]:i for i,v in enumerate(sorted_items)}

    return id2item,item2id

# mapping from list of sentences to dictionary
# with key = <word> and value = <frequency>
# example of input: [['hello','world'],...]
def word_mapping(lst_sentence,pre_emb=''):
    assert type(lst_sentence) is list

    # get dict of words
    dict_word = dict_from_list(lst_sentence)
    dict_word['<unk>'] = 10000

    if pre_emb != '':
        word2vec_model = KeyedVectors.load_word2vec_format(pre_emb,binary=True)
        vocabs = word2vec_model.wv.vocab.keys()
        for word in vocabs:
            if word not in dict_word:
                dict_word[word] = 0


    id2word,word2id = mapping(dict_word)

    print "Found %i unique words (%i in total)" % (
        len(dict_word), sum(len(x) for x in lst_sentence)
    )

    return dict_word,id2word,word2id

# similar to word_mapping
# but now use character representation instead
def char_mapping(lst_sentence):
    assert type(lst_sentence) is list
    lst_sentence_by_char = [''.join(s) for s in lst_sentence]

    # get dict of characters
    dict_char = dict_from_list(lst_sentence_by_char)
    dict_char['<unk>'] = 10000

    id2char,char2id = mapping(dict_char)

    print "Found %i unique characters" % len(dict_char)

    return dict_char,id2char,char2id

# similar to word_mapping,char_mapping
# but now for other type instead
def common_mapping(lst_x,name='x'):
    assert type(lst_x) is list

    dict_x = dict_from_list(lst_x)
    id2x,x2id = mapping(dict_x)

    print "Found %i unique %s" % (len(dict_x),name)

    return dict_x,id2x,x2id

"""
Pad the characters of the words in a sentence.
Input:
    - list of lists of ints (list of words, a word being a list of char indexes)
Output:
    - padded list of lists of ints
    - padded list of lists of ints (where chars are reversed)
    - list of ints corresponding to the index of the last character of each word
"""
def pad_word_chars(words,max_word_len = -1):
    if max_word_len > 0:
        max_length = max_word_len
    else:
        max_length = max([len(word) for word in words])

    char_for = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
    return char_for


