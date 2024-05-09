#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Zhishuai Lee
@time: 2019/10/23 10:56
@file: wordidmap.py
@desc:
"""
import pandas as pd
from collections import OrderedDict
import codecs


class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0


class DataPreProcessing(object):

    def __init__(self):
        self.docs_count = 0
        self.words_count = 0
        self.docs = []
        self.word2id = OrderedDict()
        self.word2single = 0
        self.id2word = OrderedDict()
        self.land2id_map = OrderedDict()

    def cachewordidmap(self):
        self.id2word = {k: i for i, k in self.word2id.items()}
        with codecs.open("./data/word2id.dat", 'w', 'gbk') as f:
            for word, id in self.word2id.items():
                f.write(word + "\t" + str(id) + "\n")
        with codecs.open("./data/poi2id.dat", 'w', 'gbk') as f:
            for word, id in self.land2id_map.items():
                f.write(word + "\t" + str(id) + "\n")


def preprocessing(train_file, save_file="./data/convert_word.dat"):
    dpre = DataPreProcessing()
    with codecs.open(train_file, 'r', 'GBK') as f:
        docs = f.readlines()
    while "\n" in docs:
        del docs[docs.index("\n")]
    items_idx = 0
    with codecs.open(save_file, 'w', 'gbk') as f: # 数字代替字符
        for line in docs:
            if line != "":
                tmp = line.strip().split("\t\t")
                for group in tmp:
                    if group.strip().split(",")[-1] in dpre.land2id_map:
                        pass
                    else:
                        dpre.land2id_map[group.strip().split(",")[-1]] = items_idx
                        items_idx += 1
                    # if group.strip().split(",")[-3] in dpre.land2id_map:
                    #     pass
                    # else:
                    #     dpre.land2id_map[group.strip().split(",")[-3]] = items_idx
                    #     items_idx += 1
                    idx = 0
                    for item in group.strip().split(","):
                        idx += 1
                        if len(item) > 3 and list(item)[-3] == "t":
                            f.write(item + ",")
                        elif len(item) > 3 and list(item)[-3] == "s":
                            f.write(item + ",")
                        elif idx == 2:
                            f.write(item.split(".")[0] + ",")
                        else:
                            f.write(str(dpre.land2id_map[item]))
                    f.write("\t\t")
                f.write("\n")

    with codecs.open(save_file, 'r', 'gbk') as f:
        docs = f.readlines()
    items_idx = 0
    for line in docs:
        if line != "":
            tmp = line.strip().split("\t\t")
            doc = Document()
            for item in tmp:
                if item in dpre.word2id:
                    doc.words.append(dpre.word2id[item])
                else:
                    dpre.word2id[item] = items_idx
                    doc.words.append(items_idx)
                    items_idx += 1
            doc.length = len(tmp)
            dpre.docs.append(doc)
        else:
            pass
    dpre.docs_count = len(dpre.docs)
    dpre.words_count = len(dpre.word2id)
    dpre.cachewordidmap() # 每个出行对应的单词编码
    return dpre


if __name__ == '__main__':
    preprocessing("./data/trainfile_no_homebase.dat")
