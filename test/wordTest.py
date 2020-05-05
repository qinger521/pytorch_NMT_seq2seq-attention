# from collections import Counter
# import numpy as np
# import random
# import matplotlib.pyplot as plt
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# #英文分词
# import nltk
#
# '''
#     数据预处理
#     英文我们使用nltk的word tokenizer来分词，并且使用小写字母
#     中文我们直接使用单个汉字作为基本单元
# '''
# def load_data(in_file):
#     cn = []
#     en = []
#     num_examples = 0
#     with open(in_file,'r') as f:
#         for line in f :
#             line = line.strip().split("\t")
#             en.append(["BOS"] + nltk.word_tokenize(line[0].lower()) + ["EOS"])
#             cn.append(["BOS"] + [c for c in line[1]] + ["EOS"])
#     return en , cn
#
# train_file = "../train.txt"
# dev_file = "../dev.txt"
# train_en,train_cn = load_data(train_file)
# dev_en,dev_cn = load_data(dev_file)
#
#
# def load_yuliao_data(in_file):
#     i = 1
#     with open(in_file,'r') as f:
#         for line in f :
#             if i % 2 == 0:
#                 train_en.append(["BOS"] + nltk.word_tokenize(line.lower()) + ["EOS"])
#             else:
#                 train_cn.append(["BOS"] + [c for c in line] + ["EOS"])
#             i = i+1
#
#
# load_yuliao_data("../txt/yuliao-utf-8.txt")
# print(len(train_en))
# print(len(train_cn))
# print(train_en[-10:-1])
# print(train_cn[-10:-1])
#
# '''
#     构建单词表
# '''
# UNK_IDX = 0
# PAD_IDX = 1
# def build_dict(sentences, max_words=500000):
#     word_count = Counter()
#     for sentence in sentences:
#         for s in sentence:
#             word_count[s] += 1
#     ls = word_count.most_common(max_words)
#     total_words = len(ls) + 2
#     word_dict = {w[0]: index+2 for index, w in enumerate(ls)}
#     word_dict["UNK"] = UNK_IDX
#     word_dict["PAD"] = PAD_IDX
#     return word_dict, total_words
#
# en_dict, en_total_words = build_dict(train_en)
# cn_dict, cn_total_words = build_dict(train_cn)
# # 对编码的句子根据单词表进行解码
# inv_en_dict = {v: k for k, v in en_dict.items()}
# inv_cn_dict = {v: k for k, v in cn_dict.items()}
#
# '''
#     对单词进行编码:通过词典对句子进行编号
# '''
# def encode(en_sentences, cn_sentences, en_dict, cn_dict, sort_by_len=True):
#     '''
#         Encode the sequences.
#     '''
#     length = len(en_sentences)
#     out_en_sentences = [[en_dict.get(w, 0) for w in sent] for sent in en_sentences]
#     out_cn_sentences = [[cn_dict.get(w, 0) for w in sent] for sent in cn_sentences]
#
#     # sort sentences by english lengths
#     def len_argsort(seq):
#         return sorted(range(len(seq)), key=lambda x: len(seq[x]))
#
#     # 把中文和英文按照同样的顺序排序
#     if sort_by_len:
#         sorted_index = len_argsort(out_en_sentences)
#         out_en_sentences = [out_en_sentences[i] for i in sorted_index]
#         out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]
#
#     return out_en_sentences, out_cn_sentences
#
#
# train_en, train_cn = encode(train_en, train_cn, en_dict, cn_dict)
# dev_en, dev_cn = encode(dev_en, dev_cn, en_dict, cn_dict)
#
#
# '''
#     把全部句子分成batch
# '''
# def get_minibatches(n, minibatch_size, shuffle=True):
#     idx_list = np.arange(0, n, minibatch_size) # [0, 1, ..., n-1]
#     if shuffle:
#         np.random.shuffle(idx_list)
#     minibatches = []
#     for idx in idx_list:
#         minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
#     return minibatches
#
# def prepare_data(seqs):
#     lengths = [len(seq) for seq in seqs]
#     n_samples = len(seqs)
#     max_len = np.max(lengths)
#
#     x = np.zeros((n_samples, max_len)).astype('int32')
#     x_lengths = np.array(lengths).astype("int32")
#     for idx, seq in enumerate(seqs):
#         x[idx, :lengths[idx]] = seq
#     return x, x_lengths #x_mask
#
# def gen_examples(en_sentences, cn_sentences, batch_size):
#     minibatches = get_minibatches(len(en_sentences), batch_size)
#     all_ex = []
#     for minibatch in minibatches:
#         mb_en_sentences = [en_sentences[t] for t in minibatch]
#         mb_cn_sentences = [cn_sentences[t] for t in minibatch]
#         mb_x, mb_x_len = prepare_data(mb_en_sentences)
#         mb_y, mb_y_len = prepare_data(mb_cn_sentences)
#         all_ex.append((mb_x, mb_x_len, mb_y, mb_y_len))
#     return all_ex
#
# batch_size = 128
# train_data = gen_examples(train_en, train_cn, batch_size)
# random.shuffle(train_data)
# dev_data = gen_examples(dev_en, dev_cn, batch_size)
# print(len(train_data))
# print(len(train_data[:1]))
import torch
import numpy
mb_y_len = [5,4,3]
mb_y_len = numpy.array(mb_y_len)
mb_y_len = torch.from_numpy(mb_y_len).long()
mb_out_mask = torch.arange(mb_y_len.max().item())[None, :] < mb_y_len[:, None]
print(mb_out_mask.numpy())
