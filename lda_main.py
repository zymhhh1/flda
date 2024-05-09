#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
import codecs
import os
import math
import seaborn as sns
from data import word_idmap
import matplotlib.pyplot as plt
from collections import OrderedDict
import configparser
from matplotlib.ticker import ScalarFormatter


conf = configparser.ConfigParser()
conf.read("./setting.conf")
num_taz = int(conf.get("model_args", "num_taz"))

path = os.getcwd()
train_file = os.path.join(path, os.path.normpath(conf.get("filepath", "trainfile")))
wordidmapfile = os.path.join(path, os.path.normpath(conf.get("filepath", "wordidmapfile")))
thetafile = os.path.join(path, os.path.normpath(conf.get("filepath", "thetafile")))
phifile = os.path.join(path, os.path.normpath(conf.get("filepath", "phifile")))
mufile = os.path.join(path, os.path.normpath(conf.get("filepath", "mufile")))
yitafile = os.path.join(path, os.path.normpath(conf.get("filepath", "yitafile")))
paifile = os.path.join(path, os.path.normpath(conf.get("filepath", "paifile")))
paramfile = os.path.join(path, os.path.normpath(conf.get("filepath", "paramfile")))
topNfile = os.path.join(path, os.path.normpath(conf.get("filepath", "topNfile")))
tassginfile = os.path.join(path, os.path.normpath(conf.get("filepath", "tassginfile")))

K = int(conf.get("model_args", "K"))  # 主题数
# 超参数
beta = float(conf.get("model_args", "beta"))
gamma = float(conf.get("model_args", "gamma"))
tao = float(conf.get("model_args", "tao"))
lamb = float(conf.get("model_args", "lamb"))

iter_times = int(conf.get("model_args", "iter_times"))
top_words_num = int(conf.get("model_args", "top_words_num"))
age_num = 3

cont = 2
cont1 = 1


class LDAModel(object):
    def __init__(self, dpre):
        self.dpre = dpre
        self.K = K
        self.pre_min = 0
        self.beta = beta
        self.alpha = 50.0 / K
        self.lamb = lamb
        self.gamma = gamma
        self.tao = tao
        self.tao0 = 1 / 64
        self.lamb0 = 1 / 64
        self.yita0 = 3.2
        self.mu0 = 13.9
        self.iter_times = iter_times
        self.top_words_num = top_words_num
        self.wordidmapfile = wordidmapfile
        self.thetafile = thetafile
        self.phifile = phifile
        self.mufile = mufile
        self.yitafile = yitafile
        self.topNfile = topNfile
        self.tassginfile = tassginfile
        self.paramfile = paramfile
        self.paifile = paifile
        self.p = np.zeros(self.K)
        # 单词-主题矩阵
        self.nw = np.zeros((self.dpre.words_count, self.K), dtype="int")
        self.nwsum = np.zeros(self.K, dtype="int")
        # 文档-主题矩阵
        self.nd = np.zeros((self.dpre.docs_count, self.K), dtype="int")
        self.ndsum = np.zeros(dpre.docs_count, dtype="int")
        self.Z = np.array([[0 for y in range(dpre.docs[x].length)] for x in range(dpre.docs_count)])
        self.taz = np.zeros((num_taz, self.K), dtype="int")
        self.tazsum = np.zeros(self.K, dtype="int")
        self.dow = np.zeros((age_num, self.K), dtype="int")
        self.dowsum = np.zeros(self.K, dtype="int")
        self.index = dpre.id2word
        self.todsum = np.zeros(self.K, dtype="float")
        self.duration = np.zeros(self.K, dtype="float")

        self.dura_freq = np.zeros(24, dtype="int")
        self.tod_freq = np.zeros(24, dtype="int")
        self.dow_freq = np.zeros(age_num, dtype="int")
        self.taz_freq = np.zeros(num_taz, dtype="int")

        for x in range(len(self.Z)):  # x是文档数
            self.ndsum[x] = self.dpre.docs[x].length
            for y in range(self.dpre.docs[x].length):  # y是单词
                topic = random.randint(0, self.K - 1)
                self.Z[x][y] = topic  # 给文档中的每个单词随机初始化主题
                self.nw[self.dpre.docs[x].words[y]][topic] += 1  # 单词数 * 主题个数， 每个单词对应的主题分布
                self.nd[x][topic] += 1  # 文档数 * 主题数，每个文档中分配给主题z的单词数
                self.nwsum[topic] += 1  # 主题数，每个主题下的单词个数
                # 针对每个单词 (id, time of the day, day of the week, duration, destination)

                tmp_v = self.dpre.id2word[self.dpre.docs[x].words[y]].strip().split(",")
                self.todsum[topic] = self.todsum[topic] + float(tmp_v[0][:-3])
                self.dow[int(tmp_v[1])][topic] += 1  # 7 * 主题数，每天对应的主题个数
                self.dowsum = self.dow.sum(axis=0)  # 主题数，每个主题下的天数
                self.duration[topic] = self.duration[topic] + float(tmp_v[2][:-3])
                self.taz[int(tmp_v[3])][topic] += 1  # num_taz * 主题数，每个小区对应的主题个数
                self.tazsum = self.taz.sum(axis=0)  # 主题数，每个主题下的小区数
                self.dura_freq[int(np.floor(float(tmp_v[2][:-3])))] += 1
                self.tod_freq[int(np.floor(float(tmp_v[0][:-3])))] += 1
                self.dow_freq[int(tmp_v[1])] += 1
                self.taz_freq[int(tmp_v[3])] += 1
        with codecs.open("./data/freq_dis.dat", 'w', 'gbk') as f:
            for item in list(self.dow_freq):
                f.write(str(item) + "\t")
            f.write("\n")
            for item in list(self.taz_freq):
                f.write(str(item) + "\t")
            f.close()
        self.plot_statics()

        self.tod_mean = np.sum(self.tod_freq * np.arange(24)) / self.tod_freq.sum()
        self.dura_mean = np.sum(self.dura_freq * np.arange(24)) / self.dura_freq.sum()
        # 未知的分布，建立相应纬度的二维数组  zym
        self.pai = np.zeros((self.dpre.docs_count, self.K), dtype="float")
        self.theta = np.zeros((num_taz, self.K), dtype="float")
        self.phi = np.zeros((age_num, self.K), dtype="float")
        self.mu = np.zeros(self.K, dtype="float")
        self.yita = np.zeros(self.K, dtype="float")

    def plot_statics(self):
        index = np.arange(6, 24, 2)
        # width = 0.85
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        ax0 = ax[0][0]
        ax0.bar(np.arange(6, 24), self.tod_freq[6:], color="#1E76B4")
        ax0.set_xlabel('Arrival Time', fontsize=20)
        ax0.set_ylabel('Number of Trips', fontsize=20)
        ax0.set_xticks(index)
        ax0.set_xticklabels([(str(i).zfill(2) + ":00") for i in index], rotation=-45)
        xfmt = ScalarFormatter(useMathText=True)
        xfmt.set_powerlimits((0, 0))  # Or whatever your limits are . . .
        ax0.yaxis.set_major_formatter(xfmt)
        ax1 = ax[1][0]
        index = np.arange(20)
        ax1.bar(index, self.dura_freq[:20], color="#1E76B4")
        # ax1.set_yscale("log")
        # ax1.set_ylim(1e-1, 1e4)
        ax1.set_xlabel('Stay Duration (h)', fontsize=20)
        ax1.set_ylabel('Number of Trips', fontsize=20)
        xfmt = ScalarFormatter(useMathText=True)
        xfmt.set_powerlimits((0, 0))  # Or whatever your limits are . . .
        ax1.yaxis.set_major_formatter(xfmt)
        with codecs.open("./data/poi2id.dat", 'r', 'gbk') as f:
            land = f.readlines()
        dest = dict()
        ch2en = {"Home": "Home", "文化传媒": "Press Media",
                 "美食": "Catering",
                 "医疗": "Health Serv.", "住宅区": "Resid. Areas", "写字楼": "Office Bld"
            , "生活服务": "Life Serv.", "休闲娱乐": "Recreation", "政府机构": "Gov Ins.", "公司企业": "Company"
            , "购物": "Shop Serv.", "教育培训": "Edu Ins.", "酒店": "Hotel", "旅游景点": "Attractions",
                 "汽车服务": "automobile service", "金融": "Finance Ins.", "运动健身": "Sports Serv.",
                 "其它": "Else", "丽人": "Cosmetology", "Else": "Else"}
        ax2 = ax[0][1]
        for item in land:
            if item != "":
                dest[item.split("\t")[1][:-1]] = item.split("\t")[0]

        ind = np.arange(age_num)
        x_tick = ["13~24"]
        x_tick.extend(["25~60"])
        x_tick.extend(["60+"])
        ax2.bar(ind, self.dow_freq, color="#1E76B4")
        ax2.set_xlabel("Age Group (years old)", fontsize=20)
        ax2.set_ylabel('Number of Trips', fontsize=20)
        ax2.set_xticks(ind)
        ax2.set_xticklabels(x_tick, rotation=-45)
        xfmt = ScalarFormatter(useMathText=True)
        xfmt.set_powerlimits((0, 0))  # Or whatever your limits are . . .
        ax2.yaxis.set_major_formatter(xfmt)
        ax3 = ax[1][1]
        index = sorted(enumerate(self.taz_freq), key=lambda x: x[1], reverse=True)
        dict_key = [x[0] for x in index]
        x_tick = [ch2en[dest[str(i)]] for i in dict_key]
        ind = np.arange(num_taz)
        ax3.bar(ind, [x[1] for x in index], color="#1E76B4")
        print({x_tick[i]: [x[1] for x in index][i] for i in range(len(x_tick))})
        ax3.set_xticks(ind)
        ax3.set_xticklabels(x_tick, rotation=-45)
        ax3.set_xlabel("PoIs of Destination", fontsize=20)
        ax3.set_ylabel('Number of Trips', fontsize=20)
        import matplotlib.transforms as mtrans
        trans = mtrans.Affine2D().translate(20, 0)
        for t in ax3.get_xticklabels():
            t.set_transform(t.get_transform() + trans)
        # ax.set_ylim(ymax=0.4)
        plt.tight_layout()
        xfmt = ScalarFormatter(useMathText=True)
        xfmt.set_powerlimits((0, 0))  # Or whatever your limits are . . .
        ax3.yaxis.set_major_formatter(xfmt)
        plt.savefig("data_dis.pdf")

        plt.show()

        print()

    def plt_tpc(self):
        index = np.arange(self.K)
        width = 0.8
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        prpuse = np.zeros(self.K)
        for i in range(len(self.Z)):
            for j in range(len(self.Z[i])):
                prpuse[self.Z[i][j]] += 1
        adjust_top = [0, 14, 10, 9, 6, 15, 16, 17, 1, 4, 12, 13, 2, 8, 7, 5, 11, 3]
        swap_prpuse = prpuse[adjust_top]
        print(swap_prpuse)
        srt_p = sorted(enumerate(swap_prpuse), key=lambda x: x[1], reverse=True)
        x_tick = ["{}".format(x[0]) for x in srt_p]
        ax.bar(index, [ppse[1] for ppse in srt_p], width, color="#1E76B4")
        ax.set_xticks(index)
        ax.set_xlabel("Topic Index (K=18)", fontsize=16)
        ax.set_ylabel("Number of Trips", fontsize=16)
        # ax.set_title("The number of trip records assigned to topics")
        ax.set_xticklabels(x_tick)
        xfmt = ScalarFormatter(useMathText=True)
        xfmt.set_powerlimits((0, 0))  # Or whatever your limits are . . .
        ax.yaxis.set_major_formatter(xfmt)
        plt.tight_layout()
        plt.savefig("dis_home.pdf")
        plt.show()

    def perplexity(self):
        n = 0
        ll = 0.0
        for i, doc in enumerate(self.dpre.docs):
            for j in range(self.dpre.docs[i].length):
                tmp_v = self.dpre.id2word[self.dpre.docs[i].words[j]].strip().split(",")
                Valpha = self.K * self.alpha
                Kbeta = num_taz * self.beta
                Kgamma = age_num * self.gamma
                # u:nd, nz: nw, w: dow, v: taz, s: todsum, q: duration
                p = np.zeros_like(self.p)
                tod_norm = np.zeros_like(self.p)
                dow_norm = np.zeros_like(self.p)
                # u:nd, nz: nw, w: dow, v: taz, s: todsum, q: duration
                for inde in range(self.K):
                    smean = (self.tao * self.todsum[inde] + self.tao0 * self.mu0) / (
                            self.tao * self.nwsum[inde] + self.tao0)
                    scov = 1 / (self.tao * self.nwsum[inde] + self.tao0) + 1 / self.tao

                    qmean = (self.lamb * self.duration[inde] + self.lamb0 * self.yita0) / (
                            self.lamb * self.nwsum[inde] + self.lamb0)
                    qcov = 1 / (self.lamb * self.nwsum[inde] + self.lamb0) + 1 / self.lamb

                    tod_norm[inde] = self.normal(smean, scov, float(tmp_v[0][:-3]))
                    dow_norm[inde] = self.normal(qmean, qcov, float(tmp_v[2][:-3]))
                # tod_norm = tod_norm / tod_norm.sum()
                # dow_norm = dow_norm / dow_norm.sum()
                for inde in range(self.K):
                    p[inde] = (self.nd[i][inde] + self.alpha) / (self.nd[i].sum() + Valpha) * \
                              (self.taz[int(tmp_v[3])][inde] + self.beta) / (self.tazsum[inde] + Kbeta) * \
                              (self.dow[int(tmp_v[1])][inde] + self.gamma) / (self.dowsum[inde] + Kgamma) * \
                              tod_norm[inde] * \
                              dow_norm[inde]
                ll = ll + np.log(p.sum())
                n = n + 1
        self.pre_min = np.exp(ll / (-n))
        print(self.pre_min)
        pre.append(np.exp(ll / (-n)))

    def sampling(self, i, j):
        topic = self.Z[i][j]
        word = self.dpre.docs[i].words[j]
        tmp_v = self.dpre.id2word[self.dpre.docs[i].words[j]].strip().split(",")
        self.nw[word][topic] -= 1  # 每个单词对应的主题个数
        self.ndsum[i] -= 1
        self.nd[i][topic] -= 1  # 每个文档中有主题的单词数
        self.nwsum[topic] -= 1  # 每个主题下的单词计数
        # 针对每个单词 (id, time of the day, day of the week, duration, destination)
        self.todsum[topic] = self.todsum[topic] - float(tmp_v[0][:-3])
        self.dow[int(tmp_v[1])][topic] -= 1  # 7 * 主题数，每天对应的主题个数
        self.dowsum = self.dow.sum(axis=0)  # 主题数，每个主题下的天数
        self.duration[topic] = self.duration[topic] - float(tmp_v[2][:-3])
        self.taz[int(tmp_v[3])][topic] -= 1  # num_taz * 主题数，每个小区对应的主题个数
        self.tazsum = self.taz.sum(axis=0)  # 主题数，每个主题下的小区数

        Valpha = self.K * self.alpha
        Kbeta = num_taz * self.beta
        Kgamma = age_num * self.gamma
        tod_norm = np.zeros_like(self.p)
        dow_norm = np.zeros_like(self.p)
        # u:nd, nz: nw, w: dow, v: taz, s: todsum, q: duration
        for inde in range(self.K):
            smean = (self.tao * self.todsum[inde] + self.tao0 * self.mu0) / (
                    self.tao * self.nwsum[inde] + self.tao0)
            scov = 1 / (self.tao * self.nwsum[inde] + self.tao0) + 1 / self.tao

            qmean = (self.lamb * self.duration[inde] + self.lamb0 * self.yita0) / (
                    self.lamb * self.nwsum[inde] + self.lamb0)
            qcov = 1 / (self.lamb * self.nwsum[inde] + self.lamb0) + 1 / self.lamb

            tod_norm[inde] = self.normal(smean, scov, float(tmp_v[0][:-3]))
            dow_norm[inde] = self.normal(qmean, qcov, float(tmp_v[2][:-3]))
        # tod_norm = tod_norm / tod_norm.sum()
        # dow_norm = dow_norm / dow_norm.sum()
        for inde in range(self.K):
            self.p[inde] = (self.nd[i][inde] * cont1 + self.alpha) / (self.nd[i].sum() + Valpha) * \
                           (self.taz[int(tmp_v[3])][inde] * cont + self.beta) / (self.tazsum[inde] + Kbeta) * \
                           (self.dow[int(tmp_v[1])][inde] * cont1 + self.gamma) / (self.dowsum[inde] + Kgamma) * \
                           tod_norm[inde] * \
                           dow_norm[inde]
        # np.random.normal(smean, np.sqrt(scov), 1) * \
        # np.random.normal(qmean, np.sqrt(qcov), 1)

        # self.p = self.softmax(self.p)
        self.p = self.p / self.p.sum()
        np.random.seed(100)
        topic = np.argmax(np.random.multinomial(1, self.p))
        self.nw[word][topic] += 1
        self.nwsum[topic] += 1
        self.nd[i][topic] += 1
        self.ndsum[i] += 1
        self.todsum[topic] = self.todsum[topic] + float(tmp_v[0][:-3])
        self.dow[int(tmp_v[1])][topic] += 1  # 7 * 主题数，每天对应的主题个数
        self.dowsum = self.dow.sum(axis=0)  # 主题数，每个主题下的天数
        self.duration[topic] = self.duration[topic] + float(tmp_v[2][:-3])
        self.taz[int(tmp_v[3])][topic] += 1  # num_taz * 主题数，每个小区对应的主题个数
        self.tazsum = self.taz.sum(axis=0)  # 主题数，每个主题下的小区数
        return topic

    def normal(self, u, sig, x):
        return np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def est(self):
        for x in range(self.iter_times):
            for i in range(self.dpre.docs_count):
                for j in range(self.dpre.docs[i].length):
                    topic = self.sampling(i, j)
                    self.Z[i][j] = topic
            self.perplexity()
        print(u"迭代完成。")
        self._theta()
        self._phi()
        self._pai()
        self._mu()
        self._yita()
        self.save()

    def _pai(self):
        for i in range(self.dpre.docs_count):
            self.pai[i] = (self.nd[i] + self.alpha) / (self.nd[i].sum() + self.K * self.alpha)

    def _theta(self):
        for i in range(num_taz):
            self.theta[i] = (self.taz[i] + self.beta) / (self.tazsum + num_taz * self.beta)
        for y in range(self.K):
            self.theta[:, y] /= self.theta[:, y].sum()

    def _phi(self):  # day of week
        for i in range(age_num):
            self.phi[i] = (self.dow[i] + self.gamma) / (self.dowsum + age_num * self.gamma)
        for y in range(self.K):
            self.phi[:, y] /= self.phi[:, y].sum()

    def _mu(self):
        for i in range(self.K):
            self.mu[i] = (self.tao * self.todsum[i] + self.tao0 * self.mu0) / (
                    self.tao * self.nwsum[i] + self.tao0)

    def _yita(self):
        for i in range(self.K):
            self.yita[i] = (self.lamb * self.duration[i] + self.lamb0 * self.yita0) / (
                    self.lamb * self.nwsum[i] + self.lamb0)

    def save(self):
        print(u"taz - 主题分布已保存到%s" % self.thetafile)
        with codecs.open(self.thetafile, 'w') as f:
            for x in range(num_taz):
                for y in range(self.K):
                    f.write(str(self.theta[x][y]) + '\t')
                f.write('\n')

        print(u"Day of week - 主题分布已保存到%s" % self.phifile)
        with codecs.open(self.phifile, 'w') as f:
            for x in range(age_num):
                for y in range(self.K):
                    f.write(str(self.phi[x][y]) + '\t')
                f.write('\n')

        print(u"mu nz tao - 主题分布已保存到%s" % self.mufile)
        with codecs.open(self.mufile, 'w') as f:
            for i in range(self.K):
                f.write(str(self.mu[i]) + '\t' + str(self.nwsum[i]))
                f.write('\n')
            f.write(str(self.tao) + '\t' + str(self.tao0))

        print(u"yita nz lambda - 主题分布已保存到%s" % self.yitafile)
        with codecs.open(self.yitafile, 'w') as f:
            for i in range(self.K):
                f.write(str(self.yita[i]) + '\t' + str(self.nwsum[i]))
                f.write('\n')
            f.write(str(self.lamb) + '\t' + str(self.lamb0))

        print(u"参数设置已保存到%s" % self.paramfile)
        with codecs.open(self.paramfile, 'w', 'gbk') as f:
            f.write('K=' + str(self.K) + '\n')
            f.write('alpha=' + str(self.alpha) + '\n')
            f.write('beta=' + str(self.beta) + '\n')
            f.write(u'迭代次数  iter_times=' + str(self.iter_times) + '\n')
            f.write(u'每个类的高频词显示个数  top_words_num=' + str(self.top_words_num) + '\n')

        print(u"主题topN词已保存到%s" % self.topNfile)

        with codecs.open(self.topNfile, 'w', 'gbk') as f:
            self.top_words_num = min(self.top_words_num, self.dpre.words_count)
            for x in range(self.K):
                f.write(u'第' + str(x) + u'类：' + '\n')
                twords = [(n, self.phi[n][x]) for n in range(age_num)]
                # twords.sort(key=lambda i: i[1], reverse=True)
                for y in range(self.top_words_num):
                    word = OrderedDict({value: key for key, value in self.dpre.word2id.items()})[twords[y][0]]
                    f.write('\t' * 2 + word + '\t' + str(twords[y][1]) + '\n')

        print(u"文章-词-主题分派结果已保存到%s" % self.tassginfile)

        with codecs.open(self.tassginfile, 'w') as f:
            for x in range(self.dpre.docs_count):
                for y in range(self.dpre.docs[x].length):
                    f.write(str(self.dpre.docs[x].words[y]) + ':' + str(self.Z[x][y]) + '\t')
                f.write('\n')

        print(u"模型训练完成。")


def run():
    dpre = word_idmap.preprocessing(train_file)
    lda = LDAModel(dpre)
    lda.est()
    lda.plt_tpc()


# pre.append(lda.pre_min)


if __name__ == '__main__':
    pre = []
    run()
    print(pre)



