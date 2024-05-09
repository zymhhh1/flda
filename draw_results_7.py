#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Zhishuai Lee
@time: 2019/11/4 11:08
@file: draw.py
@desc:
"""
import matplotlib.pyplot as plt
import numpy as np
import logging.config
import configparser
import codecs
import os
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.transforms as mtrans

path = os.getcwd()
# plt.style.use(['ieee'])
plt.rcParams['font.size'] = 25  # 设置字体大小
plt.rcParams['font.family'] = 'Century'  # 设置字体样式
# from matplotlib import rc
# rc('font', **{'family': 'serif', 'serif': ['Century']})
# rc('text', usetex=True)
conf = configparser.ConfigParser()
conf.read("setting.conf")

K = int(conf.get("model_args", "K"))
num_taz = int(conf.get("model_args", "num_taz"))
tassginfile = os.path.join(path, os.path.normpath(conf.get("filepath", "tassginfile")))
train_file = os.path.join(path, os.path.normpath(conf.get("filepath", "trainfile")))
age_num = 3


def draw_phi(vari, ax, tle):
	if tle in list(range(1)):
		clr = "#1E76B4"
	elif tle in list(range(1, 4)):
		clr = "#C05046"
	elif tle in list(range(4, 5)):
		clr = "#52B9A0"
	elif tle in list(range(5, 8)):
		clr = "#C6C12E"
	elif tle in list(range(8, 12)):
		clr = "#CD853F"
	elif tle in list(range(12, 15)):
		clr = "#31859B"
	elif tle in list(range(15, 18)):
		clr = "#7E649E"
	# elif tle in list(range(12, 13)):
	# 	clr = "#F59D56"
	# elif tle in list(range(13, 15)):
	# 	clr = "#C64D83"
	else:
		clr = "#7F7F7F"
	with codecs.open("./data/freq_dis.dat", 'r', 'gbk') as f:
		docs = f.readlines()
		dow_dis = [int(item) for item in docs[0].split("\t")[:-1]]
	width = 0.4
	ind = np.arange(age_num)
	x_tick = ["13-24"]
	x_tick.extend(["25-59"])
	x_tick.extend(["60+"])
	class_name, value_count = np.unique(vari, return_counts=True)
	value_count = list(value_count)
	print(value_count)
	for i in range(len(dow_dis)):
		if float(ind[i]) not in list(class_name):
			value_count.insert(ind[i], 0)
	value_count = np.array(value_count) / dow_dis
	value_count = np.exp(value_count * 100) / sum(np.exp(value_count * 100))
	# value_count = value_count / value_count.sum()
	ax.bar(ind, np.array(value_count), color=clr)
	ax.set_xlabel("Age Group (years old)", fontsize=35)
	ax.set_xticks(ind)
	fmt = '%.2f'
	yticks = mtick.FormatStrFormatter(fmt)
	ax.yaxis.set_major_formatter(yticks)
	ax.set_xticklabels(x_tick, rotation=-45, fontsize=25)
	ax.set_ylabel('P(d|z)', fontsize=35, rotation=90)
	# ax.set_ylim(0, 0.15)
	trans = mtrans.Affine2D().translate(20, 0)
	for t in ax.get_xticklabels():
		t.set_transform(t.get_transform() + trans)


edu = []


def draw_theta(vari, ax, tle):
	if tle in list(range(1)):
		clr = "#1E76B4"
	elif tle in list(range(1, 4)):
		clr = "#C05046"
	elif tle in list(range(4, 5)):
		clr = "#52B9A0"
	elif tle in list(range(5, 8)):
		clr = "#C6C12E"
	elif tle in list(range(8, 12)):
		clr = "#CD853F"
	elif tle in list(range(12, 15)):
		clr = "#31859B"
	elif tle in list(range(15, 18)):
		clr = "#7E649E"
	# elif tle in list(range(12, 13)):
	# 	clr = "#F59D56"
	# elif tle in list(range(13, 15)):
	# 	clr = "#C64D83"
	else:
		clr = "#7F7F7F"
	with codecs.open("./data/POI2id.dat", 'r', 'gbk') as f:
		land = f.readlines()
	with codecs.open("./data/freq_dis.dat", 'r', 'gbk') as f:
		docs = f.readlines()
		taz_dis = [int(item) for item in docs[1].split("\t")[:-1]]
	dest = dict()
	ch2en = {"Home": "Home", "文化传媒": "Press Media",
	        "美食": "Catering.",
	         "医疗": "%-10s" % "Heal Serv.", "住宅区": "%-10s" % "Resident.", "写字楼": "Off. Bld "
		, "生活服务": "%-10s" % "Life Serv.", "休闲娱乐": "%-10s" % "Recreat.", "政府机构": "%-10s" % "Gov Ins.",
		     "公司企业": "%-10s" % "Company", "其它": "Else"
		, "购物": "%-10s" % "Shop Serv.", "教育培训": "%-10s" % "Edu Ins.", "酒店": "Hotel", "旅游景点": "%-10s" % "Attract.",
		     "汽车服务": "automobile service", "金融": "%-10s" % "Finance.", "运动健身": "Sports Serv.", "丽人": "Cosmetology"
		, "Else": "Else"}
	for item in land:
		if item != "":
			dest[item.split("\t")[1][:-1]] = item.split("\t")[0]
	class_name, value_count = np.unique(vari, return_counts=True)
	idx_taz_dis = []
	class_name = list(class_name)
	for i in range(len(class_name)):
		if class_name[i] in list(dest.values()):
			idx_taz_dis.append(list(dest.values()).index(list(class_name)[i]))
	val_ct = np.zeros([num_taz])
	for i, item in enumerate(idx_taz_dis):
		val_ct[item] = value_count[i]
	
	value_count = val_ct / taz_dis
	value_count = np.exp(value_count * 100) / sum(np.exp(value_count * 100))
	# value_count = value_count / value_count.sum()
	index = sorted(enumerate(value_count), key=lambda x: x[1], reverse=True)
	class_name = [list(dest.values())[idx[0]] for idx in index]
	ind = np.arange(num_taz)
	dict_key = [x for x in class_name]
	x_tick = [ch2en[i] for i in dict_key]
	y = np.array([x[1] for x in index])
	ax.bar(ind, y, color=clr)
	ax.set_xlabel('POIs of Destination', fontsize=35)
	ax.set_ylabel('P(c|z)', fontsize=35, rotation=90)
	ax.set_xticks(ind)
	# ax.set_ylim(0, 0.2)
	
	fmt = '%.2f'
	yticks = mtick.FormatStrFormatter(fmt)
	ax.yaxis.set_major_formatter(yticks)
	ax.xaxis.set_ticks_position("bottom")
	ax.set_xticklabels(x_tick, rotation=-45, fontsize=25)
	# ...
	trans = mtrans.Affine2D().translate(30, 0)
	for t in ax.get_xticklabels():
		t.set_transform(t.get_transform() + trans)


def draw_mu(data, ax, tle):
	if tle in list(range(1)):
		clr = "#1E76B4"
	elif tle in list(range(1, 4)):
		clr = "#C05046"
	elif tle in list(range(4, 5)):
		clr = "#52B9A0"
	elif tle in list(range(5, 8)):
		clr = "#C6C12E"
	elif tle in list(range(8, 12)):
		clr = "#CD853F"
	elif tle in list(range(12, 15)):
		clr = "#31859B"
	elif tle in list(range(15, 18)):
		clr = "#7E649E"
	# elif tle in list(range(12, 13)):
	# 	clr = "#F59D56"
	# elif tle in list(range(13, 15)):
	# 	clr = "#C64D83"
	else:
		clr = "#7F7F7F"
	# sns.distplot(data, color=clr, kde=True, ax=ax)
	# x = np.arange(np.min(data), np.max(data), 0.1)
	x = np.arange(np.mean(data) - 2, np.mean(data) + 2.1, 0.1)
	y = normfun(x, np.mean(data), np.std(data))
	ax.plot(x, y, linestyle="-", color=clr, linewidth=3)
	ax.set_xlabel('Arrival Time', fontsize=35)
	ax.set_ylabel('P(t|z)', fontsize=35, rotation=90)
	ax.set_title("Topic: {}".format(tle + 1), fontsize=50, pad=20)
	
	ax.set_xlim(xmin=0, xmax=24)
	ax.set_ylim(0, 0.8)


def normfun(x, mu, sigma):
	pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
	return pdf


def draw_yita(data, ax, tle, flag=0):
	if tle in list(range(1)):
		clr = "#1E76B4"
	elif tle in list(range(1, 4)):
		clr = "#C05046"
	elif tle in list(range(4, 5)):
		clr = "#52B9A0"
	elif tle in list(range(5, 8)):
		clr = "#C6C12E"
	elif tle in list(range(8, 12)):
		clr = "#CD853F"
	elif tle in list(range(12, 15)):
		clr = "#31859B"
	elif tle in list(range(15, 18)):
		clr = "#7E649E"
	# elif tle in list(range(12, 13)):
	# 	clr = "#F59D56"
	# elif tle in list(range(13, 15)):
	# 	clr = "#C64D83"
	else:
		clr = "#7F7F7F"
	# sns.distplot(data, color=clr, kde=True, ax=ax)
	ax.set_xlabel('Stay Duration (h)', fontsize=35)
	x = np.arange(np.mean(data) - 3, np.mean(data) + 3.1, 0.1)
	# x = np.arange(np.min(data), np.max(data), 0.1)
	y = normfun(x, np.mean(data), np.std(data))
	ax.plot(x, y, linestyle="-", color=clr, linewidth=3)
	ax.set_ylabel('P(s|z)', fontsize=35, rotation=90)
	ax.set_xlim(xmin=0, xmax=24)
	ax.set_ylim(0, 1)


def data_norm(path, norm_flag=False):
	d = list()
	with codecs.open("./data/tmp/model_{}.dat".format(path), 'r', 'unicode_escape') as f:
		docs = f.readlines()
	if not norm_flag:
		for line in docs:
			if line != "":
				tmp = line.strip().split("\t")
				d_ = list(map(eval, tmp))
				d.append(d_)
		return np.array(d).T
	else:
		nz = list()
		mean = list()
		for line in docs[:-1]:
			if line != "":
				tmp = line.strip().split("\t")
				d_ = list(map(eval, tmp))
				nz.append(d_[1])
				mean.append(d_[0])
		tao, tao0 = map(eval, docs[-1].strip().split("\t"))
		sigma = 1 / (tao * np.array(nz) + tao0) + 1 / tao
		return mean, sigma


def run(topic, t):
	def find_idx(name, key):
		idx = []
		first_pos = 0
		for i in range(name.count(key)):
			new_list = name[first_pos:]
			next_pos = new_list.index(key)
			first_pos += next_pos + 1
			idx.append(first_pos)
		return list(np.array(idx) - 1)
	
	l_idx = []
	with codecs.open(tassginfile, 'r', 'gbk') as f:
		docs = f.readlines()
		for line in docs:
			if line != "":
				row_l = []
				for item in line.strip().split("\t"):
					z = item.split(":")[1]
					row_l.append(z)
				l_idx.append(row_l)
	
	tmp = []
	with codecs.open(train_file, 'r', 'gbk') as f:
		docs = f.readlines()
		while "\n" in docs:
			del docs[docs.index("\n")]
		for i, line in enumerate(docs):
			if line != "":
				idx = find_idx(l_idx[i], key=str(topic))
				for ite in idx:
					if t == 1:
						tod = line.strip().split("\t\t")[ite].strip().split(",")[t]
					# if tod == "3.0":
					#     tod = "4.0"
					elif t == 3:
						tod = line.strip().split("\t\t")[ite].strip().split(",")[t]
					else:
						tod = line.strip().split("\t\t")[ite].strip().split(",")[t][:-3]
					tmp.append(tod)
	if t == 3:
		return tmp
	else:
		return np.array([float(item) for item in tmp])


def main():
	path = ["mu", "phi", "yita", "theta"]
	mean = list()
	cov = list()
	hist = list()
	for item in path:
		if item == "mu" or item == "yita":
			norm = True
			mean.append(data_norm(item, norm)[0])
			cov.append(data_norm(item, norm)[1])
		else:
			norm = False
			hist.append(data_norm(item, norm))
	K = 18
	fig, ax = plt.subplots(4, K // 3, figsize=(23 * 2, 10 * 2))
	# top_idx = [10, 12, 11, 13, 14, 15, 16, 17, 18, 19]
	top_idx = [0, 14, 10, 9, 6, 15, 16, 17, 1, 4, 12, 13, 2, 8, 7, 5, 11, 3]
	# top_idx = list(np.arange(K))
	for i in range(K // 3):
		draw_mu(run(top_idx[i], 0), ax[0][i], i)
		draw_phi(run(top_idx[i], 1), ax[1][i], i)
		draw_yita(run(top_idx[i], 2), ax[2][i], i)
		draw_theta(run(top_idx[i], 3), ax[3][i], i)
	# plt.subplots_adjust(wspace=2, hspace=0)  #
	plt.tight_layout()
	# print(np.array(edu).sum())
	plt.savefig("test1.pdf")
	# j = i + K // 2
	fig, ax = plt.subplots(4, K // 3, figsize=(23 * 2, 10 * 2))
	for j in range(K // 3, K * 2 // 3):
		i = j - K // 3
		draw_mu(run(top_idx[j], 0), ax[0][i], j)
		draw_phi(run(top_idx[j], 1), ax[1][i], j)
		draw_yita(run(top_idx[j], 2), ax[2][i], j)
		draw_theta(run(top_idx[j], 3), ax[3][i], j)
	plt.subplots_adjust(wspace=2, hspace=0)  #
	plt.tight_layout()
	# print(np.array(edu).sum())
	plt.savefig("test2.pdf")
	
	fig, ax = plt.subplots(4, K // 3, figsize=(23 * 2, 10 * 2))
	for j in range(K * 2 // 3, K):
		i = j - K * 2 // 3
		draw_mu(run(top_idx[j], 0), ax[0][i], j)
		draw_phi(run(top_idx[j], 1), ax[1][i], j)
		draw_yita(run(top_idx[j], 2), ax[2][i], j)
		draw_theta(run(top_idx[j], 3), ax[3][i], j)
	plt.tight_layout()
	# print(np.array(edu).sum())
	plt.savefig("test3.pdf")


# plt.show()


if __name__ == '__main__':
	main()
