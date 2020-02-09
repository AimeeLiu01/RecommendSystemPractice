#!/usr/bin/env python
# encoding: utf-8
'''
@author: liuchang
@software: PyCharm
@file: UserCF.py
@time: 2020-02-09 19:55
@注释：基于用户的协同过滤算法
'''

#导入包
import random
import math
import time
from tqdm import tqdm

## 一、通用函数定义

# 定义装饰器、监控运行时间
def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        stop_time = time.time()
        print("Func %s, run time: %s" % (func.__name__, stop_time - start_time))
        return res
    return wrapper


# 1. 数据处理相关
class Dataset():
    def __init__(self, fp):
        self.data = self.loadData(fp)

    @timmer
    def loadData(self, fp):
        data = []
        for l in open(fp):
            data.append(tuple(map(int, l.strip().split('::')[:2])))
        return data

    @timmer
    def splitData(self, M, k, seed=1):
        """
        :param: data, 加载的所有(user,item)数据条目
        :param M: 划分的数目，最后需要取M折的平均
        :param k: 本次是第几次划分
        :param seed: random的种子数，对于不同的K应该设置成一样的
        :return: train, test
        """
        train, test = [], []
        random.seed(seed)
        for user, item in self.data:
            if random.randint(0, M-1) == k:
                test.append((user, item))
            else:
                train.append((user, item))

        # 处理成字典的形式，user->set(items)
        def convert_dict(data):
            data_dict = {}
            for user, item in data:
                if user not in data_dict:
                    data_dict[user] = set()
                data_dict[user].add(item)
            data_dict = {k: list(data_dict[k]) for k in data_dict}
            return data_dict

        return convert_dict(train), convert_dict(test)


# 2.评价指标
# 1.Precision、2.Recall、3.Coverage 4.Popularity
class Metric():
    def __init__(self, train, test, GetRecommendation):
        """
        :param train: 训练数据
        :param test: 测试数据
        :param GetRecommendation: 为某个用户获取推荐物品的接口函数
        """
        self.train = train
        self.test = test
        self.GetRecommendation = GetRecommendation
        self.recs = self.getRes()


    # 为test中的每个用户进行推荐
    def getRec(self):
        recs = {}
        for user in self.test:
            rank = self.GetRecommendation(user)
            recs[user] = rank
        return recs


    # 定义精确率指标计算方式
    def precision(self):
        all, hit = 0, 0
        for user in self.test:
            test_items = set(self.test[user])
            rank = self.recs[user]
            for item, score in rank:
                if item in test_items:
                    hit += 1
            all += len(rank)
        return round(hit / all * 100, 2)

    # 定义召回率指标计算方式
    def recall(self):
        all, hit = 0, 0
        for user in self.test:
            test_items = set(self.test[user])
            rank = self.recs[user]
            for item, score in rank:
                if item in test_items:
                    hit += 1
                all += len(test_items)
        return round(hit / all * 100, 2)

    # 定义覆盖率指标计算方式
    def coverage(self):
        all_item, recom_item = set(), set()
        for user in self.test:
            for item in self.train[user]:
                all_item.add(item)
            rank = self.recs[user]
            for item, score in rank:
                recom_item.add(item)
        return round(len(recom_item)/len(all_item)*100, 2)

    # 定义新颖度指标计算方式
    def popularity(self):
        # 计算物品的流行度
        item_pop = {}
        for user in self.train:
            for item in self.train[user]:
                if item not in item_pop:
                    item_pop[item]  = 0
                item_pop[item] += 1

        num, pop = 0,0
        for user in self.test:
            rank = self.recs[user]
            for item, score in rank:
                pop += math.log(1 + item_pop[item])
                num += 1
        return round(pop/num, 6)


    def eval(self):
        metric = {'Precision': self.precision(),
                  'Recall': self.recall(),
                  'Coverage': self.coverage(),
                  'Popularity': self.popularity()}
        print('Metric:', metric)
        return metric



## 二、算法实现(1.Random、2.MostPopular、3.UserCF、4.UserIIF)

# 1.随机推荐
def Random(train, K, N):
    """
    :param train: 训练数据集
    :param K: 可忽略
    :param N: 超参数、设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    """
    items = {}
    for user in train:
        for item in train[user]:
            items[item] = 1


    def GetRecommendation(user):
        # 随机推荐N个未见过的
        user_items = set(train[user])
        rec_items = {k: items[k] for k in items if k not in user_items}
        rec_items = list(rec_items.items())
        random.shuffle(rec_items)
        return rec_items[:N]

    return GetRecommendation



# 2.热门推荐
def MostPopular(train, K, N):
    """
    :param train: 训练数据集
    :param K: 可忽略
    :param N: 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    """
    items = {}
    for user in train:
        for item in train[user]:
            if item not in items:
                items[item] = 0
            items[item] += 1

    def GetRecommendation(user):
        # 随机推荐N个没见过的最热门的
        user_items = set(train[user])
        rec_items = {k: items[k] for k in items if k not in user_items}
        rec_items = list(sorted(rec_items.items(), key=lambda x: x[1], reverse=True))
        return rec_items[:N]

    return GetRecommendation



# 3. 基于用户余弦相似度的推荐
def UserCF(train, K, N):
    """
    :param train: 训练数据集
    :param K: 超参数，设置取TopK相似用户数目
    :param N: 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    """
    # 计算item->user的倒排索引
    item_users = {}
    for user in train:
        for item in train[user]:
            if item not in item_users:
                item_users[item] = []
            item_users[item].append(user)


    # 计算用户相似度矩阵
    sim = {}
    num = {}
    for item in item_users:
        users = item_users[item]
        for i in range(len(users)):
            u = users[i]
            if u not in num:
                num[u] = 0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for j in range(len(users)):
                if j == i:
                    continue
                v = users[j]
                if v not in sim[u]:
                    sim[u][v] = 0
                sim[u][v] += 1
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(num[u] * num[v])


    # 按照相似度排序
    sorted_user_sim = {k: list(sorted(v.items(), \
                               key=lambda x: x[1], reverse=True)) \
                       for k, v in sim.items()}

    # 获取接口函数
    def GetRecommendation(user):
        items = {}
        seen_items = set(train[user])
        for u, _ in sorted_user_sim[user][:K]:
            for item in train[u]:
                # 去掉用户见过的
                if item not in seen_items:
                    if item not in items:
                        items[item] = 0
                    items[item] += sim[user][u]
        recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs

    return GetRecommendation

























