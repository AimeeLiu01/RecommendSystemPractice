#!/usr/bin/env python
# encoding: utf-8
'''
@author: liuchang
@software: PyCharm
@file: user_demo01.py
@time: 2020-03-02 20:59
@Python实现基于User的协同过滤算法
@Link: https://www.jianshu.com/p/d15ba37755d1
'''
import numpy as np
from math import sqrt

class Recommender:
    # data: 数据集，这里指users_rating
    # K: 表示得出最相近的K个近邻
    # sim_func: 表示使用计算相似度
    # n: 表示推荐的item的个数

    def __init__(self, data, k = 3, sim_func='pearson', n=12):
        # 数据初始化
        self.k = k
        self.n = n
        self.sim_func = sim_func
        if self.sim_func == 'pearson':
            self.fn = self.pearson_sim
        if type(data).__name__ =='dict':
            self.data = data

    # pearson相似度
    def pearson_sim(self, rating1, rating2):
        sum_x = 0
        sum_y = 0
        sum_xy = 0
        sum_x2 = 0
        sum_y2 = 0
        n = 0
        for key in rating1:
            if key in rating2:
                n += 1
                x = rating1[key]
                y = rating2[key]
                sum_x += x
                sum_y += y
                sum_xy = x * y
                sum_x2 += pow(x, 2)
                sum_y2 += pow(y, 2)
        if n == 0:
            return 0

        dinominator = sqrt(n * sum_x2 - pow(sum_x, 2)) * sqrt(n * sum_y2 - pow(sum_y, 2))
        if dinominator == 0:
            return 0
        else:
            return (n * sum_xy - sum_x * sum_y) / dinominator

    # 对用户相似度进行排序
    def user_sim_sort(self, user_id):
        distances = []
        for instance in self.data:
            if instance != user_id:
                dis = self.fn(self.data[user_id], self.data[instance])
                distances.append((instance, dis))
        distances.sort(key=lambda items: items[1], reverse=True)
        return distances


    # recommand主体函数
    def recommand(self, user_id):
        # 定义一个字典，用来存储推荐的电影和分数
        recommendations = []
        # 计算出user与其他所有用户的相似度，返回一个list
        user_sim = self.user_sim_sort(user_id)
        # 计算最近的k个近邻的总距离
        total_dis = 0.0
        for i in range(self.k):
            total_dis += user_sim[i][1]
        if total_dis == 0.0:
            total_dis = 1.0

        # 将与user最相近的k个人中user没有看过的书推荐给user,并且这里又做了一个分数的计算排名
        for i in range(self.k):
            neighbor_id = user_sim[i][0]
            weight = user_sim[i][1] / total_dis
            neighbor_ratings = self.data[neighbor_id]
            user_rating = self.data[user_id]

            for item_id in neighbor_ratings:
                if item_id not in user_rating:
                    if item_id not in recommendations:
                        recommendations[item_id] = neighbor_ratings[item_id] * weight
                    else:
                        recommendations[item_id] = recommendations[item_id] + neighbor_ratings[item_id] * weight
        recommendations = list(recommendations.items())
        # 做了一个排序
        recommendations.sort(key=lambda items: items[1], reverse=True)
        return recommendations[:self.n], user_sim

if __name__ == '__main__':
    user_rating = dict()
    data_path = "./ratings.csv"
    with open(data_path, 'r') as file:
        for line in file:
            items = line.strip().split(',')
            if items[0] not in user_rating:
                user_rating[items[0]] = dict()
            users_rating[items[0]][items[1]] = dict()
            users_rating[items[0]][items[1]] = float(items[2])

    user_id = '1'
    recomm = Recommender(user_rating)
    recommendations, user_sim = recomm.recommand(user_id)
    print("movie id list:", recommendations)
    print("near list:", user_sim[:15])

