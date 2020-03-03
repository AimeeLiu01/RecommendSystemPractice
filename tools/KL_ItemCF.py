#!/usr/bin/env python
# encoding: utf-8
'''
@author: liuchang
@software: PyCharm
@file: KL_ItemCF.py
@time: 2020-03-02 17:48
'''
import pandas as pd
import numpy as np
import math
import operator

data = pd.read_excel("order_data2.xlsx")
data.head(5)

# 将空由0替换
data = data.fillna(0)
data = data[(data.product_id > 0)]
data["product_id"] = data["product_id"].apply(lambda x: int(x))

# 过滤一个用户至少买过两个商品的记录
user_data = data.groupby("user_id").size()
user_data = user_data[user_data > 1]
data = data[data.user_id.isin(user_data.keys())]
user_list = data.values.tolist()

all_product_id = list(set(data["product_id"].values.tolist()))

product_to_index = {}
index_to_product = {}
for index,value in enumerate(all_product_id):
    product_to_index[value] = index
    index_to_product[index] = value



# 第一步创建用户-物品的倒排索引
user_item_index = {}
for user_id in user_data.keys():
    product_ids = data[data.user_id == user_id]["product_id"].values.tolist()
    for index,value in enumerate(product_ids):
        product_ids[index] = product_to_index[value]
    user_item_index[user_id] = product_ids



# 第二步创建共现矩阵
product_length = len(product_to_index)
matrix_c = np.zeros((product_length,product_length))

# 循环用户-商品倒排索引 对于用一个用户购买的任意两个商品 在共现矩阵中要加1
for user_id in user_item_index:
    product_ids = user_item_index[user_id]
    for i, value in enumerate(product_ids):
        if(i < len(product_ids) - 1):
            list_other = product_ids[(i+1):len(product_ids)]
            for second_product_index in list_other:
                matrix_c[value][second_product_index] += 1
                matrix_c[second_product_index][value] += 1


# 第三步根据算法得到商品的相似矩阵
# 算法：cij/sqrt(|N(i)|*|N(j)|)
product_index_count_dic = {}
product_group = data.groupby("product_id").size()
for product_id in product_group.keys():
    product_index_count_dic[product_to_index[product_id]] = product_group[product_id]

matrix_w = np.zeros((product_length, product_length))

# 共现矩阵大于0的下标list
index_i_list, index_j_list = np.where(matrix_c > 0)
for index,value in enumerate(index_i_list):
    i = value
    j = index_j_list[index]
    score = matrix_c[i][j]/math.sqrt(product_index_count_dic[i] * product_index_count_dic[j])
    matrix_w[i][j] = score
    matrix_w[j][i] = score

a = np.zeros(product_length)
a[1] = 3
a[2] = 4
a[5] = 6
a = (a-np.min(a))/(np.max(a) - np.min(a))


def normalize(value):
    value = (value - np.min(value))/(np.max(value) - np.min(value))
    return value



# 第四步创建用户的喜好商品矩阵，并进行归一化
user_like_item_dic = {}
for user_id in user_data.keys():
    user_like_item = data[data.user_id == user_id]
    user_item_like_matrix = np.zeros(product_length)
    for i in range(len(user_like_item)):
        index = product_to_index[user_like_item.iloc[i].product_id]
        user_item_like_matrix[index] = user_like_item.iloc[i].orders_num
    user_like_item_dic[user_id] = normalize(user_item_like_matrix)


#获得最相似的k个商品
def getMostSimilar(matrix_w,index,k):
    c_list = matrix_w[index]
    similar_item = pd.DataFrame({"value":c_list})
    similar_item = similar_item.sort_values(by="value",ascending=False).iloc[0:k]
    similar_item_dic = {}
    for i in range(len(similar_item)):
        similar_item_dic[similar_item.iloc[i].name] = similar_item.iloc[i].value
    return similar_item_dic


#getMostSimilar(matrix_w,0,10)
#like_list = np.where(user_like_item_dic[753664] > 0)


def reommendItem(user_id,matrix_w,user_like_item_dic,k):
    recommend_dic = {}
    user_like_list = user_like_item_dic[user_id]
    user_like_item_index_list = np.where(user_like_list > 0)
    user_like_item_index_list = user_like_item_index_list[0]
    for product_index in user_like_item_index_list:
        like_score = user_like_list[product_index]
        most_similar_item = getMostSimilar(matrix_w,product_index,k)
        for key in most_similar_item.keys():
            if key in user_like_item_index_list:
                continue
            #最终得分是用户对商品的喜欢程度 * 商品的相似程度
            score = like_score * most_similar_item[key]
            if key in recommend_dic.keys():
                score += recommend_dic[key]
            recommend_dic[key] = score
    #返回得分最高的k个商品
    sorted_x = sorted(recommend_dic.items(), key=operator.itemgetter(1))
    sorted_x.reverse()
    return sorted_x[0:k]



# recommend_dic = reommendItem(27,matrix_w,user_like_item_dic,10)
# print("------------------")
# print(recommend_dic)

#第五步给用户推荐商品
def getAllUserRecommend():
    user_recommend = {}
    for user_id in user_like_item_dic.keys():
        print(user_id)
        recommend_dic = reommendItem(user_id,matrix_w,user_like_item_dic,10)
        value = ""
        for key in recommend_dic:
            index = key[0]
            if value == "":
                value += str(index_to_product[index])
            else:
                value += "," + str(index_to_product[index])
        user_recommend[user_id] = value
    return user_recommend




res = getAllUserRecommend()
print(res)




