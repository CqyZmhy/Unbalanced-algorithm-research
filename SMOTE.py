#! /user/bin/python 3
# -*- coding: utf-8 -*-
# author: Scc_hy
# 2018-11-17
# SMOTE
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import copy
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


class TWO_SMOTE():
    """
    不平二分类人工插值法采样
    """

    def __init__(self,
                 K_neighbors=5,
                 N_need=200,
                 random_state=42):
        self.K_neighbors = K_neighbors
        self.N_need = N_need
        self.random_state = 42

    def get_param_describe(self):
        print(
            "算法参数: \n" +
            'K_neighbors: 和正样本相近的随机样本数' + "\n" +
            "N_need: 需要增加的正样本数 (N_need // 100 * a)" + "\n" +
            "random_state: 随机器设定" + "\n"
                                    "\nover_sample 参数：\n" +
            "x_data: 需要进行过采样的全部数据集(非文本DataFrame)" + "\n" +
            "y_label: 类别标签(非文本DataFrame.Series)" + "\n"
        )

    def div_data(self, x_data, y_label):
        """
        将数据依据类分开
        """
        tp = set(y_label)
        tp_less = [a for a in tp if sum(y_label == a) < sum(y_label != a)][0]
        data_less = x_data.iloc[y_label == tp_less, :]
        data_more = x_data.iloc[y_label != tp_less, :]
        tp.remove(tp_less)
        return data_less, data_more, tp_less, list(tp)[0]

    def get_SMOTE_sample(self, x_data, y_label):
        """
        获取需要抽样的正样本
        """
        sample = []
        data_less, data_more, tp_less, tp_more = self.div_data(x_data, y_label)
        n_integ = self.N_need // 100
        data_add = copy.deepcopy(data_less)
        if n_integ == 0:
            print('WARNING: PLEASE RE-ENTER N_need')
        else:
            for i in range(n_integ - 1):
                data_out = data_less.append(data_add)

        data_out.reset_index(inplace=True, drop=True)
        return data_out, tp_less

    def over_sample(self, x_data, y_label):
        """
        SMOTE算法简单实现
        """
        sample, tp_less = self.get_SMOTE_sample(x_data, y_label)
        knn = NearestNeighbors(n_neighbors=self.K_neighbors, n_jobs=-1).fit(sample)
        n_atters = x_data.shape[1]
        label_out = copy.deepcopy(y_label)
        new = pd.DataFrame(columns=x_data.columns)
        for i in range(len(sample)):  # 1. 选择一个正样本
            # 2.选择少数类中最近的K个样本
            k_sample_index = knn.kneighbors(np.array(sample.iloc[i, :]).reshape(1, -1),
                                            n_neighbors=self.K_neighbors + 1,
                                            return_distance=False)

            # 计算插值样本
            # 3.随机选取K中的一个样本
            np.random.seed(self.random_state)
            choice_all = k_sample_index.flatten()
            choosed = np.random.choice(choice_all[choice_all != 0])

            # 4. 在正样本和随机样本之间选出一个点
            diff = sample.iloc[choosed,] - sample.iloc[i,]
            gap = np.random.rand(1, n_atters)
            new.loc[i] = [x for x in sample.iloc[i,] + gap.flatten() * diff]
            label_out = np.r_[label_out, tp_less]

        new_sample = pd.concat([x_data, new])
        new_sample.reset_index(inplace=True, drop=True)
        return new_sample, label_out

# if __name__ == '__main__':
#     iris = load_iris()
#     irisdf = pd.DataFrame(data = iris.data, columns = iris.feature_names)
#     y_label = iris.target
#     # 生成不平二分类数据
#     iris_1 = irisdf.iloc[y_label == 1,]
#     iris_2 = irisdf.iloc[y_label == 2,]
#     iris_2imb = pd.concat([iris_1, iris_2.iloc[:10, :]])
#     label_2imb =np.r_[y_label[y_label == 1], y_label[y_label == 2][:10]]
#     iris_2imb.reset_index(inplace = True, drop = True)

#     smt  = TWO_SMOTE()
#     x1_new, y1_new = smt.over_sample(iris_2imb, label_2imb)
#     print('1、手动实现：')
#     print('x1_new:',x1_new)
#     print('y1_new:',y1_new)
#     print('y1_new_shape:',y1_new.shape)
#     print('-'*45)

#     sm = SMOTE(random_state = 42, n_jobs = -1)
#     x2_new, y2_new = sm.fit_sample(iris_2imb, label_2imb)
#     print('2、工具包实现：')
#     print('x2_new:',x2_new)
#     print('x2_new_shape:',x2_new.shape)
#     print('y2_new:',y2_new)
#     print('y2_new_shape:',y2_new.shape)
#     print('-'*90)