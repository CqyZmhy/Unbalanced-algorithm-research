# author:qcyx
# /*
# # D-SMOTE 算法实现
# */
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
import random


class D_SMOTE():

    def __init__(self, random_state=42):
        self.random_state = 42

    def decide_d_smote_df_label_of_minority(self, x_data, y_label):
        """
        判断少数类与多数类
        """
        x_data.insert(x_data.shape[1], 'label', y_label)
        df_label_of_minority = x_data[x_data['label'] == 1]
        minority_length = df_label_of_minority.shape[0]

        df_label_of_majority = x_data[x_data['label'] == 0]
        majority_length = df_label_of_majority.shape[0]

        if minority_length > majority_length:
            df_label_of_minority = x_data[x_data['label'] == 0]
            df_label_of_majority = x_data[x_data['label'] == 1]

        return df_label_of_minority, df_label_of_majority

    def over_sample(self, x_data, y_label):
        """
        D-SMOTE算法简单实现
        """
        y_new = []
        df_label_of_minority, df_label_of_majority = self.decide_d_smote_df_label_of_minority(
            x_data, y_label)
        n_need = df_label_of_majority.shape[0] - df_label_of_minority.shape[0]
        for iy in range(n_need):
            if (df_label_of_minority['label'].values)[0] == 1:
                y_new.append(1)
            else:
                y_new.append(0)
        array_y_new = np.array(y_new)
        y_minority = df_label_of_minority['label']
        df_label_of_minority.drop(['label'], axis=1, inplace=True)
        y_majority = df_label_of_majority['label']
        df_label_of_majority.drop(['label'], axis=1, inplace=True)
        #         print('test9-------------------------------------------start')
        #         print(array_y_new)
        #         print('test9-------------------------------------------end')
        # 计算出各个少数类样本点 ai ( i = 1，2，…，m) 到多数类样本点 bj ( j = 1，2，…，n) 的距离之和为∑d( ai，bj)
        num = np.zeros((df_label_of_minority.shape[0], 1))
        all_num = np.zeros((1))
        for i in range(df_label_of_minority.shape[0]):
            numj = np.zeros((1))
            for j in range(df_label_of_majority.shape[0]):
                #           计算每一个少数类样本点ai到多数类样本点 bj ( j = 1，2，…，n)的距离
                dis = cdist(df_label_of_minority.iloc[[
                    i]], df_label_of_majority.iloc[[j]], metric='euclidean')
                #         print(dis)
                numj[0:] = numj[0:] + dis[0:]
            #         print(numj)
            num[i, :] = num[i, :] + numj[0:]
        #     print(num)
        print(num)

        # 求平均距离
        for k in range(df_label_of_minority.shape[0]):
            all_num[0:] = all_num[0:] + num[k, :]
        # print(all_num)
        aver_d = all_num / (df_label_of_minority.shape[0])
        print(aver_d)

        decision_boundary_set = np.zeros((df_label_of_minority.shape[0], 1))
        for m in range(df_label_of_minority.shape[0]):
            if num[m, :] < aver_d:
                decision_boundary_set[m, :] = num[m, :]
        print(decision_boundary_set)

        # 对各个决策样本计算在少数类样本集中的 k 近邻
        # all_num_k_nearest = np.zeros((df_label_of_minority.shape[0],df_label_of_minority.shape[0]))
        # print(type(all_num_k_nearest))
        num_k_nearest = np.zeros(
            (df_label_of_minority.shape[0], df_label_of_minority.shape[0]))
        n_decision_sample_i = 0
        for m in range(0, df_label_of_minority.shape[0]):
            if decision_boundary_set[m, :] > 0:
                print("-----------------------")
                print(decision_boundary_set[m, :])
                print("-----------------------")
                n_decision_sample_i = n_decision_sample_i + 1
                for l in range(0, df_label_of_minority.shape[0]):
                    dis = cdist(df_label_of_minority.iloc[[
                        m]], df_label_of_minority.iloc[[l]], metric='euclidean')
                    num_k_nearest[l:, n_decision_sample_i - 1] = dis
        #         for n in range(len(num_k_nearest)):
        #             all_num_k_nearest[n:,i] = num_k_nearest[n:,]
        #         print(all_num_k_nearest)
        # print(all_num_k_nearest)
        print("n_decision_sample_i:-----------------------")
        print(n_decision_sample_i)
        print("-----------------------")
        print(num_k_nearest)
        # print(type(num_k_nearest))
        # 对数组进行排序，返回的是排序后的索引 按列排序
        d = np.sort(num_k_nearest, axis=0)
        print(d)
        nearest = np.argsort(num_k_nearest, axis=0)
        # print(type(nearest))
        print(nearest)
        print("随机取一个数:")
        k = 6
        print(len(nearest[0]))
        r2 = random.random()
        print(r2)
        print("3-------------------------------------------start")
        N = 1000
        M = 0
        new_array = np.zeros((N, df_label_of_minority.shape[1]))
        print(new_array)
        n_new = 0
        for m in range(k):
            if n_new == df_label_of_majority.shape[0] - df_label_of_minority.shape[0]:
                break
            for j in range(0, df_label_of_minority.shape[0]):
                if n_new == df_label_of_majority.shape[0] - df_label_of_minority.shape[0]:
                    break
                else:
                    a_new = np.abs(np.array(
                        df_label_of_minority.iloc[nearest[m][n_decision_sample_i - 1]]) - df_label_of_minority.iloc[
                                       [j]]) * r2 + df_label_of_minority.iloc[[j]]
                    #                     print(type(a_new))
                    #                     y_new.append(1)
                    #                     a_new['label'] = 1
                    n_new = n_new + 1
                if j == 0:
                    final_a_new = a_new
                else:
                    final_a_new = pd.concat([final_a_new, a_new])
            print(final_a_new)
            print('-' * 45)
            if m == 0:
                all_final_a_new = final_a_new
            else:
                all_final_a_new = pd.concat([all_final_a_new, final_a_new])
        #         print('-'*45)
        #         print(type(all_final_a_new.values))
        #         print(all_final_a_new)
        #         print('-'*45)
        #         print("3-------------------------------------------end")

        #         将新合成的少数类样本加入原样本中
        new_x_data = pd.concat([df_label_of_minority, df_label_of_majority, all_final_a_new])
        new_y_data = pd.concat([y_minority, y_majority, pd.Series(array_y_new)])

        #         print('test111--------------------------------------------start')
        #         print(type(new_x_data))
        #         print(new_x_data)
        #         print(type(new_y_data))
        #         print(new_y_data)
        #         print('test111--------------------------------------------end')
        return new_x_data.values, new_y_data