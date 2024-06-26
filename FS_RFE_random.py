import matplotlib as mpl
import matplotlib.pyplot as plt
from utils_fingerprint import *
import os
import numpy as np
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import pandas as pd

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'

save_path = './data/save_data_random_FS'
os.makedirs(save_path, exist_ok=True)

load_data_path = './data/save_data_random/'

num_data = 10
# seed_num_list = [11, 22, 33, 44, 55, 66, 77, 88, 99, 100]

repre_list = ['Avalon', 'ECFP', 'FCFP', 'Rdkit', 'MACCS',
              'Chemphy', 'LSTM', 'GRU', 'MPNN']


# def RFE_FS(model, x_data, y_data, remove_index, FS_start,
#            FS_end, FS_step, RFE_step, s, save_path_FS, rank):
#     score = []  # 建立列表
#     coef_list = []
#     alpha_coef = list(np.arange(FS_start, FS_end, FS_step))
#     for i in tqdm(alpha_coef):
#         selector = RFE(model, n_features_to_select=i, step=RFE_step).fit(x_data, y_data)  # 最优特征
#         FS_x_data = selector.transform(x_data)  # 最优特征
#         once = cross_val_score(model, FS_x_data, y_data, cv=5, scoring='accuracy').mean()  # 交叉验证
#         once = float('%.4f' % once)
#         coef_list.append(i)
#         score.append(once)  # 交叉验证结果保存到列表
#     best_score = max(score)
#     best_alpha = coef_list[score.index(max(score))]
#     print(best_score, best_alpha)  # 输出最优分类结果和对应的特征数量
#
#     # plot
#     plt.figure()
#     plt.plot(alpha_coef, score, '-o')
#     plt.scatter(best_alpha, best_score, color='b', marker='o', edgecolors='r', s=200)
#     # plt.xticks(alpha_coef)
#     plt.xlabel('The number of features', fontsize=18)
#     plt.ylabel('Accuracy', fontsize=18)
#     plt.xlim([min(alpha_coef), max(alpha_coef)])
#     plt.ylim([min(score) - 0.01, max(score) + 0.01])
#     plt.savefig(os.path.join(save_path_FS, s + '_FS%d.png' % rank), dpi=1000)
#
#     # select_optimal
#     selector_best = RFE(model, n_features_to_select=best_alpha, step=20).fit(x_data, y_data)
#     RFE_FS_index = selector_best.support_
#     FS_index = remove_index[RFE_FS_index]
#     return FS_index, score, alpha_coef, best_score, best_alpha
#
#
# def remove_zero_column(x_data):
#     x_data_df = pd.DataFrame(x_data, columns=[str(i) for i in range(x_data.shape[1])])
#     x_data_remove = x_data_df.loc[:, ~(x_data_df == 0).all()]
#     x_data_rezero = np.array(x_data_remove)
#     rezero_index = np.array(x_data_remove.columns.tolist(), dtype='int')
#     return x_data_rezero, rezero_index


def save(save_path, s, train_x, train_y, test_x, test_y, rank):
    np.save(os.path.join(save_path, s + '_' + 'train_x%d.npy' % rank), train_x)
    np.save(os.path.join(save_path, s + '_' + 'train_y%d.npy' % rank), train_y)
    np.save(os.path.join(save_path, s + '_' + 'test_x%d.npy' % rank), test_x)
    np.save(os.path.join(save_path, s + '_' + 'test_y%d.npy' % rank), test_y)


#
# FS_model = SVC(kernel='linear', random_state=0)
# start = 100
# end = 150
# step_len = 1
# RFE_step = 20

for i in range(0, num_data):
    for res in range(0, 4):
        s = repre_list[res]
        load_data_path_s = os.path.join(load_data_path, s)
        train_x = np.load(os.path.join(load_data_path_s, s + '_' + 'train_x%d.npy' % i))
        print(os.path.join(load_data_path_s, s + '_' + 'train_x%d.npy' % i))
        train_y = np.load(os.path.join(load_data_path_s, s + '_' + 'train_y%d.npy' % i))
        test_x = np.load(os.path.join(load_data_path_s, s + '_' + 'test_x%d.npy' % i))
        test_y = np.load(os.path.join(load_data_path_s, s + '_' + 'test_y%d.npy' % i))
        s = s + '_RFE'
        save_path_s = os.path.join(save_path, s)
        os.makedirs(save_path_s, exist_ok=True)

        load_FS_path = os.path.join('./data/FS_RFE/', s + '_index.npy')
        FS_index = np.load(load_FS_path)
        train_fs_x = train_x[:, FS_index]
        test_fs_x = test_x[:, FS_index]

        save(save_path_s, s, train_fs_x, train_y, test_fs_x, test_y, i)
        print('%s contain the number of features:%d' % (s, len(FS_index)))

        # # Avalon
        # for i in seed_num_list:
        #     s = repre_list[0]
        #     load_data_path_Avalon = os.path.join(load_data_path, s)
        #     train_x = np.load(os.path.join(load_data_path_Avalon, s + '_' + 'train_x%d.npy' % i))
        #     print(os.path.join(load_data_path_Avalon, s + '_' + 'train_x%d.npy' % i))
        #     train_y = np.load(os.path.join(load_data_path_Avalon, s + '_' + 'train_y%d.npy' % i))
        #     test_x = np.load(os.path.join(load_data_path_Avalon, s + '_' + 'test_x%d.npy' % i))
        #     test_y = np.load(os.path.join(load_data_path_Avalon, s + '_' + 'test_y%d.npy' % i))
        #     x_data_Avalon = np.concatenate([train_x, test_x], axis=0)
        #     y_data_Avalon = np.concatenate([train_y, test_y], axis=0)
        #     x_data_Avalon_rezero, rezero_index = remove_zero_column(x_data_Avalon)
        #     s = s + '_RFE'
        #     save_path_Avalon = os.path.join(save_path, s)
        #     os.makedirs(save_path_Avalon, exist_ok=True)
        #     FS_index, Avalon_score, Avalon_alpha_coef, Avalon_best_score, Avalon_best_alpha = \
        #         RFE_FS(FS_model, x_data_Avalon_rezero, y_data_Avalon, rezero_index,
        #                start, end, step_len, RFE_step, s, save_path_Avalon, i)
        #     x_data_Avalon_FS = x_data_Avalon[:, FS_index]
        #     np.save(os.path.join(save_path_Avalon, s + '_' + 'index%d.npy' % i), FS_index)
        #     save(save_path_Avalon, s, x_data_Avalon_FS, y_data_Avalon, len(y_data_Avalon), len(train_y), i)
        #     print('Avalon contain the number of features:%d' % (len(x_data_Avalon_FS[1])))

        # # ECFP4
        # s = repre_list[1] + '_RFE'
        # _, fp_bit_list_ECFP4 = mol_to_ecfp4(mol_list, 1024)  # Avalon Fingerprint
        # x_data_ECFP4 = np.array(fp_bit_list_ECFP4, dtype='int')
        # y_data_ECFP4 = np.array(label_list, dtype='int')
        # x_data_ECFP4_rezero, rezero_index = remove_zero_column(x_data_ECFP4)
        # print('ECFP4 contain Non zero features:%d' % (len(x_data_ECFP4_rezero[1])))
        #
        # FS_index, ECFP4_score, ECFP4_alpha_coef, ECFP4_best_score, ECFP4_best_alpha = \
        #     RFE_FS(FS_model, x_data_ECFP4_rezero, y_data_ECFP4, rezero_index,
        #            start, end, step_len, RFE_step, s)
        #
        # np.save(os.path.join(save_path, s + '_' + 'index.npy'), FS_index)
        # x_data_ECFP4_FS = x_data_ECFP4[:, FS_index]
        # save(save_path, s, x_data_ECFP4_FS, y_data_ECFP4, len(data_line1), len(data_line))
        # print('ECFP4 contain the number of features:%d' % (len(x_data_ECFP4_FS[1])))
        #
        # # FCFP4
        # s = repre_list[2] + '_RFE'
        # _, fp_bit_list_FCFP4 = mol_to_fcfp4(mol_list, 1024)  # Avalon Fingerprint
        # x_data_FCFP4 = np.array(fp_bit_list_FCFP4, dtype='int')
        # y_data_FCFP4 = np.array(label_list, dtype='int')
        # x_data_FCFP4_rezero, rezero_index = remove_zero_column(x_data_FCFP4)
        # print('FCFP4 contain Non zero features:%d' % (len(x_data_FCFP4_rezero[1])))
        #
        # FS_index, FCFP4_score, FCFP4_alpha_coef, FCFP4_best_score, FCFP4_best_alpha = \
        #     RFE_FS(FS_model, x_data_FCFP4_rezero, y_data_FCFP4, rezero_index,
        #            start, end, step_len, RFE_step, s)
        #
        # np.save(os.path.join(save_path, s + '_' + 'index.npy'), FS_index)
        # x_data_FCFP4_FS = x_data_FCFP4[:, FS_index]
        # save(save_path, s, x_data_FCFP4_FS, y_data_FCFP4, len(data_line1), len(data_line))
        # print('FCFP4 contain the number of features:%d' % (len(x_data_FCFP4_FS[1])))
        #
        # # Rdkit
        # s = repre_list[3] + '_RFE'
        # _, fp_bit_list_rdkit = mol_to_fp(mol_list, 1024)  # Avalon Fingerprint
        # x_data_rdkit = np.array(fp_bit_list_rdkit, dtype='int')
        # y_data_rdkit = np.array(label_list, dtype='int')
        # x_data_rdkit_rezero, rezero_index = remove_zero_column(x_data_rdkit)
        # print('rdkit contain Non zero features:%d' % (len(x_data_rdkit_rezero[1])))
        #
        # FS_index, rdkit_score, rdkit_alpha_coef, rdkit_best_score, rdkit_best_alpha = \
        #     RFE_FS(FS_model, x_data_rdkit_rezero, y_data_rdkit, rezero_index,
        #            start, end, step_len, RFE_step, s)
        #
        # np.save(os.path.join(save_path, s + '_' + 'index.npy'), FS_index)
        # x_data_rdkit_FS = x_data_rdkit[:, FS_index]
        # save(save_path, s, x_data_rdkit_FS, y_data_rdkit, len(data_line1), len(data_line))
        # print('rdkit contain the number of features:%d' % (len(x_data_rdkit_FS[1])))

        # s = repre_list[12] + '_RFE'
        # train_data_path = './data/train_data.csv'
        # train_data = pd.read_csv(train_data_path)
        # fp_bit_MPNN_train = path_to_mpnn(train_data_path)
        # x_data_train_MPNN = np.array(fp_bit_MPNN_train)
        # train_y_MPNN = np.array(train_data['Labels'].values)
        #
        # test_data_path = 'data/test_data.csv'
        # test_data = pd.read_csv(test_data_path)
        # fp_bit_MPNN_test = path_to_mpnn(test_data_path)
        # x_data_test_MPNN = np.array(fp_bit_MPNN_test)
        # test_y_MPNN = np.array(test_data['Labels'].values)
        #
        # x_data_MPNN = np.concatenate([x_data_train_MPNN, x_data_test_MPNN], axis=0)
        # y_data_MPNN = np.concatenate([train_y_MPNN, test_y_MPNN], axis=0)
        #
        # x_data_mpnn_rezero, rezero_index = remove_zero_column(x_data_MPNN)
        # print('MPNN contain Non zero features:%d' % (len(x_data_mpnn_rezero[1])))
        #
        # FS_index, MPNN_score, MPNN_alpha_coef, MPNN_best_score, MPNN_best_alpha = \
        #     RFE_FS(FS_model, x_data_mpnn_rezero, y_data_MPNN, rezero_index,
        #            start, end, step_len, RFE_step, s)
        #
        # np.save(os.path.join(save_path, s + '_' + 'index.npy'), FS_index)
        # x_data_mpnn_FS = x_data_MPNN[:, FS_index]
        # save(save_path, s, x_data_mpnn_FS, y_data_MPNN, len(data_line1), len(data_line))
        # print('MPNN contain the number of features:%d' % (len(x_data_mpnn_FS[1])))

        # # concat ECFP and MPNN
        # s = repre_list[1] + repre_list[12] + '_RFE'
        # x_concat_ECFP_MPNN = np.concatenate([x_data_FCFP4, x_data_MPNN], axis=1)
        # y_concat_data = y_data_MPNN
        #
        # x_data_concat_rezero, rezero_index = remove_zero_column(x_concat_ECFP_MPNN)
        # print('concat contain Non zero features:%d' % (len(x_data_concat_rezero[1])))
        #
        # FS_index, ECFP_MPNN_score, ECFP_MPNN_alpha_coef, ECFP_MPNN_best_score, ECFP_MPNN_best_alpha = \
        #     RFE_FS(FS_model, x_data_concat_rezero, y_concat_data, rezero_index,
        #            start, end, step_len, RFE_step, s)
        #
        # np.save(os.path.join(save_path, s + '_' + 'index.npy'), FS_index)
        # x_data_mpnn_ecfp_FS = x_concat_ECFP_MPNN[:, FS_index]
        # save(save_path, s, x_data_mpnn_ecfp_FS, y_concat_data, len(data_line1), len(data_line))
        # print('concat contain the number of features:%d' % (len(x_data_mpnn_ecfp_FS[1])))

        # # MACCS
        # s = repre_list[4] + '_RFE'
        # _, fp_bit_list_MACCS = mol_to_maccs(mol_list)  # Avalon Fingerprint
        # x_data_MACCS = np.array(fp_bit_list_MACCS, dtype='int')
        # y_data_MACCS = np.array(label_list, dtype='int')
        # x_data_MACCS_rezero, rezero_index = remove_zero_column(x_data_MACCS)
        #
        # FS_index, MACCS_score, MACCS_alpha_coef, MACCS_best_score, MACCS_best_alpha = \
        #     RFE_FS(FS_model, x_data_MACCS_rezero, y_data_MACCS, rezero_index,
        #            start, end, step_len, RFE_step, s)
        #
        # np.save(os.path.join(save_path, s + '_' + 'index.npy'), FS_index)
        # x_data_MACCS_FS = x_data_MACCS[:, FS_index]
        # save(save_path, s, x_data_MACCS_FS, y_data_MACCS, len(data_line1), len(data_line))
        # print('MACCS contain the number of features:%d' % (len(x_data_MACCS_FS[1])))

        # # plot overall figure
        # min_xlim = min(Avalon_alpha_coef)
        # max_xlim = max(Avalon_alpha_coef)
        #
        # min_ylim = min(min(Avalon_score), min(ECFP4_score), min(FCFP4_score),
        #                min(rdkit_score)) - 0.01
        #
        # max_ylim = max(max(Avalon_score), max(ECFP4_score), max(FCFP4_score),
        #                max(rdkit_score)) + 0.01
        #
        # # min_ylim = min(min(Avalon_score), min(ECFP4_score), min(FCFP4_score),
        # #                min(rdkit_score), min(MPNN_score)) - 0.01
        # #
        # # max_ylim = max(max(Avalon_score), max(ECFP4_score), max(FCFP4_score),
        # #                max(rdkit_score), max(MPNN_score)) + 0.01
        # plt.figure()
        # plt.plot(Avalon_alpha_coef, Avalon_score, linewidth=1.5, marker="o", markersize=5)
        # plt.scatter(Avalon_best_alpha, Avalon_best_score, marker='*', s=180)
        #
        # plt.plot(ECFP4_alpha_coef, ECFP4_score, color='darkorange', linewidth=1.5, marker="o", markersize=5)
        # plt.scatter(ECFP4_best_alpha, ECFP4_best_score, color='darkorange', marker='*', s=180)
        #
        # plt.plot(FCFP4_alpha_coef, FCFP4_score, color='gold', linewidth=1.5, marker="o", markersize=5)
        # plt.scatter(FCFP4_best_alpha, FCFP4_best_score, color='gold', marker='*', s=180)
        #
        # plt.plot(rdkit_alpha_coef, rdkit_score, color='g', linewidth=1.5, marker="o", markersize=5)
        # plt.scatter(rdkit_best_alpha, rdkit_best_score, color='g', marker='*', s=180)
        #
        # # plt.plot(MACCS_alpha_coef, MACCS_score, '-o', color='purple')
        # # plt.scatter(MACCS_best_alpha, MACCS_best_score, color='r', marker='*', s=180)
        #
        # # plt.plot(MPNN_alpha_coef, MPNN_score, '-o', color='purple')
        # # plt.scatter(MPNN_best_alpha, MPNN_best_score, color='r', marker='*', s=180)
        #
        # # plt.plot(ECFP_MPNN_alpha_coef, ECFP_MPNN_score, '-o', color='pink')
        # # plt.scatter(ECFP_MPNN_best_alpha, ECFP_MPNN_best_score, color='r', marker='*', s=180)
        #
        # plt.xlabel('Num of features', fontsize=14)
        # plt.ylabel('Accuracy', fontsize=14)
        # plt.xlim([min_xlim, max_xlim])
        # plt.ylim([min_ylim, max_ylim])
        # # plt.legend(['Avalon', 'ECFP4', 'FCFP4', 'RDKit', 'LV-MPNN'])
        # plt.legend(['Avalon', 'ECFP4', 'FCFP4', 'RDKit'])
        # plt.savefig(os.path.join(save_path, 'Overall_FS.png'), dpi=1000)
