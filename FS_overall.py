import pandas as pd
from rdkit import Chem
from sklearn.feature_selection import RFE
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from utils_fingerprint import mol_to_Avalon, mol_to_ecfp4, mol_to_fcfp4, mol_to_fp
from sklearn.svm import SVC
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
repre_list = ['Avalon', 'ECFP', 'FCFP', 'Rdkit']


def RFE_FS(model, x_data, y_data, remove_index, FS_start,
           FS_end, FS_step, RFE_step, s, save_path_FS):
    score = []  # 建立列表
    coef_list = []
    alpha_coef = list(np.arange(FS_start, FS_end, FS_step))
    for i in tqdm(alpha_coef):
        selector = RFE(model, n_features_to_select=i, step=RFE_step).fit(x_data, y_data)  # 最优特征
        FS_x_data = selector.transform(x_data)  # 最优特征
        once = cross_val_score(model, FS_x_data, y_data, cv=5, scoring='accuracy').mean()  # 交叉验证
        once = float('%.4f' % once)
        coef_list.append(i)
        score.append(once)  # 交叉验证结果保存到列表
    best_score = max(score)
    best_alpha = coef_list[score.index(max(score))]
    print(best_score, best_alpha)  # 输出最优分类结果和对应的特征数量

    # plot
    plt.figure()
    plt.plot(alpha_coef, score, '-o')
    plt.scatter(best_alpha, best_score, color='b', marker='o', edgecolors='r', s=200)
    # plt.xticks(alpha_coef)
    plt.xlabel('The number of features', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.xlim([min(alpha_coef), max(alpha_coef)])
    plt.ylim([min(score) - 0.01, max(score) + 0.01])
    plt.savefig(os.path.join(save_path_FS, s + '_FS.png'), dpi=1000)

    # select_optimal
    selector_best = RFE(model, n_features_to_select=best_alpha, step=RFE_step).fit(x_data, y_data)
    RFE_FS_index = selector_best.support_
    FS_index = remove_index[RFE_FS_index]
    return FS_index, score, alpha_coef, best_score, best_alpha


def remove_zero_column(x_data):
    x_data_df = pd.DataFrame(x_data, columns=[str(i) for i in range(x_data.shape[1])])
    x_data_remove = x_data_df.loc[:, ~(x_data_df == 0).all()]
    x_data_rezero = np.array(x_data_remove)
    rezero_index = np.array(x_data_remove.columns.tolist(), dtype='int')
    return x_data_rezero, rezero_index


df = pd.read_excel('./data/Data sets for microglia cytotoxicity.xlsx')
data_line = df.values

mol_list = []
label_list = []
for data in data_line:
    smiles = data[0]
    label = data[1]
    mol = Chem.MolFromSmiles(smiles)
    mol_list.append(mol)
    label_list.append(label)
print('Data cantain:%d' % (len(mol_list)))
y_data = np.array(label_list)

save_path = './data/FS_RFE/'
os.makedirs(save_path, exist_ok=True)

FS_model = SVC(kernel='linear', random_state=0)
start = 100
end = 150
step_len = 1
RFE_step = 40

s0 = repre_list[0]
_, fp_bit_list_Avalon = mol_to_Avalon(mol_list, 1024)  # Avalon Fingerprint
x_data_Avalon = np.array(fp_bit_list_Avalon)
x_data_Avalon_rezero, Avalon_rezero_index = remove_zero_column(x_data_Avalon)
print('Avalon contain rezero contain:%d' % (x_data_Avalon_rezero.shape[1]))

s0_save = s0 + '_RFE'
Avalon_FS_index, Avalon_score, Avalon_alpha_coef, Avalon_best_score, Avalon_best_alpha = \
    RFE_FS(FS_model, x_data_Avalon_rezero, y_data, Avalon_rezero_index,
           start, end, step_len, RFE_step, s0_save, save_path)
np.save(os.path.join(save_path, s0_save + '_index.npy'), Avalon_FS_index)
print('%s contain the number of features:%d' % (s0_save, len(Avalon_FS_index)))

s1 = repre_list[1]
_, fp_bit_list_ECFP4 = mol_to_ecfp4(mol_list, 1024)
x_data_ECFP4 = np.array(fp_bit_list_ECFP4)
x_data_ECFP4_rezero, ECFP4_rezero_index = remove_zero_column(x_data_ECFP4)
print('ECFP4 contain rezero contain:%d' % (x_data_ECFP4_rezero.shape[1]))
s1_save = s1 + '_RFE'
ECFP4_FS_index, ECFP4_score, ECFP4_alpha_coef, ECFP4_best_score, ECFP4_best_alpha = \
    RFE_FS(FS_model, x_data_ECFP4_rezero, y_data, ECFP4_rezero_index,
           start, end, step_len, RFE_step, s1_save, save_path)
np.save(os.path.join(save_path, s1_save + '_index.npy'), ECFP4_FS_index)
print('%s contain the number of features:%d' % (s1_save, len(ECFP4_FS_index)))

s2 = repre_list[2]
_, fp_bit_list_FCFP4 = mol_to_fcfp4(mol_list, 1024)
x_data_FPFC4 = np.array(fp_bit_list_FCFP4)
x_data_FPFC4_rezero, FPFC4_rezero_index = remove_zero_column(x_data_FPFC4)
print('FCFP4 contain rezero contain:%d' % (x_data_FPFC4_rezero.shape[1]))
s2_save = s2 + '_RFE'
FPFC4_FS_index, FPFC4_score, FCFP4_alpha_coef, FCFP4_best_score, FCFP4_best_alpha = \
    RFE_FS(FS_model, x_data_FPFC4_rezero, y_data, FPFC4_rezero_index,
           start, end, step_len, RFE_step, s2_save, save_path)
np.save(os.path.join(save_path, s2_save + '_index.npy'), FPFC4_FS_index)
print('%s contain the number of features:%d' % (s2_save, len(FPFC4_FS_index)))

s3 = repre_list[3]
_, fp_bit_list_rdkit = mol_to_fp(mol_list, 1024)
x_data_rdkit = np.array(fp_bit_list_rdkit)
x_data_rdkit_rezero, rdkit_rezero_index = remove_zero_column(x_data_rdkit)
print('rdkit contain rezero contain:%d' % (x_data_rdkit_rezero.shape[1]))
s3_save = s3 + '_RFE'
rdkit_FS_index, rdkit_score, rdkit_alpha_coef, rdkit_best_score, rdkit_best_alpha = \
    RFE_FS(FS_model, x_data_rdkit_rezero, y_data, rdkit_rezero_index,
           start, end, step_len, RFE_step, s3_save, save_path)
np.save(os.path.join(save_path, s3_save + '_index.npy'), rdkit_FS_index)
print('%s contain the number of features:%d' % (s3_save, len(rdkit_FS_index)))

# plot overall figure
min_xlim = min(Avalon_alpha_coef)
max_xlim = max(Avalon_alpha_coef)

min_ylim = min(min(Avalon_score), min(ECFP4_score), min(FPFC4_score),
               min(rdkit_score)) - 0.01

max_ylim = max(max(Avalon_score), max(ECFP4_score), max(FPFC4_score),
               max(rdkit_score)) + 0.02

plt.figure()
plt.plot(Avalon_alpha_coef, Avalon_score, linewidth=1.5, marker="o", markersize=5)
plt.scatter(Avalon_best_alpha, Avalon_best_score, marker='*', s=180)

plt.plot(ECFP4_alpha_coef, ECFP4_score, color='darkorange', linewidth=1.5, marker="o", markersize=5)
plt.scatter(ECFP4_best_alpha, ECFP4_best_score, color='darkorange', marker='*', s=180)

plt.plot(FCFP4_alpha_coef, FPFC4_score, color='gold', linewidth=1.5, marker="o", markersize=5)
plt.scatter(FCFP4_best_alpha, FCFP4_best_score, color='gold', marker='*', s=180)

plt.plot(rdkit_alpha_coef, rdkit_score, color='g', linewidth=1.5, marker="o", markersize=5)
plt.scatter(rdkit_best_alpha, rdkit_best_score, color='g', marker='*', s=180)

plt.xlabel('Num of features', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xlim([min_xlim, max_xlim])
plt.ylim([min_ylim, max_ylim])
# plt.legend(['Avalon', 'ECFP4', 'FCFP4', 'RDKit', 'LV-MPNN'])
plt.legend([repre_list[0], repre_list[1], repre_list[2], repre_list[3]], loc='upper center', ncol=4)
plt.savefig(os.path.join(save_path, 'Overall_FS.png'), dpi=1000)
