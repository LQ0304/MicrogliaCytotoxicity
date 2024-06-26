import numpy as np
import os
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.rdmolops import FastFindRings
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
import joblib
import matplotlib as mpl
from PIL import ImageDraw, ImageFont
import shap
import pandas as pd
import matplotlib.pyplot as plt

# 给定shap_index_list = [35, 203, 101, 335, 301, 479, 549, 459, 875]
# 得到对应的子结构，高亮结构分子，对应三个模型的shap分析

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 14


def mol_to_sparse_ecfp4(mol):
    Sparse_Info = {}
    mol.UpdatePropertyCache()
    FastFindRings(mol)  # property update
    fp = AllChem.GetMorganFingerprint(mol, 2, bitInfo=Sparse_Info)
    return fp, Sparse_Info


def mol_to_ecfp4(mol, fpszie):
    BitVec_Info = {}
    mol.UpdatePropertyCache()
    FastFindRings(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fpszie, bitInfo=BitVec_Info)
    fp_bit = np.zeros(len(fp))
    DataStructs.ConvertToNumpyArray(fp, fp_bit)
    return fp, fp_bit, BitVec_Info


def mol_atom_bond(mol, patt):
    hit_atom = mol.GetSubstructMatch(patt)
    hit_bond = []
    for bond in patt.GetBonds():
        aid1 = hit_atom[bond.GetBeginAtomIdx()]
        aid2 = hit_atom[bond.GetEndAtomIdx()]
        hit_bond.append(mol.GetBondBetweenAtoms(aid1, aid2).GetIdx())
    return hit_atom, hit_bond


sub_img_size_x = 500
sub_img_size_y = 700
font = ImageFont.truetype('times.ttf', 30)
n_bit = 1024

# load non FS
'''6种分子表征形式：不同分子描述符'''
repre_list = ['Avalon', 'ECFP', 'FCFP', 'Rdkit']
s = repre_list[1]
load_data_path = './data/save_data_BM/ECFP/'
train_x_origin = np.load(os.path.join(load_data_path, s + '_' + 'train_x7.npy'))
train_y_origin = np.load(os.path.join(load_data_path, s + '_' + 'train_y7.npy'))
test_x_origin = np.load(os.path.join(load_data_path, s + '_' + 'test_x7.npy'))
test_y_origin = np.load(os.path.join(load_data_path, s + '_' + 'test_y7.npy'))
x_data_origin = np.array(np.concatenate([train_x_origin, test_x_origin], axis=0), dtype='int')
y_data_origin = np.array(np.concatenate([train_y_origin, test_y_origin], axis=0), dtype='int')

# load FS feature
FS_path = './data/FS_RFE/ECFP_RFE_index.npy'
FS_index = np.load(FS_path)
print('RFE model select %d features' % (len(FS_index)))

# load SVM predict model
load_model_path_SVM = './data/FS_BM_SVM_Results/ECFP_RFE/FS_BM_SVM_ECFP_RFE7.pkl'
Predict_model_SVM = joblib.load(load_model_path_SVM)
# load SVM explanation model
load_inter_model_SVM = './data/Shap_SVM_RFE_Results/shap_SVM_model.pkl'
inter_model_SVM = joblib.load(load_inter_model_SVM)

# load RF predict model
load_model_path_RF = './data/FS_BM_RF_Results/ECFP_RFE/FS_BM_RF_ECFP_RFE7.pkl'
Predict_model_RF = joblib.load(load_model_path_RF)
# load RF explanation model
load_inter_model_RF = './data/Shap_RF_RFE_Results/shap_RF_model.pkl'
inter_model_RF = joblib.load(load_inter_model_RF)

# load GBDT predict model
load_model_path_GBDT = './data/FS_BM_GBDT_Results/ECFP_RFE/FS_BM_GBDT_ECFP_RFE7.pkl'
Predict_model_GBDT = joblib.load(load_model_path_GBDT)
# load GBDT explanation model
load_inter_model_GBDT = './data/Shap_GBDT_RFE_Results/shap_GBDT_model.pkl'
inter_model_GBDT = joblib.load(load_inter_model_GBDT)

# read train_data and test_data
with open('./data/BM_Train_Test_dataset/BM_train_dataset7.txt', 'r') as f:
    data_line1 = f.readlines()
with open('./data/BM_Train_Test_dataset/BM_test_dataset7.txt', 'r') as f:
    data_line2 = f.readlines()
data_line = data_line1 + data_line2
print('Train and test data contain:%d' % (len(data_line)))

# 按照训练集和测试集的顺序得到对应的mol_name，compound_name
mol_list = []
label_list = []
plt_compound_name_list = []
for data in data_line:
    smiles, compound_name, label = data.strip('\n').split('\t')
    mol = Chem.MolFromSmiles(smiles)
    mol_list.append(mol)
    label = int(label)
    label_list.append(label)
    plt_compound_name_list.append(compound_name)

# shap_index_list = [35, 203, 101, 335, 301, 479, 549, 459, 875]
shap_index_list = [301, 580, 101, 549, 35, 459]
# shap_index_list = [203]

for shap_index in shap_index_list:

    save_picture_path = './data/Inter5_novel/Pictures%d' % (shap_index)
    os.makedirs(save_picture_path, exist_ok=True)

    # Contain feature number
    X_data_contain_feature = x_data_origin[:, shap_index]
    Contain_index = list(np.where(X_data_contain_feature == 1)[0])
    Contain_amount = len(Contain_index)
    print('Contain sub feature:%d' % Contain_amount)

    smart_list = []
    sub_mol_list = []
    Contain_smiles_list = []
    Contain_compound_name_List = []
    Contain_label_list = []
    for mol_ind in Contain_index:
        FS_x_data = (np.array(x_data_origin[mol_ind, FS_index], dtype='int')).reshape(1, -1)
        FS_x_data_df = pd.DataFrame(FS_x_data, columns=[str(int(i)) for i in list(FS_index)])

        inter_values_SVM = inter_model_SVM.shap_values(FS_x_data)
        inter_values_RF = inter_model_RF.shap_values(FS_x_data)
        inter_values_GBDT = inter_model_GBDT.shap_values(FS_x_data)

        mol = mol_list[mol_ind]
        smiles = Chem.MolToSmiles(mol, canonical=True)
        actual_label = int(label_list[mol_ind])
        plt_compound_name = plt_compound_name_list[mol_ind]

        Contain_smiles_list.append(smiles)
        Contain_compound_name_List.append(plt_compound_name)
        Contain_label_list.append(str(actual_label))

        _, _, bitVec_info = mol_to_ecfp4(mol, n_bit)
        _, sparse_info = mol_to_sparse_ecfp4(mol)

        # predict label
        predict_label_SVM = Predict_model_SVM.predict(FS_x_data)
        predict_label_RF = Predict_model_RF.predict(FS_x_data)
        predict_label_GBDT = Predict_model_GBDT.predict(FS_x_data)

        for info_key, info_value in bitVec_info.items():
            if info_key == shap_index:
                radius = info_value[0][1]
                rank_atom = info_value[0][0]
                amap = {}
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, rank_atom)
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                subsmiles = Chem.MolToSmiles(submol, canonical=True)
                # subsmiles = Chem.MolToSmiles(submol)
                if subsmiles not in smart_list:
                    smart_list.append(subsmiles)
                    # img_sub = Draw.MolToImage(submol, kekulize=False, wedgeBonds=False)
                    # img_sub.save(os.path.join(save_picture_path, 'sub_structure%d.png' % (mol_ind)),
                    #              bbox_inches='tight')
                    for sparse_info_key, _ in sparse_info.items():
                        if sparse_info_key % n_bit == shap_index:
                            # query_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)  # 去掉手性
                            query_mol = Chem.MolFromSmiles(smiles)
                            query_mol.UpdatePropertyCache(strict=False)
                            Chem.Kekulize(query_mol, clearAromaticFlags=True)  # 否则会出现错误
                            res = Draw.DrawMorganBit(query_mol, sparse_info_key,
                                                     sparse_info, whichExample=0)  # 如果用这个，就得用Sparse_Morgan
                            res.save(os.path.join(save_picture_path, 'sub_structure%d.png' % (mol_ind)),
                                     dpi=(600, 600),
                                     bbox_inches='tight')
                if mol.HasSubstructMatch(submol):
                    print('%d smiles contain substructure' % (mol_ind))
                    hit_atom_list, hit_bond_list = mol_atom_bond(mol, submol)
                    legend = 'Actual_label: %d' % (actual_label) + '\n' + \
                             'Pre_RF_label:%d' % (predict_label_RF) + '\n' + \
                             'Pre_SVM_label:%d' % (predict_label_SVM) + '\n' + \
                             'Pre_GBDT_label:%d' % (predict_label_GBDT) + '\n' + \
                             '%s' % (plt_compound_name)

                    img = Draw.MolToImage(mol,
                                          size=(sub_img_size_x, sub_img_size_y),
                                          highlightAtoms=hit_atom_list,
                                          highlightBonds=hit_bond_list, legend=None)
                    draw = ImageDraw.Draw(img)  # 画图
                    draw.text((sub_img_size_x / 2 - 100, sub_img_size_y - 160), legend, font=font, fill=(0, 0, 0))
                    img.save(os.path.join(save_picture_path,
                                          'molecules%d_%s.png' % (mol_ind, plt_compound_name)),
                             dpi=(1000, 1000))
                    img.close()

                    # fig, ax1 = plt.subplots(constrained_layout=True)
                    shap.force_plot(inter_model_SVM.expected_value,
                                    inter_values_SVM,
                                    FS_x_data_df,
                                    figsize=(15, 2),
                                    matplotlib=True,
                                    show=False)
                    plt.savefig(os.path.join(save_picture_path, 'shap_SVM_%d_%s.png' % (mol_ind, plt_compound_name)),
                                dpi=600,
                                bbox_inches='tight')
                    plt.close()
                    # fig, ax1 = plt.subplots(constrained_layout=True)
                    shap.force_plot(inter_model_RF.expected_value,
                                    inter_values_RF,
                                    FS_x_data_df,
                                    figsize=(15, 2),
                                    matplotlib=True,
                                    show=False)
                    plt.savefig(os.path.join(save_picture_path, 'shap_RF_%d_%s.png' % (mol_ind, plt_compound_name)),
                                dpi=600,
                                bbox_inches='tight')
                    plt.close()
                    # fig, ax1 = plt.subplots(constrained_layout=True)
                    shap.force_plot(inter_model_GBDT.expected_value,
                                    inter_values_GBDT,
                                    FS_x_data_df,
                                    figsize=(15, 2),
                                    matplotlib=True,
                                    show=False)
                    plt.savefig(os.path.join(save_picture_path, 'shap_GBDT_%d_%s.png' % (mol_ind, plt_compound_name)),
                                dpi=600,
                                bbox_inches='tight')
                    plt.close()

    print(smart_list)
    with open('./data/Inter5_novel/Smart%d' % shap_index, 'w') as f:
        for smart in smart_list:
            f.write(smart + '\n')

    print('Contain%d_is %d' % (shap_index, len(Contain_smiles_list)))
    with open('./data/Inter5_novel/smiles_name%d' % shap_index, 'w') as f:
        for s in range(len(Contain_smiles_list)):
            smiles = Contain_smiles_list[s]
            compound_name = Contain_compound_name_List[s]
            label = Contain_label_list[s]
            f.write(smiles + '\t' + compound_name + '\t' + label + '\n')

# for sparse_info_key, _ in sparse_info.items():
#     if sparse_info_key % n_bit == origin_feature_index:
#         query_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)  # 去掉手性
#         query_mol = Chem.MolFromSmiles(query_smiles)
#         # Chem.AssignStereochemistry(query_mol, flagPossibleStereoCenters=True, force=True)
#         res = Draw.DrawMorganBit(query_mol, sparse_info_key, sparse_info)  # 如果用这个，就得用Sparse_Morgan
#         res.show()


#     else:
#         for n in range(len(info_value)):
#             info = info_value[n]
#             radius = info[1]
#             rank_atom = info[0]
#             amap = {}
#             env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, rank_atom)
#             submol = Chem.PathToSubmol(mol, env, atomMap=amap)
#             subsmiles = Chem.MolToSmiles(submol, canonical=True)
#             if subsmiles not in smart_list:
#                 smart_list.append(subsmiles)
#                 # img_sub = Draw.MolToImage(submol, kekulize=False, wedgeBonds=False)
#                 # img_sub.save(os.path.join(save_picture_path, 'sub_structure%d.png' % (mol_ind)),
#                 #              bbox_inches='tight')
#                 for sparse_info_key, _ in sparse_info.items():
#                     if sparse_info_key % n_bit == shap_index:
#                         query_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)  # 去掉手性
#                         query_mol = Chem.MolFromSmiles(query_smiles)
#                         query_mol.UpdatePropertyCache(strict=False)
#                         Chem.Kekulize(query_mol, clearAromaticFlags=True)  # 否则会出现错误
#                         res = Draw.DrawMorganBit(query_mol, sparse_info_key,
#                                                  sparse_info)  # 如果用这个，就得用Sparse_Morgan
#                         res.save(os.path.join(save_picture_path, 'sub_structure%d_%d.png' % (mol_ind, n)),
#                                  dpi=(600, 600),
#                                  bbox_inches='tight')
#             if mol.HasSubstructMatch(submol):
#                 print('%d smiles contain substructure' % (mol_ind))
#                 hit_atom_list, hit_bond_list = mol_atom_bond(mol, submol)
#                 legend = 'Actual_label: %d' % (actual_label) + '\n' + \
#                          'Pre_RF_label:%d' % (predict_label_RF) + '\n' + \
#                          'Pre_SVM_label:%d' % (predict_label_SVM) + '\n' + \
#                          'Pre_GBDT_label:%d' % (predict_label_GBDT) + '\n' + \
#                          '%s' % (plt_compound_name)
#
#                 img = Draw.MolToImage(mol,
#                                       size=(sub_img_size_x, sub_img_size_y),
#                                       highlightAtoms=hit_atom_list,
#                                       highlightBonds=hit_bond_list, legend=None)
#                 draw = ImageDraw.Draw(img)  # 画图
#                 draw.text((sub_img_size_x / 2 - 100, sub_img_size_y - 160), legend, font=font,
#                           fill=(0, 0, 0))
#                 img.save(os.path.join(save_picture_path,
#                                       'molecules%d_%s_%s_%d.png' % (
#                                           mol_ind, plt_mol_name, plt_compound_name, n)))
#                 img.close()
#
#                 # fig, ax1 = plt.subplots(constrained_layout=True)
#                 shap.force_plot(inter_model_SVM.expected_value,
#                                 inter_values_SVM,
#                                 FS_x_data_df,
#                                 figsize=(15, 2),
#                                 matplotlib=True,
#                                 show=False)
#                 plt.savefig(os.path.join(save_picture_path, 'shap_SVM_%d_%d.png' % (mol_ind, n)), dpi=600,
#                             bbox_inches='tight')
#
#                 # fig, ax1 = plt.subplots(constrained_layout=True)
#                 shap.force_plot(inter_model_RF.expected_value,
#                                 inter_values_RF,
#                                 FS_x_data_df,
#                                 figsize=(15, 2),
#                                 matplotlib=True,
#                                 show=False)
#                 plt.savefig(os.path.join(save_picture_path, 'shap_RF_%d_%d.png' % (mol_ind, n)), dpi=600,
#                             bbox_inches='tight')
#
#                 # fig, ax1 = plt.subplots(constrained_layout=True)
#                 shap.force_plot(inter_model_GBDT.expected_value,
#                                 inter_values_GBDT,
#                                 FS_x_data_df,
#                                 figsize=(15, 2),
#                                 matplotlib=True,
#                                 show=False)
#                 plt.savefig(os.path.join(save_picture_path, 'shap_GBDT_%d_%d.png' % (mol_ind, n)), dpi=600,
#                             bbox_inches='tight')
#                 plt.close()
#
# else:
#     break
