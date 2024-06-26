import os
import joblib
import matplotlib as mpl
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem
from PIL import ImageDraw, ImageFont
import pandas as pd
import matplotlib.pyplot as plt
import shap
from rdkit.Chem.rdmolops import FastFindRings

# 给定ind 和shap_index, 可以得到对应的molecules.jpg和
# 对应的shap_RF, shap_SVM, shap_GBDT,可以调节图片大小
compound_ind = 248
# shap_index_list = [370, 948]
# compound-301 370 948
# compound-71 370
# compound-111  104
# compound-248  233
shap_index_list = [751, 231, 233]
sub_img_size_x = 500
sub_img_size_y = 600


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


mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 14

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

# read txt file
n_bit = 1024
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
# plt_mol_name_list = []
plt_compound_name_list = []
for data in data_line:
    smiles, mol_compound_name, label = data.strip('\n').split('\t')
    mol = Chem.MolFromSmiles(smiles)
    mol_list.append(mol)
    label = int(label)
    label_list.append(label)
    plt_compound_name_list.append(mol_compound_name)

plt_compound_name = 'compound-' + str(compound_ind)
ind = plt_compound_name_list.index(plt_compound_name)

# load FS feature
FS_path = './data/FS_RFE/ECFP_RFE_index.npy'
FS_index = np.load(FS_path)
print('RFE model select %d features' % (len(FS_index)))

# load predict model
load_model_SVM_path = './data/FS_BM_SVM_Results/ECFP_RFE/FS_BM_SVM_ECFP_RFE7.pkl'
Predict_model_SVM = joblib.load(load_model_SVM_path)

load_model_RF_path = './data/FS_BM_RF_Results/ECFP_RFE/FS_BM_RF_ECFP_RFE7.pkl'
Predict_model_RF = joblib.load(load_model_RF_path)

load_model_GBDT_path = './data/FS_BM_GBDT_Results/ECFP_RFE/FS_BM_GBDT_ECFP_RFE7.pkl'
Predict_model_GBDT = joblib.load(load_model_GBDT_path)

FS_x_data = (np.array(x_data_origin[ind, FS_index], dtype='int')).reshape(1, -1)
FS_x_data_df = pd.DataFrame(FS_x_data, columns=[str(int(i)) for i in list(FS_index)])
mol = mol_list[ind]

save_error_path = './data/Typical_Plot/molecules%d_%s/' % (ind, plt_compound_name)
os.makedirs(save_error_path, exist_ok=True)

_, bitVec, bitVec_info = mol_to_ecfp4(mol, n_bit)

for shap_index in shap_index_list:
    print(' %s Contain %s:%d' % (plt_compound_name, shap_index, int(bitVec[shap_index])))

actual_label = int(label_list[ind])
predict_label_SVM = int(Predict_model_SVM.predict(FS_x_data))
predict_label_RF = int(Predict_model_RF.predict(FS_x_data))
predict_label_GBDT = int(Predict_model_GBDT.predict(FS_x_data))
predict_value = predict_label_SVM + predict_label_RF + predict_label_GBDT

hit_atom_list = []
hit_bond_list = []

for info_key, info_value in bitVec_info.items():
    for shap_index in shap_index_list:
        if info_key == shap_index:
            if len(info_value) == 1:
                radius = info_value[0][1]
                rank_atom = info_value[0][0]
                amap = {}
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, rank_atom)
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                subsmiles = Chem.MolToSmiles(submol, canonical=True)
                print(str(shap_index) + ':' + subsmiles)

                if mol.HasSubstructMatch(submol):
                    # print('%d smiles contain substructure' % (mol_ind))
                    hit_atom, hit_bond = mol_atom_bond(mol, submol)
                    hit_atom_list.extend(list(hit_atom))
                    hit_bond_list.extend(hit_bond)

    # # plot molecule
    # legend = 'Actual_label: %d' % (actual_label) + '\n' + \
    #          'Pre_SVM_label:%d' % (predict_label_SVM) + '\n' + \
    #          '%s' % (plt_compound_name)

legend = 'Actual_label: %d' % (actual_label) + '\n' + \
         'Pre_RF_label:%d' % (predict_label_RF) + '\n' + \
         'Pre_SVM_label:%d' % (predict_label_SVM) + '\n' + \
         'Pre_GBDT_label:%d' % (predict_label_GBDT) + '\n' + \
         '%s' % (plt_compound_name)

if len(hit_atom_list) != 0:
    img = Draw.MolToImage(mol,
                          size=(sub_img_size_x, sub_img_size_y),
                          highlightAtoms=hit_atom_list,
                          highlightBonds=hit_bond_list, legend=None)
else:
    img = Draw.MolToImage(mol,
                          size=(sub_img_size_x, sub_img_size_y),
                          legend=None)

draw = ImageDraw.Draw(img)  # 画图
font = ImageFont.truetype('times.ttf', 30)
draw.text((sub_img_size_x / 2 - 100, sub_img_size_y - 160), legend, font=font, fill=(0, 0, 0))
img.save(os.path.join(save_error_path, 'molecules%d.png' % (ind)), dpi=(1000, 1000))
