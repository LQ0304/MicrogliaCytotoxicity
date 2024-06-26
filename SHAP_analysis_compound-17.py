from rdkit import Chem
import joblib
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont
from rdkit.Chem import Draw
import matplotlib as mpl
from rdkit.Chem.rdmolops import FastFindRings
from rdkit.Chem import Draw, AllChem
from rdkit import Chem, DataStructs

save_path = './data/SHAP_analysis_compound-71'
os.makedirs(save_path, exist_ok=True)


# subsize_y = 47, 71(110, 160)

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
mpl.rcParams['font.size'] = 16

shap_index = 101
sub_img_size_x = 500
sub_img_size_y = 500

shap_ind = '71'
shap_smiles_0 = 'CC(C)=CCC/C(C)=C/CC[C@@]1(C)Oc2c(c(=O)oc3cc(O)ccc23)[C@@H]1C'
shap_mol = Chem.MolFromSmiles(shap_smiles_0)
shap_smiles_can = Chem.MolToSmiles(shap_mol)
shap_mol = Chem.MolFromSmiles(shap_smiles_can)
#
# shap_ind = 'ind_71_22'
# shap_smiles_71_22 = 'CC(C)=CCC/C(C=N)=C/CC[C@@]1(C)Oc2c(c(=O)oc3cc(O)ccc23)[C@@H]1C'
# shap_mol = Chem.MolFromSmiles(shap_smiles_71_22)
#
# shap_ind = 'ind_71_23'
# shap_smiles_71_23 = 'CC(C)=CCC/C(C=O)=C/CC[C@@]1(C)Oc2c(c(=O)oc3cc(O)ccc23)[C@@H]1C'
# shap_mol = Chem.MolFromSmiles(shap_smiles_71_23)


_, shap_ecfp4, shap_bitVec_info = mol_to_ecfp4(shap_mol, 1024)

print('Contain #237:%d'%(int(shap_ecfp4[237])))

# load FS feature
FS_path = './data/FS_RFE/ECFP_RFE_index.npy'
FS_index = np.load(FS_path)
print('RFE model select %d features' % (len(FS_index)))

# load predict model
load_model_SVM_path = './data/FS_BM_SVM_Results/ECFP_RFE/FS_BM_SVM_ECFP_RFE7.pkl'
Predict_model_SVM = joblib.load(load_model_SVM_path)

FS_x_data = (np.array(shap_ecfp4[FS_index], dtype='int')).reshape(1, -1)
FS_x_data_df = pd.DataFrame(FS_x_data, columns=[str(int(i)) for i in list(FS_index)])
predict_label_SVM = int(Predict_model_SVM.predict(FS_x_data))

hit_atom_list = []
hit_bond_list = []

for info_key, info_value in shap_bitVec_info.items():
    if info_key == shap_index:
        radius = info_value[0][1]
        rank_atom = info_value[0][0]
        amap = {}
        env = Chem.FindAtomEnvironmentOfRadiusN(shap_mol, radius, rank_atom)
        submol = Chem.PathToSubmol(shap_mol, env, atomMap=amap)
        subsmiles = Chem.MolToSmiles(submol, canonical=True)

        if shap_mol.HasSubstructMatch(submol):
            # print('%d smiles contain substructure' % (mol_ind))
            hit_atom_list, hit_bond_list = mol_atom_bond(shap_mol, submol)

# plot molecule
legend = 'Pre_SVM_label:%d' % (predict_label_SVM) + '\n' + \
         '%s' % (shap_ind)

if len(hit_atom_list) != 0:
    img = Draw.MolToImage(shap_mol,
                          size=(sub_img_size_x, sub_img_size_y),
                          highlightAtoms=hit_atom_list,
                          highlightBonds=hit_bond_list, legend=None)
else:
    img = Draw.MolToImage(shap_mol,
                          size=(sub_img_size_x, sub_img_size_y),
                          legend=None)

draw = ImageDraw.Draw(img)  # 画图
font = ImageFont.truetype('times.ttf', 30)
draw.text((sub_img_size_x / 2 - 100, sub_img_size_y - 160), legend, font=font, fill=(0, 0, 0))
img.save(os.path.join(save_path, 'molecules_compound_%s.png' % (shap_ind)), dpi=(1000, 1000))

# load explanation model
load_inter_model_SVM = './data/Shap_SVM_RFE_Results/shap_SVM_model.pkl'
inter_model_SVM = joblib.load(load_inter_model_SVM)

inter_values_SVM = inter_model_SVM.shap_values(FS_x_data)

# shap force:SVM
shap.force_plot(inter_model_SVM.expected_value,
                inter_values_SVM,
                FS_x_data_df,
                figsize=(15, 2),
                matplotlib=True,
                show=False,
                contribution_threshold=0.07)  # contribution_threshold=0.035控制不同显示值
plt.savefig(os.path.join(save_path, 'shap_SVM_%s.png' % (shap_ind)), dpi=600,
            bbox_inches='tight')
plt.close()


