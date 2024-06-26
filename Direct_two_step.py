from rdkit.Chem import Draw
from Valid_actions_basis import Molecule
import os
from rdkit import Chem
import matplotlib as mpl
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import FastFindRings
import numpy as np
from rdkit import DataStructs
import joblib
import pandas as pd
from PIL import ImageDraw, ImageFont
from rdkit.Chem.Scaffolds import MurckoScaffold

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 14


def mol_to_ecfp4_bit(mol, fpszie):
    BitVec_Info = {}
    mol.UpdatePropertyCache()
    FastFindRings(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fpszie, bitInfo=BitVec_Info)
    fp_bit = np.zeros(len(fp))
    DataStructs.ConvertToNumpyArray(fp, fp_bit)
    return fp, fp_bit, BitVec_Info


def mol_to_ecfp4(mol_list, fpszie):
    if isinstance(mol_list, list):
        print("mol_list is list")
        fp_bit_list = []
        fp_list = []
        for mol in mol_list:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fpszie)
                fp_bit = np.zeros(len(fp))
                DataStructs.ConvertToNumpyArray(fp, fp_bit)
            except ValueError as e:
                print(e)
                fp = [np.nan]
                fp_bit = [np.nan]
            fp_bit_list.append(fp_bit)
            fp_list.append(fp)
    else:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol_list, 2, nBits=fpszie)
            fp_bit_list = np.zeros(len(fp))
            DataStructs.ConvertToNumpyArray(fp, fp_bit_list)
            fp_list = fp
        except ValueError as e:
            print(e)
            fp_list = [np.nan]
            fp_bit_list = [np.nan]
    return fp_list, fp_bit_list


def mol_atom_bond(mol, patt):
    hit_atom = mol.GetSubstructMatch(patt)
    hit_bond = []
    for bond in patt.GetBonds():
        aid1 = hit_atom[bond.GetBeginAtomIdx()]
        aid2 = hit_atom[bond.GetEndAtomIdx()]
        hit_bond.append(mol.GetBondBetweenAtoms(aid1, aid2).GetIdx())
    return list(hit_atom), hit_bond


def all_valid_states_actions(observation, hit_atom, hit_bond):
    add_atom_types = ['C', 'N', 'O']
    # init_mol = Chem.MolFromSmiles(observation)  # 初始分子为mol形式
    ft_molecule = Molecule(atom_types=add_atom_types, init_mol=observation, max_steps=10,
                           allow_atom_list=hit_atom, allow_bond_list=hit_bond)
    ft_molecule.initialize()
    states_smiles = list(ft_molecule._valid_states)
    Actions = list(ft_molecule._valid_actions)
    Atoms_list = ft_molecule._valid_atom_list
    Bonds_list = ft_molecule._valid_bond_list
    States_mol = ft_molecule._valid_mol_list
    return states_smiles, Actions, Atoms_list, Bonds_list, States_mol


def tox_predict(fp, model_path):
    # 将SMILES转换为分子指纹
    SVM_model = joblib.load(model_path)
    pre_tox = int(SVM_model.predict(fp))
    return pre_tox


# Toxicity模型加载，对应index
tox_model_path = './data/FS_BM_SVM_Results/ECFP_RFE/FS_BM_SVM_ECFP_RFE7.pkl'
tox_index = np.load('./data/FS_RFE/ECFP_RFE_index.npy')
print('Toxicity_Model contain: %d features' % (len(tox_index)))

# known data BM,calculate
smiles_dict_BM = joblib.load('./data/sort_smiles_BM_dict.pkl')
BM_mol_list = []
for BM_smiles, item in smiles_dict_BM:
    BM_mol = Chem.MolFromSmiles(BM_smiles)
    BM_mol_list.append(BM_mol)
BM_fp_list, _ = mol_to_ecfp4(BM_mol_list, 2048)  # 训练集的BM，用于计算骨架相似度

feature = 101
sub_img_size_x = 500
sub_img_size_y = 700
font = ImageFont.truetype('times.ttf', 30)

rank_list = [49, 68, 69, 260]

for rank in rank_list:
    print('reprocess_%d_dataset' % rank)
    load_path = './Two_step/Generate_molecule_compound-%d/' % (rank)
    df = pd.read_excel(os.path.join(load_path, 'compound-%d_generate_smiles.xlsx' % rank))
    data_line = df.values

    for s in range(len(data_line)):
        data = data_line[s]
        Tox_label = data[1]
        Bit_flag = data[4]

        parent_smiles = data[0]
        parent_mol = Chem.MolFromSmiles(parent_smiles)

        parent_fp, _ = mol_to_ecfp4(parent_mol, 2048)  # 训练集的BM，用于计算相似度

        _, _, bitVec_info = mol_to_ecfp4_bit(parent_mol, 1024)  # 用于得到对应bit信息
        hit_atom_list = []
        hit_bond_list = []
        submol = None
        for info_key, info_value in bitVec_info.items():
            if info_key == feature:
                radius = info_value[0][1]
                rank_atom = info_value[0][0]
                amap = {}
                env = Chem.FindAtomEnvironmentOfRadiusN(parent_mol, radius, rank_atom)
                submol = Chem.PathToSubmol(parent_mol, env, atomMap=amap)
                if parent_mol.HasSubstructMatch(submol):
                    hit_atom_list, hit_bond_list = mol_atom_bond(parent_mol, submol)

                    path = './Further_Two_step/Generate_molecule_compound-%d/' % (rank)
                    save_path = os.path.join(path, 'further_%d' % (s))
                    os.makedirs(save_path, exist_ok=True)

                    states_smiles, actions, atom_list, bond_list, states_mol = all_valid_states_actions(parent_mol,
                                                                                                        hit_atom_list,
                                                                                                        hit_bond_list)

                    Pre_tox_list = []
                    Ts_list = []
                    Bit_flag_list = []
                    Flag_list = []
                    BM_sim_list = []
                    for i in range(len(states_smiles)):
                        smiles = states_smiles[i]
                        mol = states_mol[i]
                        act = actions[i]
                        generate_fp, _ = mol_to_ecfp4(mol, 2048)

                        gen_atom_BM = MurckoScaffold.GetScaffoldForMol(mol)
                        gen_graph_BM = MurckoScaffold.MakeScaffoldGeneric(gen_atom_BM)
                        gen_graph_BM_fp, _ = mol_to_ecfp4(gen_graph_BM, 2048)

                        # calculate Ts tanimoto
                        Ts = DataStructs.TanimotoSimilarity(parent_fp, generate_fp)
                        sim_Ts = float('%.2f' % (Ts))
                        Ts_list.append(sim_Ts)

                        hit_atom_t1_list = atom_list[i]
                        hit_bond_t1_list = bond_list[i]

                        # is or not contain submol
                        if mol.HasSubstructMatch(submol):
                            flag = 1
                        else:
                            flag = 0
                        Flag_list.append(flag)

                        _, gen_fp_bit = mol_to_ecfp4(mol, 1024)

                        # ecfp4_bit is or not contain #101
                        if int(gen_fp_bit[feature]) == 0:
                            bit_flag = 0
                        else:
                            bit_flag = 1
                        Bit_flag_list.append(bit_flag)
                        fp_bit_gen = gen_fp_bit.reshape(1, -1)[:, tox_index]

                        # calculate BM tanimoto
                        sim_bm_list = []
                        for orgin_BM_fp in BM_fp_list:
                            sim = DataStructs.TanimotoSimilarity(gen_graph_BM_fp, orgin_BM_fp)
                            sim_bm_list.append(sim)
                        max_sim = round(max(sim_bm_list), 3)
                        BM_sim_list.append(max_sim)

                        if max_sim >= 0.5:
                            Pre_tox_value = tox_predict(fp_bit_gen, tox_model_path)
                        else:
                            Pre_tox_value = 0
                        Pre_tox_list.append(Pre_tox_value)

                        legend = 'Pre_SVM_label:' + str(Pre_tox_value) + '\n' + \
                                 'Action:' + str(act) + '\n'
                        # 'Flag:' + str(flag) + '\n' + \
                        # 'Bit_flag:' + str(bit_flag)

                        img = Draw.MolToImage(mol,
                                              size=(sub_img_size_x, sub_img_size_y),
                                              highlightAtoms=hit_atom_t1_list,
                                              highlightBonds=hit_bond_t1_list, legend=None)
                        draw = ImageDraw.Draw(img)  # 画图
                        draw.text((sub_img_size_x / 2 - 100, sub_img_size_y - 160), legend, font=font, fill=(0, 0, 0))
                        img.save(os.path.join(save_path, str(i) + '.png'), dpi=(1000, 1000))
                        img.close()
                    print('%d_States contain:%d' % (s, len(states_smiles)))

                    data = {'smiles': states_smiles,
                            'Tox_label': Pre_tox_list,
                            'Ts': Ts_list,
                            'BM_Ts': BM_sim_list,
                            'Bit_Flag': Bit_flag_list,
                            'Flag': Flag_list}
                    df = pd.DataFrame(data)
                    df.to_excel(os.path.join(path, 'compound-%d_generate_smiles%d.xlsx' % (rank, s)),
                                index=False)
