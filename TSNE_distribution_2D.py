from utils_fingerprint import mol_to_ecfp4
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib as mpl
import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'

save_path = './data/BM_pictures'
os.makedirs(save_path, exist_ok=True)

test_ind = 7
for i in range(test_ind, test_ind + 1):
    with open('./data/BM_Train_Test_dataset/BM_train_dataset%d.txt' % i, 'r') as f:
        dataline1 = f.readlines()

    Train_Smiles_list = []
    Train_fp_list = []
    Train_label_list = []
    for line in dataline1:
        line = line.strip('\n')
        smiles, name, label = line.split('\t')
        mol = Chem.MolFromSmiles(smiles)
        Train_Smiles_list.append(smiles)
        mol_graph_BM = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))  # mol_graph_BM
        _, train_fp_ecfp4 = mol_to_ecfp4(mol_graph_BM, 1024)
        Train_fp_list.append(train_fp_ecfp4)
        Train_label_list.append(int(label))
    print('Train dataset contain:%d' % (len(Train_Smiles_list)))
    print('Train dataset contain non-tox:%d' % (len((np.where(np.array(Train_label_list) == 1))[0])))
    print('Train dataset contain tox:%d' % (len((np.where(np.array(Train_label_list) == 0))[0])))

    with open('./data/BM_Train_Test_dataset/BM_test_dataset%d.txt' % i, 'r') as f:
        dataline2 = f.readlines()

    Test_Smiles_list = []
    Test_fp_list = []
    Test_label_list = []
    for data in dataline2:
        datas = data.strip('\n')
        smiles, name, label = datas.split('\t')
        mol = Chem.MolFromSmiles(smiles)
        Test_Smiles_list.append(smiles)
        mol_graph_BM = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
        _, test_fp_ecfp4 = mol_to_ecfp4(mol_graph_BM, 1024)
        Test_fp_list.append(test_fp_ecfp4)
        Test_label_list.append(int(label))
    print('Test dataset contain:%d' % (len(Test_fp_list)))
    print('Test dataset contain non-tox:%d' % (len((np.where(np.array(Test_label_list) == 1))[0])))
    print('Test dataset contain tox:%d' % (len((np.where(np.array(Test_label_list) == 0))[0])))

    overall = Train_fp_list + Test_fp_list
    # T-SNE analysis
    tsne = TSNE(n_components=2, random_state=222)
    component_tsne_data = tsne.fit_transform(overall)
    # component_tsne_Test = tsne.fit_transform(Test_fp_list)
    component_tsne_Train = component_tsne_data[0:len(dataline1), :]
    component_tsne_Test = component_tsne_data[len(dataline1):len(overall), :]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    ax.scatter(component_tsne_Train[:, 0], component_tsne_Train[:, 1],
               s=100, marker='o', alpha=1)
    ax.scatter(component_tsne_Test[:, 0], component_tsne_Test[:, 1],
               s=100, marker='o', alpha=1)
    ax.set_xlabel('Component 1', fontsize=14)
    ax.set_ylabel('Component 2', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.title('The T-SNE distribution of murcko generic scaffold', fontsize=16)
    # ax.set_zlabel('Component 3', fontsize=24)
    plt.legend(['${Train_{dataset}}$', '${Test_{dataset}}$'], fontsize=12)  # 742, 653, 236
    # plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    fig.savefig(os.path.join(save_path, 'TSNE_analysis_%d_dataset_2D.png' % i), dpi=1000)

# # T-SNE analysis 3D
# tsne = TSNE(n_components=3, random_state=0)
# component_tsne_data = tsne.fit_transform(overall)
# component_tsne_Train = component_tsne_data[0:len(dataline1), :]
# component_tsne_Test = component_tsne_data[len(dataline1):len(overall), :]
#
# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(component_tsne_Train[:, 0], component_tsne_Train[:, 1], component_tsne_Train[:, 2],
#            s=150, marker='o', alpha=0.8)
# ax.scatter(component_tsne_Test[:, 0], component_tsne_Test[:, 1], component_tsne_Test[:, 2],
#            s=150, marker='o', alpha=0.8)
# ax.set_xlabel('Component 1', fontsize=24)
# ax.set_ylabel('Component 2', fontsize=24)
# ax.set_zlabel('Component 3', fontsize=24)
# plt.legend(['${Train_{dataset}}$', '${Test_{dataset}}$'], fontsize=18)  # 742, 653, 236
# fig.savefig(os.path.join(save_path, 'TSNE_analysis_dataset_3D.png'), dpi=200)
