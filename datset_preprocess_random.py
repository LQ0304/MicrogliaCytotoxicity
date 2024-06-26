'''
在随机划分数据集的条件下，得到对应的分子表征
Avalon、ECFP、FCFP、rdkit、MACCS
Chempy
LV_LSTM、LV_GRU、LV_DMPNN
'''

from utils_fingerprint import *
import os
import numpy as np
import pandas as pd
import glob

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def save(save_path, s, x_data, y_data, train_len, overall_len, rank):
    np.save(os.path.join(save_path, s + '_' + 'train_x%d.npy' % rank), x_data[0:train_len])
    np.save(os.path.join(save_path, s + '_' + 'train_y%d.npy' % rank), y_data[0:train_len])
    np.save(os.path.join(save_path, s + '_' + 'test_x%d.npy' % rank), x_data[train_len:overall_len])
    np.save(os.path.join(save_path, s + '_' + 'test_y%d.npy' % rank), y_data[train_len:overall_len])


def save_DL(save_path, s, train_x_data, train_y_data, test_x_data, test_y_data, rank):
    np.save(os.path.join(save_path, s + '_' + 'train_x%d.npy' % rank), train_x_data)
    np.save(os.path.join(save_path, s + '_' + 'train_y%d.npy' % rank), train_y_data)
    np.save(os.path.join(save_path, s + '_' + 'test_x%d.npy' % rank), test_x_data)
    np.save(os.path.join(save_path, s + '_' + 'test_y%d.npy' % rank), test_y_data)


save_path = './data/save_data_random/'
os.makedirs(save_path, exist_ok=True)

dataset_path = './data/Random_Train_Test_dataset/'

# load data
for rank in range(0, 10):
    with open(os.path.join(dataset_path, 'random_train_dataset%d.txt' % rank), 'r') as f:
        data_line1 = f.readlines()
    print('train data contain:%d' % (len(data_line1)))

    # load data
    with open(os.path.join(dataset_path, 'random_test_dataset%d.txt' % rank), 'r') as f:
        data_line2 = f.readlines()
    print('test data contain:%d' % (len(data_line2)))

    data_line = data_line1 + data_line2

    mol_list = []
    label_list = []
    for data in data_line:
        smiles, label = data.strip('\n').split(' ')
        mol = Chem.MolFromSmiles(smiles)
        mol_list.append(mol)
        label = int(label)
        label_list.append(label)
    index = np.where(np.array(label_list) == 1)
    print('Active contain: %d' % (index[0].shape[0]))
    print('Non active contain: %d' % (len(mol_list) - (index[0].shape[0])))

    '''6种分子表征形式：不同分子描述符'''
    repre_list = ['Avalon', 'ECFP', 'FCFP', 'Rdkit', 'MACCS', 'Chemphy', 'LSTM', 'GRU', 'MPNN']

    s = repre_list[0]
    _, fp_bit_list_Avalon = mol_to_Avalon(mol_list, 1024)  # Avalon Fingerprint
    x_data_Avalon = np.array(fp_bit_list_Avalon)
    y_data_Avalon = np.array(label_list)
    save_path_Avalon = os.path.join(save_path, s)
    os.makedirs(save_path_Avalon, exist_ok=True)
    save(save_path_Avalon, s, x_data_Avalon, y_data_Avalon,
         len(data_line1), len(data_line), rank)

    s = repre_list[1]
    _, fp_bit_list_ECFP4 = mol_to_ecfp4(mol_list, 1024)
    x_data_ECFP4 = np.array(fp_bit_list_ECFP4)
    y_data_ECFP4 = np.array(label_list)
    save_path_ECFP4 = os.path.join(save_path, s)
    os.makedirs(save_path_ECFP4, exist_ok=True)
    save(save_path_ECFP4, s, x_data_ECFP4, y_data_ECFP4,
         len(data_line1), len(data_line), rank)

    s = repre_list[2]
    _, fp_bit_list_FPFC4 = mol_to_fcfp4(mol_list, 1024)
    x_data_FPFC4 = np.array(fp_bit_list_FPFC4)
    y_data_FPFC4 = np.array(label_list)
    save_path_FCFP4 = os.path.join(save_path, s)
    os.makedirs(save_path_FCFP4, exist_ok=True)
    save(save_path_FCFP4, s, x_data_FPFC4, y_data_FPFC4,
         len(data_line1), len(data_line), rank)

    s = repre_list[3]
    _, fp_bit_list_rdkit = mol_to_fp(mol_list, 1024)
    x_data_rdkit = np.array(fp_bit_list_rdkit)
    y_data_rdkit = np.array(label_list)
    save_path_rdkit = os.path.join(save_path, s)
    os.makedirs(save_path_rdkit, exist_ok=True)
    save(save_path_rdkit, s, x_data_rdkit, y_data_rdkit,
         len(data_line1), len(data_line), rank)

    s = repre_list[4]
    _, fp_bit_list_maccs = mol_to_maccs(mol_list)
    x_data_maccs = np.array(fp_bit_list_maccs)
    y_data_maccs = np.array(label_list)
    save_path_macss = os.path.join(save_path, s)
    os.makedirs(save_path_macss, exist_ok=True)
    save(save_path_macss, s, x_data_maccs, y_data_maccs,
         len(data_line1), len(data_line), rank)

    s = repre_list[5]
    fp_bit_chemphy = mol_to_physicochemical(mol_list)
    x_data_chemphy = np.array(fp_bit_chemphy)
    y_data_chemphy = np.array(label_list)
    save_path_chemphy = os.path.join(save_path, s)
    os.makedirs(save_path_chemphy, exist_ok=True)
    save(save_path_chemphy, s, x_data_chemphy, y_data_chemphy,
         len(data_line1), len(data_line), rank)

    s = repre_list[6]
    train_data_path = './data/Random_Train_Test_dataset_CSV/random_train_dataset%d' % rank
    train_data = pd.read_csv(train_data_path)
    fp_bit_LSTM_train = path_to_lstm(train_data_path)
    x_data_train_LSTM = np.array(fp_bit_LSTM_train)
    train_y_LSTM = np.array(train_data['Labels'].values)

    test_data_path = 'data/Random_Train_Test_dataset_CSV/random_test_dataset%d' % rank
    test_data = pd.read_csv(test_data_path)
    fp_bit_LSTM_test = path_to_lstm(test_data_path)
    x_data_test_LSTM = np.array(fp_bit_LSTM_test)
    test_y_LSTM = np.array(test_data['Labels'].values)

    save_path_LSTM = os.path.join(save_path, s)
    os.makedirs(save_path_LSTM, exist_ok=True)

    save_DL(save_path_LSTM, s, x_data_train_LSTM, train_y_LSTM, x_data_test_LSTM, test_y_LSTM, rank)

    s = repre_list[7]
    train_data_path = './data/Random_Train_Test_dataset_CSV/random_train_dataset%d' % rank
    train_data = pd.read_csv(train_data_path)
    fp_bit_GRU_train = path_to_gru(train_data_path)
    x_data_train_GRU = np.array(fp_bit_GRU_train)
    train_y_GRU = np.array(train_data['Labels'].values)

    test_data_path = 'data/Random_Train_Test_dataset_CSV/random_test_dataset%d' % rank
    test_data = pd.read_csv(test_data_path)
    fp_bit_GRU_test = path_to_gru(test_data_path)
    x_data_test_GRU = np.array(fp_bit_GRU_test)
    test_y_GRU = np.array(test_data['Labels'].values)

    save_path_GRU = os.path.join(save_path, s)
    os.makedirs(save_path_GRU, exist_ok=True)

    save_DL(save_path_GRU, s, x_data_train_GRU, train_y_GRU, x_data_test_GRU, test_y_GRU, rank)

    s = repre_list[8]
    train_data_path = './data/Random_Train_Test_dataset_CSV/random_train_dataset%d' % rank
    train_data = pd.read_csv(train_data_path)
    fp_bit_MPNN_train = path_to_mpnn(train_data_path)
    x_data_train_MPNN = np.array(fp_bit_MPNN_train)
    train_y_MPNN = np.array(train_data['Labels'].values)

    test_data_path = './data/Random_Train_Test_dataset_CSV/random_test_dataset%d' % rank
    test_data = pd.read_csv(test_data_path)
    fp_bit_MPNN_test = path_to_mpnn(test_data_path)
    x_data_test_MPNN = np.array(fp_bit_MPNN_test)
    test_y_MPNN = np.array(test_data['Labels'].values)

    save_path_MPNN = os.path.join(save_path, s)
    os.makedirs(save_path_MPNN, exist_ok=True)

    save_DL(save_path_MPNN, s, x_data_train_MPNN, train_y_MPNN, x_data_test_MPNN, test_y_MPNN, rank)
