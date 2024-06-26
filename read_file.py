import pandas as pd
import numpy as np
import os


def get_amount(path):
    df = pd.read_excel(path, usecols=[1, 4])
    data_line = df.values
    data_line = np.array([x[0] for x in data_line if x[1] == 0])
    amount_one = len(np.where(data_line == 1)[0])
    amount_zero = len(np.where(data_line == 0)[0])
    return amount_zero, amount_one


def get_values(path):
    df = pd.read_excel(path)
    data_line = df.values
    smiles_list = []
    label_list = []
    ts_list = []
    bm_ts_list = []
    bit_flag_list = []
    flag_list = []
    for data in data_line:
        smiles = data[0]
        label = int(data[1])
        ts = data[2]
        bm_ts = data[3]
        bit_flag = data[4]
        flag = data[5]
        if bit_flag == 0:
            smiles_list.append(smiles)
            label_list.append(label)
            ts_list.append(ts)
            bm_ts_list.append(bm_ts)
            bit_flag_list.append(bit_flag)
            flag_list.append(flag)
    return smiles_list, label_list, ts_list, bm_ts_list, bit_flag_list, flag_list


file_list = ['One_step', 'Two_step', 'Further_Two_step']
total_zero = 0
total_one = 0
Smiles_list = []
Label_list = []
Ts_list = []
BM_Ts_list = []
Bit_flag_list = []
Flag_list = []

for file_name in file_list:
    load_path = r'./%s' % (file_name)
    type_zero = 0
    type_one = 0
    for root, _, files in os.walk(load_path):
        for file in files:
            if os.path.splitext(file)[-1] == '.xlsx':
                excel_path = os.path.join(root, file)
                a_zero, a_one = get_amount(excel_path)
                smiles_list, label_list, ts_list, bm_ts_list, bit_flag_list, flag_list = get_values(excel_path)

                for i in range(len(smiles_list)):
                    smiles = smiles_list[i]
                    if smiles not in Smiles_list:
                        Smiles_list.append(smiles)
                        Label_list.append(label_list[i])
                        Ts_list.append(ts_list[i])
                        BM_Ts_list.append(bm_ts_list[i])
                        Bit_flag_list.append(bit_flag_list[i])
                        Flag_list.append(flag_list[i])
                type_zero += a_zero
                type_one += a_one
    total_one += type_one
    total_zero += type_zero
    print('%s contain 0: %d, Amount of 1: %d' % (file_name, type_zero, type_one))
print('Amount 0: %d, Amount of 1: %d' % (total_zero, total_one))

save_data = {'Smiles': Smiles_list,
             'Pre_cytotoxicity': Label_list,
             'Ts': Ts_list,
             'BM_Ts': BM_Ts_list,
             'Bit_flag': Bit_flag_list,
             'Flag': Flag_list}
save_data_df = pd.DataFrame(save_data)
save_data_df.to_excel('./data/Modification_BM_data.xlsx', index=False)

print('Remove duplicate contain %d , among contain 1: %d, 0:%d'
      % (len(Smiles_list),
         len(np.where(np.array(Label_list) == 1)[0]),
         len(np.where(np.array(Label_list) == 0)[0])))
