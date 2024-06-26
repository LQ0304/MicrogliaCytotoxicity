import numpy as np
from rdkit import Chem
# from whales_descriptors import do_whales
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, DataStructs, MACCSkeys, BRICS, Recap, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd


# # WHALES分子描述符
# def mol_to_WHALES(mol_list):
#     WHALES_descriptors, labels = do_whales.main(mol_list,
#                                                 charge_threshold=0,
#                                                 do_charge=True,
#                                                 property_name='')
#     return WHALES_descriptors, labels


def mol_to_Avalon(mol_list, fpszie):
    if isinstance(mol_list, list):
        print("mol_list is list")
        fp_bit_list = []
        fp_list = []
        for mol in mol_list:
            try:
                fp = pyAvalonTools.GetAvalonFP(mol, nBits=fpszie)
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
            fp = pyAvalonTools.GetAvalonFP(mol_list, nBits=fpszie)
            fp_bit_list = np.zeros(len(fp))
            DataStructs.ConvertToNumpyArray(fp, fp_bit_list)
            fp_list = fp
        except ValueError as e:
            print(e)
            fp_list = [np.nan]
            fp_bit_list = [np.nan]
    return fp_list, fp_bit_list


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


def mol_to_fcfp4(mol_list, fpszie):
    if isinstance(mol_list, list):
        print("mol_list is list")
        fp_bit_list = []
        fp_list = []
        for mol in mol_list:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fpszie, useFeatures=True)
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
            fp = AllChem.GetMorganFingerprintAsBitVect(mol_list, 2, nBits=fpszie, useFeatures=True)
            fp_bit_list = np.zeros(len(fp))
            DataStructs.ConvertToNumpyArray(fp, fp_bit_list)
            fp_list = fp
        except ValueError as e:
            print(e)
            fp_list = [np.nan]
            fp_bit_list = [np.nan]
    return fp_list, fp_bit_list


def mol_to_maccs(mol_list):
    if isinstance(mol_list, list):
        print("mol_list is list")
        fp_bit_list = []
        fp_list = []
        for mol in mol_list:
            try:
                fp = MACCSkeys.GenMACCSKeys(mol)
                fp_bit = np.zeros(len(fp))
                DataStructs.ConvertToNumpyArray(fp, fp_bit)
            except ValueError as e:
                print(e)
                fp = [np.nan]
                fp_bit = [np.nan]
            fp_list.append(fp)
            fp_bit_list.append(fp_bit)
    else:
        try:
            fp = MACCSkeys.GenMACCSKeys(mol_list)
            fingerprint = np.zeros(len(fp))
            DataStructs.ConvertToNumpyArray(fp, fingerprint)
            fp_list = fp
            fp_bit_list = fingerprint
        except ValueError as e:
            print(e)
            fp_list = [np.nan]
            fp_bit_list = [np.nan]
    return fp_list, fp_bit_list


def mol_to_fp(mol_list, fpszie):  # 与rdkit分子描述符相同
    if isinstance(mol_list, list):
        print("mol_list is list")
        fp_bit_list = []
        fp_list = []
        for mol in mol_list:
            try:
                fp = Chem.RDKFingerprint(mol, fpSize=fpszie)
                fp_bit = np.zeros(len(fp))
                DataStructs.ConvertToNumpyArray(fp, fp_bit)
            except ValueError as e:
                print(e)
                fp = [np.nan]
                fp_bit = [np.nan]
            fp_list.append(fp)
            fp_bit_list.append(fp_bit)

    else:
        try:
            fp = Chem.RDKFingerprint(mol_list, fpSize=fpszie)
            fp_bit_list = np.zeros(len(fp))
            DataStructs.ConvertToNumpyArray(fp, fp_bit_list)
            fp_list = fp
        except ValueError as e:
            print(e)
            fp_list = [np.nan]
            fp_bit_list = [np.nan]
    return fp_list, fp_bit_list


'''Brcis子结构表征：
（1）目标域分子得到的Brics片段库；
（2）建立Brics片段库字典；
（3）得到每个分子的索引码
（4）将索引码转化为one-hot码'''


def mol_to_brics(mol_list):
    # Brics分解分子
    del_list = []
    brics_list = []
    allfrags = set()
    for mol in mol_list:
        brics_frag = sorted(BRICS.BRICSDecompose(mol))
        if brics_frag == []:
            del_list.append(mol)
        else:
            brics_list.append(brics_frag)
            allfrags.update(brics_frag)
    print("一共有%d个独特的子结构" % (len(allfrags)))

    # 建立Brics字典,以供索引码构建
    allfrags_list = list(sorted(allfrags))
    brics_dict = {}
    for i in range(len(allfrags_list)):
        brics_dict[allfrags_list[i]] = i

    # 建立索引码
    index_list = []
    for frag_one in brics_list:
        frag_one_index = []
        for frag in frag_one:
            frag_one_index.append(brics_dict[frag])
        index_list.append(frag_one_index)

    # 将索引码转化为one-hot码
    fp_bit_list = []
    for one_index_list in index_list:
        init_list = [0] * len(brics_dict)
        for frag_index in one_index_list:
            init_list[frag_index] = 1
        fp_bit_list.append(init_list)
    return fp_bit_list, index_list, len(brics_dict), brics_dict


def mol_to_recap(mol_list, labels_list):
    if isinstance(mol_list, list):
        print("mol_list is list")
        No_decompose_list = []
        ALL_recap_list = []
        Recap_fragemnts = set()
        for i in range(len(mol_list)):
            mol = mol_list[i]
            label = labels_list[i]
            decomp = Recap.RecapDecompose(mol)
            leaves_mol = [leaf.mol for leaf in decomp.GetLeaves().values()]
            if leaves_mol != []:
                leaves_smiles = [Chem.MolToSmiles(mol) for mol in leaves_mol]
                if label == 1:
                    Recap_fragemnts.update(leaves_smiles)
            else:
                No_decompose_list.append(mol)  # 将不能分解的分子直接保留
                leaves_smiles = [Chem.MolToSmiles(mol)]
                # if label == 1:
                #     Recap_fragemnts.update(leaves_smiles)  # recap_dict中包含有活性的不能分解的分子
            ALL_recap_list.append(leaves_smiles)
    print('Racap_dict contain:%d' % (len(Recap_fragemnts)))
    print('Dont decompose contain:%d' % (len(No_decompose_list)))  # 121个分子
    print('All contain:%d' % (len(ALL_recap_list)))

    # 建立一个字典
    all_fragements_list = list(Recap_fragemnts)
    Recap_dict = {}
    for i in range(len(all_fragements_list)):
        Recap_dict[all_fragements_list[i]] = i

    # 建立索引码
    index_list = []
    for one_mol_list in ALL_recap_list:  # 一个分子
        mol_recap_list = []
        for frag in one_mol_list:  # 一个分子的一个片段
            if frag in Recap_dict.keys():  # 只包含Brics_dict键的保留
                mol_recap_list.append(Recap_dict[frag])  #
        index_list.append(mol_recap_list)

    # 将索引码变为one-hot码
    fp_bit_list = []
    for one_index_list in index_list:
        init_list = [0] * len(Recap_dict)
        for one_frag in one_index_list:
            init_list[one_frag] = 1
        fp_bit_list.append(init_list)
    return fp_bit_list, index_list, len(Recap_fragemnts), Recap_dict


# 生成200位分子性质描述符 输入为分子mol的列表 输出为分子描述符列表
def mol_to_physicochemical(mol_list):
    descs = [desc_name[0] for desc_name in Descriptors._descList]  # [17-24 feature produced]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    descriptors = pd.DataFrame([desc_calc.CalcDescriptors(mol) for mol in mol_list])
    x = descriptors.values
    return x


import torch.nn as nn
from Utils.build_vocab import WordVocab
from torch.nn import functional as F
import torch
import math
from Utils.utils import split
import random
import math
from torch.autograd import Variable
import re


# 生成RNN构建分子表征模型提取分子表征 输入为csv文件的路径，其中需要有一列名为'smiles'内容是SMILES，一列名为'Labels'内容是标签
def path_to_rnn(path):
    class Encoder(nn.Module):
        def __init__(self, input_size, embed_size, hidden_size,
                     n_layers=1, dropout=0.5):
            super(Encoder, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.embed_size = embed_size
            self.embed = nn.Embedding(input_size, embed_size)
            self.RNN = nn.RNN(embed_size, hidden_size, n_layers,
                              dropout=dropout, bidirectional=True)

        def forward(self, src, hidden=None):
            # src: (T,B)
            embedded = self.embed(src)  # (T,B,H)
            outputs, hidden = self.RNN(embedded, hidden)  # (T,B,2H), (2L,B,H)
            # sum bidirectional outputs
            outputs = (outputs[:, :, :self.hidden_size] +
                       outputs[:, :, self.hidden_size:])
            return outputs, hidden  # (T,B,H), (2L,B,H)

    class Attention(nn.Module):
        def __init__(self, hidden_size):
            super(Attention, self).__init__()
            self.hidden_size = hidden_size
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.rand(hidden_size))
            stdv = 1. / math.sqrt(self.v.size(0))
            self.v.data.uniform_(-stdv, stdv)

        def forward(self, hidden, encoder_outputs):
            timestep = encoder_outputs.size(0)
            h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
            encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
            attn_energies = self.score(h, encoder_outputs)
            return F.relu(attn_energies).unsqueeze(1)

        def score(self, hidden, encoder_outputs):
            # [B*T*2H]->[B*T*H]
            energy = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)))
            energy = energy.transpose(1, 2)  # [B*H*T]
            v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
            energy = torch.bmm(v, energy)  # [B*1*T]
            return energy.squeeze(1)  # [B*T]

    class Decoder(nn.Module):
        def __init__(self, embed_size, hidden_size, output_size,
                     n_layers=1, dropout=0.2):
            super(Decoder, self).__init__()
            self.embed_size = embed_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.n_layers = n_layers

            self.embed = nn.Embedding(output_size, embed_size)
            self.dropout = nn.Dropout(dropout, inplace=True)
            self.attention = Attention(hidden_size)
            self.RNN = nn.RNN(hidden_size + embed_size, hidden_size,
                              n_layers, dropout=dropout)
            self.out = nn.Linear(hidden_size * 2, output_size)

        def forward(self, input, last_hidden, encoder_outputs):
            # Get the embedding of the current input word (last output word)
            embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
            embedded = self.dropout(embedded)
            # Calculate attention weights and apply to encoder outputs
            attn_weights = self.attention(last_hidden[-1], encoder_outputs)
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
            context = context.transpose(0, 1)  # (1,B,N)
            # Combine embedded input word and attended context, run through RNN
            rnn_input = torch.cat([embedded, context], 2)
            output, hidden = self.RNN(rnn_input, last_hidden)
            output = output.squeeze(0)  # (1,B,N) -> (B,N)
            context = context.squeeze(0)
            output = self.out(torch.cat([output, context], 1))
            output = F.log_softmax(output, dim=1)  # log???
            return output, hidden, attn_weights

    class RNNSeq2Seq(nn.Module):
        def __init__(self, in_size, hidden_size, out_size, n_layers):
            super(RNNSeq2Seq, self).__init__()
            self.encoder = Encoder(in_size, hidden_size, hidden_size, n_layers)
            self.decoder = Decoder(hidden_size, hidden_size, out_size, n_layers)

        def forward(self, src, trg, teacher_forcing_ratio=0.5):  # (T,B)
            batch_size = src.size(1)
            max_len = trg.size(0)
            vocab_size = self.decoder.output_size
            outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()  # (T,B,V)
            encoder_output, hidden = self.encoder(src)  # (T,B,H), (2L,B,H)
            hidden = hidden[:self.decoder.n_layers]  # (L,B,H)
            output = Variable(trg.data[0, :])  # sos
            for t in range(1, max_len):
                output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)  # (B,V), (L,B,H)
                outputs[t] = output
                is_teacher = random.random() < teacher_forcing_ratio
                top1 = output.data.max(dim=1)[1]  # (B)
                output = Variable(trg.data[t] if is_teacher else top1).cuda()
            return outputs  # (T,B,V)

        def _encode(self, src):
            # src: (T,B)
            embedded = self.encoder.embed(src)  # (T,B,H)
            _, hidden = self.encoder.RNN(embedded, None)  # (T,B,2H), (2L,B,H)
            hidden = hidden.detach().numpy()
            return np.hstack([np.mean(hidden, axis=0)])
            # return np.hstack(hidden[:3])
            # out = np.hstack([np.mean(hidden[2:],axis=0)])
            # return out # (B,4H)

        def encode(self, src):
            # src: (T,B)
            batch_size = src.shape[1]
            if batch_size <= 100:
                return self._encode(src)
            else:  # Batch is too large to load
                print('There are {:d} molecules. It will take a little time.'.format(batch_size))
                st, ed = 0, 100
                out = self._encode(src[:, st:ed])  # (B,4H)
                while ed < batch_size:
                    st += 100
                    ed += 100
                    out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
                return out

    def get_inputs(sm):

        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(sm)]
        ids = [vocab.stoi.get(token, unk_index) for token in tokens]
        ids = [sos_index] + ids
        seg = [1] * len(ids)

        return ids, seg

    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)

    def bit2np(bitvector):
        bitstring = bitvector.ToBitString()
        intmap = map(int, bitstring)
        return np.array(list(intmap))

    def extract_spe(smiles, targets):
        x, X, y = [], [], []
        for sm, target in zip(smiles, targets):
            mol = Chem.MolFromSmiles(sm)
            if mol is None:
                print(sm)
                continue
            x_split = [split(sm)]
            xid, xseg = get_array(x_split)
            fp_spe = trfm.encode(torch.t(xid[:, 1:]))[:, :]
            x.append(sm)
            X.append(fp_spe[0])
            y.append(target)
        return x, np.array(X), np.array(y)

    df = pd.read_csv(path)
    print("data size：", df.shape)
    vocab = WordVocab.load_vocab('data/vocab.pkl')
    trfm = RNNSeq2Seq(in_size=len(vocab), hidden_size=256, out_size=len(vocab), n_layers=3)
    trfm.load_state_dict(
        torch.load('model/Representation/rnnseq2seq/rnn_4_20000.pkl', map_location=torch.device('cpu')))
    trfm.eval()
    print('Total parameters:', sum(p.numel() for p in trfm.parameters()))

    unk_index = 2
    sos_index = 1

    x_spe, X_spe, y_spe = extract_spe(df['smiles'].values, df['Labels'].values)  # 提取分子描述符
    x_spe_data = np.array(X_spe)
    return x_spe_data


# LSTM 构建分子表征模型提取分子表征 输入为csv文件的路径，其中需要有一列名为'smiles'内容是SMILES，一列名为'Labels'内容是标签
def path_to_lstm(path):
    class Encoder(nn.Module):
        def __init__(self, input_size, embed_size, hidden_size,
                     n_layers=1, dropout=0.5):
            super(Encoder, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.embed_size = embed_size
            self.embed = nn.Embedding(input_size, embed_size)
            self.LSTM = nn.LSTM(embed_size, hidden_size, n_layers,
                                dropout=dropout, bidirectional=True)

        def forward(self, src, hidden=None):
            # src: (T,B)
            embedded = self.embed(src)  # (T,B,H)
            outputs, hidden = self.LSTM(embedded, hidden)  # (T,B,2H), (2L,B,H)
            # sum bidirectional outputs
            outputs = (outputs[:, :, :self.hidden_size] +
                       outputs[:, :, self.hidden_size:])
            return outputs, hidden  # (T,B,H), (2L,B,H)

    class Attention(nn.Module):
        def __init__(self, hidden_size):
            super(Attention, self).__init__()
            self.hidden_size = hidden_size
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.rand(hidden_size))
            stdv = 1. / math.sqrt(self.v.size(0))
            self.v.data.uniform_(-stdv, stdv)

        def forward(self, hidden, encoder_outputs):
            timestep = encoder_outputs.size(0)
            h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
            encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
            attn_energies = self.score(h, encoder_outputs)
            return F.relu(attn_energies).unsqueeze(1)

        def score(self, hidden, encoder_outputs):
            # [B*T*2H]->[B*T*H]
            energy = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)))
            energy = energy.transpose(1, 2)  # [B*H*T]
            v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
            energy = torch.bmm(v, energy)  # [B*1*T]
            return energy.squeeze(1)  # [B*T]

    class Decoder(nn.Module):
        def __init__(self, embed_size, hidden_size, output_size,
                     n_layers=1, dropout=0.2):
            super(Decoder, self).__init__()
            self.embed_size = embed_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.n_layers = n_layers

            self.embed = nn.Embedding(output_size, embed_size)
            self.dropout = nn.Dropout(dropout, inplace=True)
            self.attention = Attention(hidden_size)
            self.LSTM = nn.LSTM(hidden_size + embed_size, hidden_size,
                                n_layers, dropout=dropout)
            self.out = nn.Linear(hidden_size * 2, output_size)

        def forward(self, input, last_hidden, encoder_outputs):
            # Get the embedding of the current input word (last output word)
            embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
            embedded = self.dropout(embedded)
            # Calculate attention weights and apply to encoder outputs
            attn_weights = self.attention(last_hidden[0][-1], encoder_outputs)
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
            context = context.transpose(0, 1)  # (1,B,N)
            # Combine embedded input word and attended context, run through RNN
            rnn_input = torch.cat([embedded, context], 2)
            output, hidden = self.LSTM(rnn_input, last_hidden)
            output = output.squeeze(0)  # (1,B,N) -> (B,N)
            context = context.squeeze(0)
            output = self.out(torch.cat([output, context], 1))
            output = F.log_softmax(output, dim=1)  # log???
            return output, hidden, attn_weights

    class RNNSeq2Seq(nn.Module):
        def __init__(self, in_size, hidden_size, out_size, n_layers):
            super(RNNSeq2Seq, self).__init__()
            self.encoder = Encoder(in_size, hidden_size, hidden_size, n_layers)
            self.decoder = Decoder(hidden_size, hidden_size, out_size, n_layers)

        def forward(self, src, trg, teacher_forcing_ratio=0.5):  # (T,B)
            batch_size = src.size(1)
            max_len = trg.size(0)
            vocab_size = self.decoder.output_size
            outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()  # (T,B,V)
            encoder_output, hidden = self.encoder(src)  # (T,B,H), (2L,B,H)
            hidden0 = hidden[0][:self.decoder.n_layers]
            hidden1 = hidden[1][:self.decoder.n_layers]  # (L,B,H)
            hidden = (hidden0, hidden1)
            output = Variable(trg.data[0, :])  # sos
            for t in range(1, max_len):
                output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)  # (B,V), (L,B,H)
                outputs[t] = output
                is_teacher = random.random() < teacher_forcing_ratio
                top1 = output.data.max(dim=1)[1]  # (B)
                output = Variable(trg.data[t] if is_teacher else top1).cuda()
            return outputs  # (T,B,V)

        def _encode(self, src):
            # src: (T,B)
            embedded = self.encoder.embed(src)  # (T,B,H)
            _, (hidden, ct) = self.encoder.LSTM(embedded, None)  # (T,B,2H), (2L,B,H)
            hidden = hidden.detach().numpy()
            out = np.hstack([np.mean(hidden, axis=0)])
            return out  # (B,4H)

        def encode(self, src):
            # src: (T,B)
            batch_size = src.shape[1]
            if batch_size <= 100:
                return self._encode(src)
            else:  # Batch is too large to load
                print('There are {:d} molecules. It will take a little time.'.format(batch_size))
                st, ed = 0, 100
                out = self._encode(src[:, st:ed])  # (B,4H)
                while ed < batch_size:
                    st += 100
                    ed += 100
                    out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
                return out

    # mask_index = 4
    def get_inputs(sm):
        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(sm)]
        ids = [vocab.stoi.get(token, unk_index) for token in tokens]
        ids = [sos_index] + ids
        seg = [1] * len(ids)
        return ids, seg

    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)

    def bit2np(bitvector):
        bitstring = bitvector.ToBitString()
        intmap = map(int, bitstring)
        return np.array(list(intmap))

    def extract_spe(smiles, targets):
        x, X, y = [], [], []
        for sm, target in zip(smiles, targets):
            mol = Chem.MolFromSmiles(sm)
            if mol is None:
                print(sm)
                continue
            x_split = [split(sm)]
            xid, xseg = get_array(x_split)
            fp_spe = trfm.encode(torch.t(xid[:, 1:]))[:, :]
            x.append(sm)
            X.append(fp_spe[0])
            y.append(target)
        return x, np.array(X), np.array(y)

    df = pd.read_csv(path)
    print("data size：", df.shape)

    vocab = WordVocab.load_vocab('data/vocab.pkl')
    trfm = RNNSeq2Seq(in_size=len(vocab), hidden_size=256, out_size=len(vocab), n_layers=3)
    trfm.load_state_dict(
        torch.load('model/Representation/lstmseq2seq/lstm_4_20000.pkl', map_location=torch.device('cpu')))
    trfm.eval()
    print('Total parameters:', sum(p.numel() for p in trfm.parameters()))

    unk_index = 2
    sos_index = 1

    x_spe, X_spe, y_spe = extract_spe(df['smiles'].values, df['Labels'].values)  # 提取分子描述符

    x_spe_data = np.array(X_spe)
    return x_spe_data


# GRU 构建分子表征模型提取分子表征 输入为csv文件的路径，其中需要有一列名为'smiles'内容是SMILES，一列名为'Labels'内容是标签
def path_to_gru(path):
    class Encoder(nn.Module):
        def __init__(self, input_size, embed_size, hidden_size,
                     n_layers=1, dropout=0.5):
            super(Encoder, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.embed_size = embed_size
            self.embed = nn.Embedding(input_size, embed_size)
            self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                              dropout=dropout, bidirectional=True)

        def forward(self, src, hidden=None):
            # src: (T,B)
            embedded = self.embed(src)  # (T,B,H)
            outputs, hidden = self.gru(embedded, hidden)  # (T,B,2H), (2L,B,H)
            # sum bidirectional outputs
            outputs = (outputs[:, :, :self.hidden_size] +
                       outputs[:, :, self.hidden_size:])
            return outputs, hidden  # (T,B,H), (2L,B,H)

    class Attention(nn.Module):
        def __init__(self, hidden_size):
            super(Attention, self).__init__()
            self.hidden_size = hidden_size
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.rand(hidden_size))
            stdv = 1. / math.sqrt(self.v.size(0))
            self.v.data.uniform_(-stdv, stdv)

        def forward(self, hidden, encoder_outputs):
            timestep = encoder_outputs.size(0)
            h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
            encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
            attn_energies = self.score(h, encoder_outputs)
            return F.relu(attn_energies).unsqueeze(1)

        def score(self, hidden, encoder_outputs):
            # [B*T*2H]->[B*T*H]
            energy = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)))
            energy = energy.transpose(1, 2)  # [B*H*T]
            v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
            energy = torch.bmm(v, energy)  # [B*1*T]
            return energy.squeeze(1)  # [B*T]

    class Decoder(nn.Module):
        def __init__(self, embed_size, hidden_size, output_size,
                     n_layers=1, dropout=0.2):
            super(Decoder, self).__init__()
            self.embed_size = embed_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.n_layers = n_layers

            self.embed = nn.Embedding(output_size, embed_size)
            self.dropout = nn.Dropout(dropout, inplace=True)
            self.attention = Attention(hidden_size)
            self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                              n_layers, dropout=dropout)
            self.out = nn.Linear(hidden_size * 2, output_size)

        def forward(self, input, last_hidden, encoder_outputs):
            # Get the embedding of the current input word (last output word)
            embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
            embedded = self.dropout(embedded)
            # Calculate attention weights and apply to encoder outputs
            attn_weights = self.attention(last_hidden[-1], encoder_outputs)
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
            context = context.transpose(0, 1)  # (1,B,N)
            # Combine embedded input word and attended context, run through RNN
            rnn_input = torch.cat([embedded, context], 2)
            output, hidden = self.gru(rnn_input, last_hidden)
            output = output.squeeze(0)  # (1,B,N) -> (B,N)
            context = context.squeeze(0)
            output = self.out(torch.cat([output, context], 1))
            output = F.log_softmax(output, dim=1)  # log???
            return output, hidden, attn_weights

    import random
    class RNNSeq2Seq(nn.Module):
        def __init__(self, in_size, hidden_size, out_size, n_layers):
            super(RNNSeq2Seq, self).__init__()
            self.encoder = Encoder(in_size, hidden_size, hidden_size, n_layers)
            self.decoder = Decoder(hidden_size, hidden_size, out_size, n_layers)

        def forward(self, src, trg, teacher_forcing_ratio=0.5):  # (T,B)
            batch_size = src.size(1)
            max_len = trg.size(0)
            vocab_size = self.decoder.output_size
            outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()  # (T,B,V)
            encoder_output, hidden = self.encoder(src)  # (T,B,H), (2L,B,H)
            hidden = hidden[:self.decoder.n_layers]  # (L,B,H)
            output = Variable(trg.data[0, :])  # sos
            for t in range(1, max_len):
                output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)  # (B,V), (L,B,H)
                outputs[t] = output
                is_teacher = random.random() < teacher_forcing_ratio
                top1 = output.data.max(dim=1)[1]  # (B)
                output = Variable(trg.data[t] if is_teacher else top1).cuda()
            return outputs  # (T,B,V)

        def _encode(self, src):
            # src: (T,B)
            embedded = self.encoder.embed(src)  # (T,B,H)
            _, hidden = self.encoder.gru(embedded, None)  # (T,B,2H), (2L,B,H)
            hidden = hidden.detach().numpy()
            return np.hstack([np.mean(hidden, axis=0)])
            # return np.hstack(hidden[2:])  # (B,4H)

        def encode(self, src):
            # src: (T,B)
            batch_size = src.shape[1]
            if batch_size <= 100:
                return self._encode(src)
            else:  # Batch is too large to load
                print('There are {:d} molecules. It will take a little time.'.format(batch_size))
                st, ed = 0, 100
                out = self._encode(src[:, st:ed])  # (B,4H)
                while ed < batch_size:
                    st += 100
                    ed += 100
                    out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
                return out

    df = pd.read_csv(path)
    print("data size：", df.shape)

    vocab = WordVocab.load_vocab('data/vocab.pkl')
    trfm = RNNSeq2Seq(in_size=len(vocab), hidden_size=256, out_size=len(vocab), n_layers=3)
    trfm.load_state_dict(
        torch.load('model/Representation/gruseq2seq/gru_4_20000.pkl', map_location=torch.device('cpu')))
    trfm.eval()
    print('Total parameters:', sum(p.numel() for p in trfm.parameters()))

    pad_index = 0
    unk_index = 2
    sos_index = 1

    def get_inputs(sm):

        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(sm)]
        ids = [vocab.stoi.get(token, unk_index) for token in tokens]
        ids = ids
        seg = [1] * len(ids)

        return ids, seg

    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)

    def bit2np(bitvector):
        bitstring = bitvector.ToBitString()
        intmap = map(int, bitstring)
        return np.array(list(intmap))

    def extract_spe(smiles, targets):
        x, X, y = [], [], []
        for sm, target in zip(smiles, targets):
            mol = Chem.MolFromSmiles(sm)
            if mol is None:
                print(sm)
                continue
            x_split = [split(sm)]
            xid, xseg = get_array(x_split)
            fp_spe = trfm.encode(torch.t(xid[:, 1:]))[:, :]
            x.append(sm)
            X.append(fp_spe[0])
            y.append(target)
        return x, np.array(X), np.array(y)

    x_spe, X_spe, y_spe = extract_spe(df['smiles'].values, df['Labels'].values)  # 提取分子描述符

    x_spe_data = np.array(X_spe)
    return x_spe_data


# 原始Transformer提取分子表征 输入为csv文件的路径，其中需要有一列名为'smiles'内容是SMILES，一列名为'Labels'内容是标签
def path_to_origninal(path):
    class PositionalEncoding(nn.Module):
        "Implement the PE function. No batch support?"

        def __init__(self, d_model, dropout, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            # Compute the positional encodings once in log space.
            pe = torch.zeros(max_len, d_model)  # (T,H)
            position = torch.arange(0., max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + Variable(self.pe[:, :x.size(1)],
                             requires_grad=False)
            return self.dropout(x)

    class TrfmSeq2seq(nn.Module):
        def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
            super(TrfmSeq2seq, self).__init__()
            self.in_size = in_size
            self.hidden_size = hidden_size
            # model_layer
            self.embed = nn.Embedding(in_size, hidden_size)
            self.pe = PositionalEncoding(hidden_size, dropout)
            self.trfm = nn.Transformer(d_model=hidden_size, nhead=4,
                                       num_encoder_layers=n_layers, num_decoder_layers=n_layers,
                                       dim_feedforward=hidden_size)
            self.out = nn.Linear(hidden_size, out_size)

        def forward(self, src):
            # src: (T,B)
            embedded = self.embed(src)  # (T,B,H)
            embedded = self.pe(embedded)  # (T,B,H)
            hidden = self.trfm(embedded, embedded)  # (T,B,H)
            out = self.out(hidden)  # (T,B,V)
            out = F.log_softmax(out, dim=2)  # (T,B,V)
            return out  # (T,B,V)

        def _encode(self, src):
            # src: (T,B)
            embedded = self.embed(src)  # (T,B,H)
            embedded = self.pe(embedded)  # (T,B,H)
            output = embedded
            for i in range(self.trfm.encoder.num_layers - 1):
                output = self.trfm.encoder.layers[i](output, None)  # (T,B,H)
            penul = output.detach().numpy()
            output = self.trfm.encoder.layers[-1](output, None)  # (T,B,H)
            if self.trfm.encoder.norm:
                output = self.trfm.encoder.norm(output)  # (T,B,H)
            output = output.detach().numpy()
            # mean, max, first*2
            return np.hstack(
                [np.mean(output, axis=0), np.max(output, axis=0), output[0, :, :], penul[0, :, :]])  # (B,4H)

        def encode(self, src):
            # src: (T,B)
            batch_size = src.shape[1]
            if batch_size <= 100:
                return self._encode(src)
            else:
                print('There are {:d} molecules. It will take a little time.'.format(batch_size))
                st, ed = 0, 100
                out = self._encode(src[:, st:ed])  # (B,4H)
                while ed < batch_size:
                    st += 100
                    ed += 100
                    out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
                return out

    df = pd.read_csv(path)
    print("data size：", df.shape)

    vocab = WordVocab.load_vocab('data/vocab.pkl')
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('model/Representation/trfm/trfm_4_9123.pkl', map_location=torch.device('cpu')))
    trfm.eval()
    print('Total parameters:', sum(p.numel() for p in trfm.parameters()))

    pad_index = 0
    unk_index = 2

    sos_index = 1

    def get_inputs(sm):

        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(sm)]
        ids = [vocab.stoi.get(token, unk_index) for token in tokens]
        ids = ids
        seg = [1] * len(ids)
        return ids, seg

    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)

    def bit2np(bitvector):
        bitstring = bitvector.ToBitString()
        intmap = map(int, bitstring)
        return np.array(list(intmap))

    def extract_spe(smiles, targets):
        x, X, y = [], [], []
        for sm, target in zip(smiles, targets):
            mol = Chem.MolFromSmiles(sm)
            if mol is None:
                print(sm)
                continue
            x_split = [split(sm)]
            xid, xseg = get_array(x_split)
            fp_spe = trfm.encode(torch.t(xid))[:, :256]
            x.append(sm)
            X.append(fp_spe[0])
            y.append(target)
        return x, np.array(X), np.array(y)

    x_spe, X_spe, y_spe = extract_spe(df['smiles'].values, df['Labels'].values)  # 提取分子描述符

    x_spe_data = np.array(X_spe)
    return x_spe_data


# 无掩码Transformer提取分子表征 输入为csv文件的路径，其中需要有一列名为'smiles'内容是SMILES，一列名为'Labels'内容是标签
def path_to_nomask(path):
    class PositionalEncoding(nn.Module):
        "Implement the PE function. No batch support?"

        def __init__(self, d_model, dropout, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            # Compute the positional encodings once in log space.
            pe = torch.zeros(max_len, d_model)  # (T,H)
            position = torch.arange(0., max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + Variable(self.pe[:, :x.size(1)],
                             requires_grad=False)
            return self.dropout(x)

    class TrfmSeq2seq(nn.Module):
        def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
            super(TrfmSeq2seq, self).__init__()
            self.in_size = in_size
            self.hidden_size = hidden_size
            # model_layer
            self.embed = nn.Embedding(in_size, hidden_size)
            self.pe = PositionalEncoding(hidden_size, dropout)
            self.trfm = nn.Transformer(d_model=hidden_size, nhead=4,
                                       num_encoder_layers=n_layers, num_decoder_layers=n_layers,
                                       dim_feedforward=hidden_size)
            self.out = nn.Linear(hidden_size, out_size)

        def forward(self, src):
            # src: (T,B)
            embedded = self.embed(src)  # (T,B,H)
            embedded = self.pe(embedded)  # (T,B,H)
            hidden = self.trfm(embedded, embedded)  # (T,B,H)
            out = self.out(hidden)  # (T,B,V)
            out = F.log_softmax(out, dim=2)  # (T,B,V)
            return out  # (T,B,V)

        def _encode(self, src):
            # src: (T,B)
            embedded = self.embed(src)  # (T,B,H)
            embedded = self.pe(embedded)  # (T,B,H)
            output = embedded
            for i in range(self.trfm.encoder.num_layers - 1):
                output = self.trfm.encoder.layers[i](output, None)  # (T,B,H)
            penul = output.detach().numpy()
            output = self.trfm.encoder.layers[-1](output, None)  # (T,B,H)
            if self.trfm.encoder.norm:
                output = self.trfm.encoder.norm(output)  # (T,B,H)
            output = output.detach().numpy()
            # mean, max, first*2
            return np.hstack(
                [np.mean(output, axis=0), np.max(output, axis=0), output[0, :, :], penul[0, :, :]])  # (B,4H)

        def encode(self, src):
            # src: (T,B)
            batch_size = src.shape[1]
            if batch_size <= 100:
                return self._encode(src)
            else:
                print('There are {:d} molecules. It will take a little time.'.format(batch_size))
                st, ed = 0, 100
                out = self._encode(src[:, st:ed])  # (B,4H)
                while ed < batch_size:
                    st += 100
                    ed += 100
                    out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
                return out

    df = pd.read_csv(path)
    print("data size：", df.shape)

    vocab = WordVocab.load_vocab('data/vocab.pkl')
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(
        torch.load('model/Representation/nomasktrfm/nomasktrfm_4_0.pkl', map_location=torch.device('cpu')))
    trfm.eval()
    print('Total parameters:', sum(p.numel() for p in trfm.parameters()))

    pad_index = 0
    unk_index = 2
    # eos_index = 2
    sos_index = 1

    # mask_index = 4
    def get_inputs(sm):

        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(sm)]
        ids = [vocab.stoi.get(token, unk_index) for token in tokens]
        ids = [sos_index] + ids
        seg = [1] * len(ids)
        return ids, seg

    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)

    def bit2np(bitvector):
        bitstring = bitvector.ToBitString()
        intmap = map(int, bitstring)
        return np.array(list(intmap))

    def extract_spe(smiles, targets):
        x, X, y = [], [], []
        for sm, target in zip(smiles, targets):
            mol = Chem.MolFromSmiles(sm)
            if mol is None:
                print(sm)
                continue
            x_split = [split(sm)]
            xid, xseg = get_array(x_split)
            fp_spe = trfm.encode(torch.t(xid))[:, :256]
            x.append(sm)
            X.append(fp_spe[0])
            y.append(target)
        return x, np.array(X), np.array(y)

    x_spe, X_spe, y_spe = extract_spe(df['smiles'].values, df['Labels'].values)  # 提取分子描述符

    x_spe_data = np.array(X_spe)
    return x_spe_data


# 交互掩码Transformer提取分子表征 输入为csv文件的路径，其中需要有一列名为'smiles'内容是SMILES，一列名为'Labels'内容是标签
def path_to_doublemask(path):
    def get_attn_subsequent_mask(seq):
        # seq = seq.t()
        attn_shape = [seq.size(0), seq.size(0)]
        # attn_shape: [batch_size, tgt_len, tgt_len]
        subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
        subsequence_mask = torch.from_numpy(subsequence_mask).bool()  # .byte()
        return subsequence_mask  # [tgt_len, tgt_len]

    def get_attn_pad_mask(seq):
        # eq(zero) is PAD token
        seq = seq.t()
        pad_attn_mask = seq.data.eq(0)  # batch_size x 1 x len_k, one is masking
        return pad_attn_mask  # batch_size x seq_len

    ## 11
    def get_dec_enc_attn_subsequent_mask(tgt, src):
        """
        seq: [batch_size, tgt_len]
        """
        attn_shape = [tgt.size(0), src.size(0)]
        # attn_shape: [batch_size, tgt_len, tgt_len]
        subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个三角矩阵
        # subsequence_mask = np.zeros(attn_shape) # 生成一个三角矩阵

        subsequence_mask = torch.from_numpy(subsequence_mask).bool()  # .byte()
        return subsequence_mask  # [tgt_len, src_len]

    # 位置编码
    class PositionalEncoding(nn.Module):
        "Implement the PE function. No batch support?"

        def __init__(self, d_model, dropout, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            # Compute the positional encodings once in log space.
            pe = torch.zeros(max_len, d_model)  # (T,H)
            position = torch.arange(0., max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + Variable(self.pe[:, :x.size(1)],
                             requires_grad=False)
            return self.dropout(x)

    class TrfmSeq2seq(nn.Module):
        def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
            super(TrfmSeq2seq, self).__init__()
            self.in_size = in_size
            self.hidden_size = hidden_size
            # model_layer
            self.embed = nn.Embedding(in_size, hidden_size)
            self.pe = PositionalEncoding(hidden_size, dropout)
            self.trfm = nn.Transformer(d_model=hidden_size, nhead=4,
                                       num_encoder_layers=n_layers, num_decoder_layers=n_layers,
                                       dim_feedforward=hidden_size)
            self.out = nn.Linear(hidden_size, out_size)

        def forward(self, src, tgt):
            # src: (T,B)
            src_key_padding_mask = get_attn_pad_mask(src).cuda()  # (B,S)
            tgt_mask = get_attn_subsequent_mask(tgt).cuda()  # (T,T)
            memory_mask = get_dec_enc_attn_subsequent_mask(tgt, src).cuda()  # (T,S)
            tgt_key_padding_mask = get_attn_pad_mask(tgt).cuda()  # (B,T)
            memory_key_padding_mask = src_key_padding_mask.cuda()  # (B,S)

            src_embedded = self.embed(src)  # (T,B,H)
            src_embedded = self.pe(src_embedded)  # (T,B,H)

            tgt_embedded = self.embed(tgt)  # (T,B,H)
            tgt_embedded = self.pe(tgt_embedded)  # (T,B,H)

            mem = self.trfm.encoder(src_embedded, src_key_padding_mask=src_key_padding_mask)
            hidden = self.trfm.decoder(tgt_embedded, mem, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask)

            out = self.out(hidden)  # (T,B,V)
            out = F.log_softmax(out, dim=2)  # (T,B,V)
            return out  # (T,B,V)

        def _encode(self, src):
            # src: (T,B)
            src_key_padding_mask = get_attn_pad_mask(src)  # .cuda()  # (B,S)
            embedded = self.embed(src)  # (T,B,H)
            embedded = self.pe(embedded)  # (T,B,H)
            output = embedded
            for i in range(self.trfm.encoder.num_layers - 1):
                output = self.trfm.encoder.layers[i](output, src_key_padding_mask=src_key_padding_mask)  # (T,B,H)
            penul = output.detach().numpy()
            output = self.trfm.encoder.layers[-1](output, src_key_padding_mask=src_key_padding_mask)  # (T,B,H)
            if self.trfm.encoder.norm:
                output = self.trfm.encoder.norm(output)  # (T,B,H)
            output = output.detach().numpy()
            # mean, max, first*2
            return np.hstack(
                [np.mean(output, axis=0), np.max(output, axis=0), output[0, :, :], penul[0, :, :]])  # (B,4H)

        def encode(self, src):
            # src: (T,B)
            batch_size = src.shape[1]
            if batch_size <= 100:
                return self._encode(src)
            else:
                print('There are {:d} molecules. It will take a little time.'.format(batch_size))
                st, ed = 0, 100
                out = self._encode(src[:, st:ed])  # (B,4H)
                while ed < batch_size:
                    st += 100
                    ed += 100
                    out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
                return out

    df = pd.read_csv(path)
    print("data size：", df.shape)

    vocab = WordVocab.load_vocab('data/vocab.pkl')

    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(
        torch.load('model/Representation/doublemasktrfm/doublemasktrfm_3_9123.pkl', map_location=torch.device('cpu')))
    trfm.eval()
    print('Total parameters:', sum(p.numel() for p in trfm.parameters()))

    pad_index = 0
    unk_index = 2
    sos_index = 1

    def get_inputs(sm):

        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(sm)]
        ids = [vocab.stoi.get(token, unk_index) for token in tokens]
        ids = ids
        seg = [1] * len(ids)

        return ids, seg

    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)

    def bit2np(bitvector):
        bitstring = bitvector.ToBitString()
        intmap = map(int, bitstring)
        return np.array(list(intmap))

    def extract_spe(smiles, targets):
        x, X, y = [], [], []
        for sm, target in zip(smiles, targets):
            mol = Chem.MolFromSmiles(sm)
            if mol is None:
                print(sm)
                continue
            x_split = [split(sm)]
            xid, xseg = get_array(x_split)
            fp_spe = trfm.encode(torch.t(xid))[:, :256]
            x.append(sm)
            X.append(fp_spe[0])
            y.append(target)
        return x, np.array(X), np.array(y)

    x_spe, X_spe, y_spe = extract_spe(df['smiles'].values, df['Labels'].values)  # 提取分子描述符

    x_spe_data = np.array(X_spe)
    return x_spe_data


from tqdm import trange
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data
from chemprop.utils import load_args, load_checkpoint, load_scalers
from chemprop.parsing import parse_predict_args


# mpnn提取分子表征 输入为csv文件的路径，其中需要第一列内容是SMILES，第二列内容是标签
def path_to_mpnn(path):
    args = parse_predict_args()
    args.gpu = 0
    args.checkpoint_paths = 'model/mpnn/model.pt'
    args.test_path = path
    print('Loading data')
    test_data = get_data(path=args.test_path, args=args, use_compound_names=args.use_compound_names,
                         skip_invalid_smiles=False)
    model = load_checkpoint(args.checkpoint_paths, cuda=args.cuda)
    data = test_data
    batch_size = args.batch_size
    disable_progress_bar = False
    model.eval()
    preds = []

    num_iters, iter_step = len(data), batch_size
    for i in trange(0, num_iters, iter_step, disable=disable_progress_bar):
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch = mol_batch.smiles()
        # Run model
        batch = smiles_batch
        with torch.no_grad():
            batch_preds = model.encoder(batch, )

        batch_preds = batch_preds.data.cpu().numpy()
        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)
    preds = np.array(preds)
    print(preds.shape)

    return preds
