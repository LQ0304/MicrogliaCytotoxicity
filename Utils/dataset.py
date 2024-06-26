import torch
from torch.utils.data import Dataset

from .enumerator import SmilesEnumerator
from .utils import split
import torch
from torch.utils.data import Dataset
from .utils import SmilesEnumerator
import numpy as np
import re
import math


class SmileDataset(Dataset):

    def __init__(self, args, data, content, block_size, aug_prob=0.5, prop=None, scaffold=None, scaffold_maxlen=None):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.sca = scaffold
        self.scaf_max_len = scaffold_maxlen
        self.debug = args.debug
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob

    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        # smiles, prop, scaffold = self.data[idx], self.prop[idx], self.sca[idx]    # self.prop.iloc[idx, :].values  --> if multiple properties
        smiles = smiles.strip()
        # scaffold = scaffold.strip()

        p = np.random.uniform()
        if p < self.aug_prob:
            smiles = self.tfm.randomize_smiles(smiles)

        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        smiles += str('<') * (self.max_len - len(regex.findall(smiles)))

        if len(regex.findall(smiles)) > self.max_len:
            smiles = smiles[:self.max_len]

        smiles = regex.findall(smiles)

        # scaffold += str('<')*(self.scaf_max_len - len(regex.findall(scaffold)))

        # if len(regex.findall(scaffold)) > self.scaf_max_len:
        #     scaffold = scaffold[:self.scaf_max_len]
        #
        # scaffold=regex.findall(scaffold)

        dix = [self.stoi[s] for s in smiles]
        # sca_dix = [self.stoi[s] for s in scaffold]

        # sca_tensor = torch.tensor(sca_dix, dtype=torch.long)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        # prop = torch.tensor([prop], dtype=torch.long)
        # prop = torch.tensor([prop], dtype = torch.float)
        return x, y  # , prop, sca_tensor

    # def __getitem__(self, idx):
    #     smiles, scaffold = self.data[idx], self.sca[idx]  # self.prop.iloc[idx, :].values  --> if multiple properties
    #     smiles = smiles.strip()
    #     scaffold = scaffold.strip()
    #
    #     p = np.random.uniform()
    #     if p < self.aug_prob:
    #         smiles = self.tfm.randomize_smiles(smiles)
    #
    #     pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    #     regex = re.compile(pattern)
    #     smiles += str('<') * (self.max_len - len(regex.findall(smiles)))
    #
    #     if len(regex.findall(smiles)) > self.max_len:
    #         smiles = smiles[:self.max_len]
    #
    #     smiles = regex.findall(smiles)
    #
    #     scaffold += str('<') * (self.scaf_max_len - len(regex.findall(scaffold)))
    #
    #     if len(regex.findall(scaffold)) > self.scaf_max_len:
    #         scaffold = scaffold[:self.scaf_max_len]
    #
    #     scaffold = regex.findall(scaffold)
    #
    #     dix = [self.stoi[s] for s in smiles]
    #     sca_dix = [self.stoi[s] for s in scaffold]
    #
    #     sca_tensor = torch.tensor(sca_dix, dtype=torch.long)
    #     x = torch.tensor(dix[:-1], dtype=torch.long)
    #     y = torch.tensor(dix[1:], dtype=torch.long)
    #     # prop = torch.tensor([prop], dtype=torch.long)
    #     # prop = torch.tensor([prop], dtype=torch.float)
    #     return x, y, sca_tensor


class SmileScaDataset(Dataset):

    def __init__(self, args, data, content, block_size, aug_prob=0.5, prop=None, scaffold=None, scaffold_maxlen=None):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.sca = scaffold
        self.scaf_max_len = scaffold_maxlen
        self.debug = args.debug
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob

    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        smiles, scaffold = self.data[idx], self.sca[idx]  # self.prop.iloc[idx, :].values  --> if multiple properties
        smiles = smiles.strip()
        scaffold = scaffold.strip()

        p = np.random.uniform()
        if p < self.aug_prob:
            smiles = self.tfm.randomize_smiles(smiles)

        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        smiles += str('<') * (self.max_len - len(regex.findall(smiles)))

        if len(regex.findall(smiles)) > self.max_len:
            smiles = smiles[:self.max_len]

        smiles = regex.findall(smiles)

        scaffold += str('<') * (self.scaf_max_len - len(regex.findall(scaffold)))

        if len(regex.findall(scaffold)) > self.scaf_max_len:
            scaffold = scaffold[:self.scaf_max_len]

        scaffold = regex.findall(scaffold)

        dix = [self.stoi[s] for s in smiles]
        sca_dix = [self.stoi[s] for s in scaffold]

        sca_tensor = torch.tensor(sca_dix, dtype=torch.long)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        # prop = torch.tensor([prop], dtype=torch.long)
        # prop = torch.tensor([prop], dtype=torch.float)
        return x, y, sca_tensor


class SmilePropDataset(Dataset):

    def __init__(self, args, data, content, block_size, aug_prob=0.5, prop=None, scaffold=None, scaffold_maxlen=None):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.sca = scaffold
        self.scaf_max_len = scaffold_maxlen
        self.debug = args.debug
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob

    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        # smiles = self.data[idx]
        smiles, prop = self.data[idx], self.prop[idx]  # self.prop.iloc[idx, :].values  --> if multiple properties
        smiles = smiles.strip()

        p = np.random.uniform()
        if p < self.aug_prob:
            smiles = self.tfm.randomize_smiles(smiles)

        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        smiles += str('<') * (self.max_len - len(regex.findall(smiles)))

        if len(regex.findall(smiles)) > self.max_len:
            smiles = smiles[:self.max_len]

        smiles = regex.findall(smiles)

        dix = [self.stoi[s] for s in smiles]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        # prop = torch.tensor([prop], dtype=torch.long)
        prop = torch.tensor([prop], dtype=torch.float)
        return x, y, prop  # , sca_tensor


class SmilePropDataset(Dataset):

    def __init__(self, args, data, content, block_size, aug_prob=0.5, prop=None, scaffold=None, scaffold_maxlen=None):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.sca = scaffold
        self.scaf_max_len = scaffold_maxlen
        self.debug = args.debug
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob

    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        smiles, prop = self.data[idx], self.prop.iloc[idx,:]  # self.prop.iloc[idx, :].values  --> if multiple properties
        smiles = smiles.strip()

        p = np.random.uniform()
        if p < self.aug_prob:
            smiles = self.tfm.randomize_smiles(smiles)

        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        smiles += str('<') * (self.max_len - len(regex.findall(smiles)))

        if len(regex.findall(smiles)) > self.max_len:
            smiles = smiles[:self.max_len]

        smiles = regex.findall(smiles)

        dix = [self.stoi[s] for s in smiles]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        # prop = torch.tensor([prop], dtype=torch.long)
        prop = torch.tensor([prop], dtype=torch.float)
        return x, y, prop  # , sca_tensor


class SmilePropScaDataset(Dataset):

    def __init__(self, args, data, content, block_size, aug_prob=0.5, prop=None, scaffold=None, scaffold_maxlen=None):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.sca = scaffold
        self.scaf_max_len = scaffold_maxlen
        self.debug = args.debug
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob

    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):

        smiles, prop, scaffold = self.data[idx], self.prop.iloc[idx,:], self.sca[idx]  # self.prop.iloc[idx, :].values  --> if multiple properties
        smiles = smiles.strip()
        scaffold = scaffold.strip()

        p = np.random.uniform()
        if p < self.aug_prob:
            smiles = self.tfm.randomize_smiles(smiles)

        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        smiles += str('<') * (self.max_len - len(regex.findall(smiles)))

        if len(regex.findall(smiles)) > self.max_len:
            smiles = smiles[:self.max_len]

        smiles = regex.findall(smiles)

        scaffold += str('<') * (self.scaf_max_len - len(regex.findall(scaffold)))

        if len(regex.findall(scaffold)) > self.scaf_max_len:
            scaffold = scaffold[:self.scaf_max_len]

        scaffold = regex.findall(scaffold)

        dix = [self.stoi[s] for s in smiles]
        sca_dix = [self.stoi[s] for s in scaffold]

        sca_tensor = torch.tensor(sca_dix, dtype=torch.long)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        # prop = torch.tensor([prop], dtype=torch.long)
        prop = torch.tensor([prop], dtype=torch.float)
        return x, y, prop, sca_tensor
PAD = 0
MAX_LEN = 163

class Randomizer(object):

    def __init__(self):
        self.sme = SmilesEnumerator()
    
    def __call__(self, sm):
        sm_r = self.sme.randomize_smiles(sm) # Random transoform
        if sm_r is None:
            sm_spaced = split(sm) # Spacing
        else:
            sm_spaced = split(sm_r) # Spacing
        sm_split = sm_spaced.split()
        if len(sm_split)<=MAX_LEN - 2:
            return sm_split # List
        else:
            return split(sm).split()

    def random_transform(self, sm):
        '''
        function: Random transformation for SMILES. It may take some time.
        input: A SMILES
        output: A randomized SMILES
        '''
        return self.sme.randomize_smiles(sm)

class Seq2seqDataset(Dataset):

    def __init__(self, smiles, vocab, seq_len=163, transform=Randomizer()):
        self.smiles = smiles
        self.vocab = vocab
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        sm = self.smiles[item]
        sm = sm.split(' ')
        # sm = self.transform(sm) # List
        content = [self.vocab.stoi.get(token) for token in sm]
        X = [self.vocab.sos_index] + content
        padding = [self.vocab.pad_index]*(self.seq_len - len(X))
        X.extend(padding)
        return torch.tensor(X)

