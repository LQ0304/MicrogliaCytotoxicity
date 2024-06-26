"""
数据预处理第三步：
计算词汇库（字典）
Input:语料库
Output:词汇库
"""
import argparse
import pickle
from collections import Counter


class TorchVocab(object):
    """
    :property freqs: collections.Counter, 保存语料中单词出现频率的对象
    :property stoi: collections.defaultdict, 显示string → id的对应字典
    :property itos: collections.defaultdict, 显示id → string的对应字典
    """
    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """
        :param counter: collections.Counter, 用于测量数据中单词出现频率的计数器
        :param max_size: int, vocabulary的最大大小，如果为None则没有最大值，默认default=None
        :param min_freq: int, vocabulary中出现单词的最小频率。少于此出现次数的单词不能添加到词汇表中。
        :param specials: list of str, vocabulary中预先注册的token
        :param vectors: list of vectors, 预训练向量（例如）Vocab.load_vectors
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = 1 # max(min_freq, 1)

        self.itos = list(specials)
        # 创建词汇表时不计算special tokens的出现频率
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # 首先按频率排序，然后按字母排序
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
        
        # 出现频率小于min_freq的不添加到词汇表
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # 通过替换字典的k和v创建stoi
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    # 内置方法：判断是否相等，相等返回True，不等返回False
    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    # 内置方法：返回长度
    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"], max_size=max_size, min_freq=min_freq) # 继承TorchVocab(object)

    # override用
    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    # override用
    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


# 从文本文件创建词汇
class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        print("Building Vocab")
        counter = Counter()
        # 循环读取smiles码，记录词汇保存为字典counter
        for line in texts:
            if isinstance(line, list): # 判断line是否属于list
                words = line
            else:
                words = line.replace("\n", "").replace("\t", "").split() # 去除换行和tab符并拆分为列表

            for word in words:
                counter[word] += 1

        super().__init__(counter, max_size=max_size, min_freq=min_freq) # 继承Vocab(TorchVocab)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod # 静态方法
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description='Build a vocabulary pickle')
    parser.add_argument('--corpus_path', '-c', type=str, default='data/chembl_28_200_corpus.txt', help='path to th ecorpus') # 语料库路径
    parser.add_argument('--out_path', '-o', type=str, default='data/vocab_gene.pkl', help='output file') # 输出词汇库路径
    parser.add_argument('--min_freq', '-m', type=int, default=500, help='minimum frequency for vocabulary')
    parser.add_argument('--vocab_size', '-v', type=int, default=None, help='max vocabulary size')
    parser.add_argument('--encoding', '-e', type=str, default='utf-8', help='encoding of corpus')
    args = parser.parse_args()

    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(args.out_path)

if __name__=='__main__':
    main()