import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Variable

class Generator(nn.Module):

    def __init__(self, voc, voc_size, num_layers, embed_dim, hidden_dim):
        super(Generator, self).__init__()
        self.voc = voc
        self.voc_size = voc_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(voc_size, embed_dim)
        self.rnn = nn.RNN(input_size=embed_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          bias=True,
                          batch_first=True,
                          dropout=0,
                          bidirectional=False)
        self.rnn2out = nn.Linear(hidden_dim, voc_size)
        self.NLL_loss = nn.NLLLoss()

    def forward(self, x, init_state):  # x 128 'GO',

        x = x.unsqueeze(1)                      # 128*1
        x = self.embedding(x)                   # 128*1*200
        out, h = self.rnn(x, init_state)        # out 128*1*512 h 2 3*128*512
        out = self.rnn2out(out.squeeze(1))      # [128,62]
        return out, h

    def init_h(self, batch_size):

        if torch.cuda.is_available():
            initial_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda()
        else:
            initial_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return initial_state


    def likelihood(self, inputs):
        inputs = inputs.long()
        batch_size, seq_length = inputs.size()
        start_token = Variable(torch.zeros(batch_size, 1).long())  # 定义一个全0向量，形状是128*1
        start_token[:] = self.voc['>']  # 把start_token中的所有元素定义为【‘GO’】所对应的索引61
        train_x = torch.cat((start_token, inputs[:, :]), 1)
        h = self.init_h(batch_size)
        log_probs = Variable(torch.zeros(batch_size))  # 定义一个128个一维的变量（用来存储每个SMILES encode的log_probs和）
        entropy = Variable(torch.zeros(batch_size))

        for step in range(seq_length):                             # for循环的作用是从一个SMILES码开始取元素所对应的索引值，以列学习每个词                                                                 # 学习每一个SMILES的encode的前一个元素与后一个元素的关系
            logits, h = self.forward(train_x[:, step], h)          #（以列进行学习） 取x的所有行，第i列 和 隐状态作为self.rnn的输入，输出为GRU的输出logits：128*62， h ：2 3*128*512
            log_prob = F.log_softmax(logits,1)                     # F.log_softmax是Pytorch中定义的函数，输出为log_prob
            prob = F.softmax(logits,1)                             # F.softmaxPytorch中定义的函数，输出为prob 128*62 经过softmax函数后概率变为（0,1）之间，每一行的概率和为1
            log_probs += NLLLoss(log_prob, inputs[:, step])        # NLLLoss函数的输入为log_prob，和target（128*max_length）以列形式，随着列的遍历，逐列相加，最后返回128*1
            entropy += -torch.sum((log_prob * prob), 1)            # 目的是计算每个SMILES的熵值
        return log_probs, entropy

    def PGLoss(self, score, reward):
        """Policy gradicent method for loss function construction under reinforcement learning

        Arguments:
            score (FloatTensor): m X n matrix of the probability for for generating
                each token in each SMILES string. m is the No. of samples,
                n is n is the maximum size of the tokens for the whole SMILES strings.
                In general, it is the output of likelihood methods. It requies gradient.
            reward (FloatTensor): if it is the final reward given by environment (predictor),
                it will be m X 1 matrix, the m is the No. of samples.
                if it is the step rewards obtained by Monte Carlo Tree Search based on environment,
                it will b m X n matrix, the m is the No. of samples, n is the No. of sequence length.
        Returns:
            loss (FloatTensor): The loss value calculated with REINFORCE loss function
                It requies gradient for parameter update.
        """
        loss = score * reward  # 每个字母的概率*活性的概率 [500,100]
        loss = -loss.mean()  # 所有数的均值
        return loss


    def sample(self, batch_size, max_length=100):
        """
            Sample a batch of sequences 对一批序列进行采样

            Args:
                batch_size : Number of sequences to sample 输入：采样序列的数量128，自定义序列的最大长度为140
                max_length:  Maximum length of the sequences

            Outputs:
            seqs: (batch_size, seq_length) The sampled sequences. 输出：采样序列的encode，每个序列的log似然函数128*1，序列的熵128*1
            log_probs : (batch_size) Log likelihood for each sequence.
            entropy: (batch_size) The entropies for the sequences. Not
                                    currently used.
        """
        start_token = Variable(torch.zeros(batch_size).long())       # 定义一个start_token为128全0矩阵
        start_token[:] = self.voc['>']                       # start_token中的所有元素为['Go']所对应的字典的索引值
        h = self.init_h(batch_size)                                  # 定义一个初始的隐状态3*128*512
        x = start_token                                              # x为start_token  128*1（元素为61）

        sequences = []                                               # 定义一个列表sequences

        for step in range(max_length):                               # 遍历step（140）
            logits, h = self.forward(x, h)                           # logits:128*62      # 取x的所有行，第i列 和 隐状态作为self.rnn的输入，输出为GRU的输出logits：128*62， h ：3*128*512
            prob = F.softmax(logits,1)                               # prob:128*62        # F.log_softmax是Pytorch中定义的函数，输出为log_prob
            x = torch.multinomial(prob,1).view(-1)                   # x：128*1           # torch.multinomial(prob)随机抽样Random sampling的索引，采样权重最大的或者次之，.view(-1)是不告诉函数有多少列的情况下，根据原数据和batchsize自动分配维数（在这里是1维）。
            sequences.append(x.view(-1, 1))                          # 按一列连接         # 在sequences列表中加入x（不知道多少行，1列）
        sequences = torch.cat(sequences, 1)                          # 按列（1）进行拼接，行（0）
        return sequences.data                                        # 返回序列， log_probs, entropy

def NLLLoss(inputs, targets):  # target 128
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).cuda()  # 若cuda可获得，则调用cuda
    else:
        target_expanded = torch.zeros(inputs.size())         # 否则,正常实现

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)  # 128*69
   # .scatter函数，给标签一个one-hot码，按照.contiguous()的行对target_expanded加入1.0
    loss = Variable(target_expanded) * inputs  # 128*69
    loss = torch.sum(loss, 1)                  # 128
    return loss