from rdkit import Chem
from rdkit.Chem import Draw
import copy
import itertools
import os


def atom_valence(atom_types):
    """创建与atom_types对应的化合价列表
    atom_types ['C', 'O']  返回   [4, 2].
    Args:
      atom_types:   原子列表, e.g. ['C', 'O'].
    Returns:
      原子价列表.
    """
    periodic_table = Chem.GetPeriodicTable()
    return [max(list(periodic_table.GetValenceList(atom_type))) for atom_type in atom_types]


# 这里可以修改分子的动作
def get_valid_actions(state, atom_types,
                      allow_removal,
                      allow_no_modification,
                      allowed_ring_sizes,
                      allow_bonds_between_rings,
                      allow_atom_list,
                      allow_bond_list):
    """状态s给定后获取相应有效动作的集合

    Args:
        state: String SMILES字符串; 如果是None或空字符串，我们假设为没有原子或键的“空”状态。
        atom_types: 原子类型集合, e.g. {'C','N','O'}.
        allow_removal: Bool.是否允许删除原子和键的操作
        allow_no_modification: Bool.是否包含'no-op'不修饰操作。
        allowed_ring_sizes: 集合.允许环尺寸的整数集，用于删除一些会产生不允许大小的环的操作。
        allow_bonds_between_rings: Bool.是否允许在两个环中的原子之间添加键的动作。
    Returns:
        包含有效操作的字符串SMILES集合（当前状态下所有可以接受的下一个状态）
    Raises:
        ValueError: 如果state不代表有效分子。
    """
    if not state:  # 如果state是空的
        return copy.deepcopy(atom_types)
    # mol = Chem.MolFromSmiles(state)
    mol = state
    if mol is None:
        raise ValueError('Received invalid state: %s' % state)
    atom_valences = {atom_type: atom_valence([atom_type])[0] for atom_type in atom_types}  # {'C':4,'N':3,'O':2}
    atoms_with_free_valence = {}
    # 自由价（free valence）在分子中，虽然每个原子已与一定数目的原子结合在一起，
    # 但还有一定的再与其他原子结合的能力，这种剩余的化合价称为自由价或余价, 允许用新键替换至少一个原子的隐式H。
    for i in range(1, max(atom_valences.values())):
        atoms_with_free_valence[i] = [atom.GetIdx() for atom in mol.GetAtoms()
                                      if (atom.GetNumImplicitHs() >= i and
                                          atom.GetIdx() in allow_atom_list)]
        # {1:[原子序数], 2:[原子序数], 3:[原子序数]},有重复
    atom_addition, action_atom_addition, atom_addition_atom_list, \
    atom_addition_bond_list, atom_addition_mol_list = \
        _atom_addition(mol,
                       atom_types=atom_types,
                       atom_valences=atom_valences,
                       atoms_with_free_valence=atoms_with_free_valence,
                       allow_atom_list=allow_atom_list,
                       allow_bond_list=allow_bond_list)
    bond_addition, action_bond_addition, bond_addition_atom_list, \
    bond_addition_bond_list, bond_addition_mol_list = \
        _bond_addition(mol,
                       atoms_with_free_valence=atoms_with_free_valence,
                       allowed_ring_sizes=allowed_ring_sizes,
                       allow_bonds_between_rings=allow_bonds_between_rings,
                       allow_atom_list=allow_atom_list,
                       allow_bond_list=allow_bond_list)
    update_state_action(atom_addition, bond_addition, action_atom_addition, action_bond_addition,
                        atom_addition_atom_list, bond_addition_atom_list,
                        atom_addition_bond_list, bond_addition_bond_list,
                        atom_addition_mol_list, bond_addition_mol_list)
    if allow_removal:
        bond_removal, action_bond_removal, bond_removal_atom_list, \
        bond_removal_bond_list, bond_removal_mol_list = \
            _bond_removal(mol,
                          allow_atom_list=allow_atom_list,
                          allow_bond_list=allow_bond_list)
        update_state_action(atom_addition, bond_removal, action_atom_addition, action_bond_removal,
                            atom_addition_atom_list, bond_removal_atom_list,
                            atom_addition_bond_list, bond_removal_bond_list,
                            atom_addition_mol_list, bond_removal_mol_list)
    if allow_no_modification:
        add_no_modification_state, add_no_modification_action, add_no_modification_atom_list, \
        add_no_modification_bond_list, add_no_modification_mol_list = _no_modification(mol,
                                                                                       allow_atom_list=allow_atom_list,
                                                                                       allow_bond_list=allow_bond_list)
        update_state_action(atom_addition, add_no_modification_state, action_atom_addition, add_no_modification_action,
                            atom_addition_atom_list, add_no_modification_atom_list,
                            atom_addition_bond_list, add_no_modification_bond_list,
                            atom_addition_mol_list, add_no_modification_mol_list)

    return atom_addition, action_atom_addition, atom_addition_atom_list, atom_addition_bond_list, atom_addition_mol_list


def update_state_action(state_a, state_b, action_a, action_b, atom_a, atom_b, bond_a, bond_b, state_mol_a, state_mol_b):
    for i, state in enumerate(state_b):
        if state not in state_a:
            state_a.append(state)
            action_a.append(action_b[i])
            atom_a.append(atom_b[i])
            bond_a.append(bond_b[i])
            state_mol_a.append(state_mol_b[i])


def _atom_addition(state,
                   atom_types,
                   atom_valences,
                   atoms_with_free_valence,
                   allow_atom_list,
                   allow_bond_list):
    """添加原子的有效动作。
    原则是 a.找到原子原分子中的原子剩余价态；b.新添加原子价态>=键的价态

    Actions:
      * Add atom 分子图添加原子并用一个键链接
     每个增加的原子都由一个键连接到分子上. 每个现有的原子 每个现有原子的化合价允许键的类型(123键) 无芳香键
     e.g. {'C', 'O'},  (1)双键连C  (2)单键连C  (3)双键连O  (4)单键连O

    Args:
       state: RDKit Mol.分子MOL形式
       atom_types: 原子类型集合.
       atom_valences: 原子价字典
       atoms_with_free_valence: 自由价字典
       eg:{1: [0, 1, 3, 10, 11], 2: [1, 11], 3: []}遍历分子得所有原子，依次列出可以连接一价/二价/三价的原子的索引。

    Returns:
       SMILES集合; 可用的行动。
    """
    bond_order = {1: Chem.BondType.SINGLE,
                  2: Chem.BondType.DOUBLE,
                  3: Chem.BondType.TRIPLE}
    atom_addition = []  # 添加原子状态集
    action_atom_addition = []  # 添加原子动作集
    atom_addition_atom_list = []
    atom_addition_bond_list = []
    atom_addition_mol_list = []

    for i in bond_order:  # 遍历单，双和三键
        for atom in atoms_with_free_valence[i]:  # 满足键的原分子中对应的原子序数
            for element in atom_types:  # 遍历添加原子列表的原子
                if atom_valences[element] >= i:  # 添加的原子价态是否满足键的连接要求
                    valid_act = []
                    atom_addition_atom = allow_atom_list[:]
                    atom_addition_bond = allow_bond_list[:]

                    new_state = Chem.RWMol(state)  # RWMol类（用于分子读写的类）。这个类在修改分子方面，性能更好，它可以提供一个“活动的”分子，并且共享了mol对象的操作接口
                    idx = new_state.AddAtom(Chem.Atom(element))  # 返回值是新添加原子的索引
                    new_state.AddBond(atom, idx, bond_order[i])  # 在满足要求的原子索引与新添加原子的索引之间添加对应的键

                    bond_idx = new_state.GetBondBetweenAtoms(atom, idx).GetIdx()  # 得到对应键的索引

                    act = '+' + str(i) + element
                    valid_act.append(act)

                    atom_addition_atom.append(idx)
                    atom_addition_bond.append(bond_idx)

                    # for atom in state.GetAtoms():
                    #     atom.SetProp("atomNote", str(atom.GetIdx()))
                    # img1 = Draw.MolToImage(state, size=(1000, 1000))
                    # img1.show()
                    #
                    # for atom in new_state.GetAtoms():
                    #     atom.SetProp("atomNote", str(atom.GetIdx()))
                    # img2 = Draw.MolToImage(new_state, size=(1000, 1000))
                    # img2.show()

                    sanitize_result = Chem.SanitizeMol(new_state, catchErrors=True)  # 核对检查分子,计算凯库勒式/检查化合价/芳香性/共轭及杂化
                    if sanitize_result != Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
                        continue  # 分子消毒失败 跳过本次循环继续

                    # for atom in new_state.GetAtoms():
                    #     atom.SetProp("atomNote", str(atom.GetIdx()))
                    # img3 = Draw.MolToImage(new_state, size=(1000, 1000))
                    # img3.show()

                    add_state = Chem.MolToSmiles(new_state)

                    if add_state not in atom_addition:
                        atom_addition.append(add_state)
                        action_atom_addition.append(valid_act)
                        atom_addition_atom_list.append(atom_addition_atom)
                        atom_addition_bond_list.append(atom_addition_bond)
                        atom_addition_mol_list.append(new_state)

    return atom_addition, action_atom_addition, atom_addition_atom_list, atom_addition_bond_list, atom_addition_mol_list


def _bond_addition(state,
                   atoms_with_free_valence,
                   allowed_ring_sizes,
                   allow_bonds_between_rings,
                   allow_atom_list,
                   allow_bond_list):
    """添加键的有效操作
    Actions:
      * 0->{1,2,3}
      * 1->{2,3}
      * 2->{3}
    注意，芳香键不允许被修改
    Args:
      state: RDKit Mol.
      atoms_with_free_valence: 自由价字典. e.g. atoms_with_free_valence[2]中的所有原子索引至少有两个可用的价位。
      allowed_ring_sizes: 集合.允许环尺寸的整数集，用于删除一些会产生不允许大小的环的操作 [5,6].
      allow_bonds_between_rings: Bool.是否允许在两个环中的原子之间添加键的动作。

    Returns:
      包含有效操作的字符串SMILES集合.
    """
    bond_orders = [None,
                   Chem.BondType.SINGLE,
                   Chem.BondType.DOUBLE,
                   Chem.BondType.TRIPLE]
    bond_addition = []
    action_bond_addition = []
    bond_addition_atom_list = []
    bond_addition_bond_list = []
    bond_addition_mol_list = []

    # 遍历自由价字典，分别获取自由价为1及对应原子索引，一直到自由价为3.
    for valence, atoms in atoms_with_free_valence.items():  # 1, [0, 1, 3, 10, 11]
        for atom1, atom2 in itertools.combinations(atoms, 2):  # e.g自由价为1对应的原子索引为[0, 1, 3, 10, 11]，则返回两两不重复组合
            valid_action = []
            bond_addition_atom = allow_atom_list[:]
            bond_addition_bond = allow_bond_list[:]

            bond = Chem.Mol(state).GetBondBetweenAtoms(atom1, atom2)  # 得到目前两个原子之间的键类型
            new_state = Chem.RWMol(state)
            Chem.Kekulize(new_state, clearAromaticFlags=True)
            # 转化后，类型中虽然变为单键和双键，但依然是芳香键,之所以仍然为芳香键，是因为分子有一个跟芳香性相关的属性Flags，记录了芳香性的信息。
            # 可以在kelulize时将clearAromaticFlags参数设置为True

            # 两分子之间有键的情况
            if bond is not None:
                if bond.GetBondType() not in bond_orders:
                    continue  # Skip aromatic bonds.键的类型在定义bond_orders中
                # idx = bond.GetIdx()  # 获取键的索引
                # 在原键基础上用2键或3键替换
                bond_order = bond_orders.index(bond.GetBondType())  # 原分子两原子之间键的类型1;2;3
                bond_order += valence  # 加上对应遍历的价态作为新键的类型，eg：1+1/2+1/1+2/

                act = '+' + str(valence)
                valid_action.append(act)

                if bond_order < len(bond_orders):  # 需要替换的键的类型小于4，即新键类型只能由四种对应原子之间四种键的类型0，1，2，3
                    idx = bond.GetIdx()  # 获取键的索引
                    if idx not in bond_addition_bond:
                        bond_addition_bond = bond_addition_bond.append(idx)
                    bond.SetBondType(bond_orders[bond_order])  # 修改为新建的类型
                    new_state.ReplaceBond(idx, bond)

                else:
                    continue
            # 不允许在环上原子进行连接键，跳过本次循环。
            elif (not allow_bonds_between_rings and
                  (state.GetAtomWithIdx(atom1).IsInRing() and
                   state.GetAtomWithIdx(atom2).IsInRing())):
                continue
            # 如果当前两个原子之间的距离不在允许的环尺寸内
            elif (allowed_ring_sizes is not None and
                  len(Chem.rdmolops.GetShortestPath(state, atom1, atom2)) not in allowed_ring_sizes):
                # 使用Bellman-Ford算法找到两个原子之间的最短路径。
                continue
            else:  # 两个原子之间无键的情况,一定成环
                new_state.AddBond(atom1, atom2, bond_orders[valence])
            sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
            if sanitization_result != Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
                continue
            add_state = Chem.MolToSmiles(new_state)
            if (add_state not in bond_addition) and valid_action:  # 新加的状态不在状态集里并且action不为空集
                bond_addition.append(add_state)
                action_bond_addition.append(valid_action)
                bond_addition_atom_list.append(bond_addition_atom)
                bond_addition_bond_list.append(bond_addition_bond)
                bond_addition_mol_list.append(new_state)

    return bond_addition, action_bond_addition, bond_addition_atom_list, \
           bond_addition_bond_list, bond_addition_mol_list


def _bond_removal(state, allow_atom_list, allow_bond_list):
    """移除键的有效操作.
    Actions:
           * 3->{2，1，0}
           * 2->{1，0}
           * 1->{0}
    只有当生成的图中有0个或一个断开的原子时，键才会被移除(单>无键);不允许创建多原子断开连接的片段。.
    Args:
        state: RDKit Mol.
    Returns:
        包含有效操作的字符串SMILES集合.
    """
    bond_orders = [None,
                   Chem.BondType.SINGLE,
                   Chem.BondType.DOUBLE,
                   Chem.BondType.TRIPLE]
    bond_removal = []
    action_bond_removal = []
    bond_removal_atom_list = []
    bond_removal_bond_list = []
    bond_removal_mol_list = []

    for valence in [1, 2, 3]:  # 遍历价态
        # 当valence=1时, 单键-0；双键-单键；三键-双键；
        # 当valence=2时, 双键-0；三键-单键；
        # 当valence=3时, 三键-0
        for bond in state.GetBonds():  # 遍历分子的键
            valid_action = []
            bond_addition_atom = allow_atom_list[:]
            bond_removal_bond = allow_bond_list[:]

            bond = Chem.Mol(state).GetBondBetweenAtoms(bond.GetBeginAtomIdx(),
                                                       bond.GetEndAtomIdx())
            if bond.GetIdx() in allow_bond_list:
                if bond.GetBondType() not in bond_orders:  # 仅对单双三键进行处理
                    continue  # 跳过芳香键
                new_state = Chem.RWMol(state)
                Chem.Kekulize(new_state, clearAromaticFlags=True)
                # 计算新的键替换旧键.
                bond_order = bond_orders.index(bond.GetBondType())
                bond_order -= valence

                act = '-' + str(valence)
                valid_action.append(act)

                if bond_order > 0:  # 3键变2或1键；2键变1键
                    idx = bond.GetIdx()
                    bond.SetBondType(bond_orders[bond_order])
                    new_state.ReplaceBond(idx, bond)
                    sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)

                    if sanitization_result != Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
                        continue
                    add_state = Chem.MolToSmiles(new_state)
                    if add_state not in bond_removal:
                        bond_removal.append(add_state)
                        action_bond_removal.append(valid_action)
                        bond_removal_atom_list.append(bond_addition_atom)
                        bond_removal_bond_list.append(bond_removal_bond)
                        bond_removal_mol_list.append(new_state)
            else:
                continue

            # elif bond_order == 0:  # 完全移除这个单键.
            #     atom1 = bond.GetBeginAtom().GetIdx()
            #     atom2 = bond.GetEndAtom().GetIdx()
            #     new_state.RemoveBond(atom1, atom2)
            #     sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
            #     if sanitization_result != Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
            #         continue
            #     smiles = Chem.MolToSmiles(new_state)
            #     # eg：'Cc1cn2c(n1)OC(Cl)=CC2.O'  分解后为a.一个独立的分子; b.一个分子和一个原子; c.两个分子，并用点分开
            #     parts = sorted(smiles.split('.'), key=len)  # eg：['O', 'Cc1cn2c(n1)OC(Cl)=CC2'] 用点拆开，并根据长度进行排序
            #     # 有效的键移除动作集定义为移除现有键的动作，只生成一个独立的分子;
            #     # 一个分子和一个原子;
            #     # 若是两个分子，则将其进行删除（由于不能对分子进行大型修改）
            #     if len(parts) == 1 or len(parts[0]) == 1:  # 一个分子或一个分子，与一个原子
            #         if parts[-1] not in bond_removal:
            #             bond_removal.append(parts[-1])
            #             action_bond_removal.append(valid_action)

    return bond_removal, action_bond_removal, \
           bond_removal_atom_list, bond_removal_bond_list, bond_removal_mol_list


def _no_modification(state, allow_atom_list, allow_bond_list):
    add_no_modification_state = [Chem.MolToSmiles(state)]
    add_no_modification_action = [[]]
    add_no_modification_mol_list = [state]
    return add_no_modification_state, add_no_modification_action, \
           allow_atom_list, allow_bond_list, add_no_modification_mol_list


class Molecule(object):
    """定义生成分子的马尔可夫决策过程"""

    def __init__(self,
                 atom_types,
                 init_mol,
                 allow_removal=True,
                 allow_no_modification=False,  # False
                 allow_bonds_between_rings=False,  # True,
                 allowed_ring_sizes=[6],
                 max_steps=10,
                 allow_atom_list=None,
                 allow_bond_list=None):
        """初始化 MDP 参数."""
        # if isinstance(init_mol, Chem.Mol):  # 判断初始分子是否为mol类型 并将mol类型的初始分子转化为smiles码
        #     init_mol = init_mol
        self.init_mol = init_mol
        self.atom_types = atom_types
        self.allow_removal = allow_removal
        self.allow_no_modification = allow_no_modification
        self.allow_bonds_between_rings = allow_bonds_between_rings
        self.allowed_ring_sizes = allowed_ring_sizes  # 允许成环尺寸
        self._state = None
        self._valid_states = []
        self._valid_actions = []
        self._valid_atom_list = []
        self._valid_bond_list = []
        self._valid_mol_list = []

        self.max_steps = max_steps
        self.counter = self.max_steps
        self._max_bonds = 4
        atom_types = list(self.atom_types)
        self._max_new_bonds = dict(list(zip(atom_types, atom_valence(atom_types))))
        self.allow_atom_list = allow_atom_list
        self.allow_bond_list = allow_bond_list

    @property  # property属性的定义和调用要注意以下几点：调用时，无需括号，加上就错了；并且仅有一个self参数
    def state(self):
        return self._state

    @property
    def num_steps_taken(self):
        return self.counter

    def initialize(self):
        """将MDP重置为其初始状态."""
        self._state = self.init_mol
        self._valid_states, self._valid_actions, self._valid_atom_list, self._valid_bond_list, \
        self._valid_mol_list = \
            self.get_valid_actions(
                force_rebuild=True)
        self.counter = 0

    def get_valid_actions(self, state=None, force_rebuild=False):
        if state is None:
            if self._valid_actions and not force_rebuild:
                return copy.deepcopy(self._valid_actions)
        state = self._state
        # if isinstance(state, Chem.Mol):
        #     state = Chem.MolToSmiles(state)
        self._valid_states, self._valid_actions, \
        self._valid_atom_list, self._valid_bond_list, self._valid_mol_list = get_valid_actions(
            state,
            atom_types=self.atom_types,
            allow_removal=self.allow_removal,
            allow_no_modification=self.allow_no_modification,
            allowed_ring_sizes=self.allowed_ring_sizes,
            allow_bonds_between_rings=self.allow_bonds_between_rings,
            allow_atom_list=self.allow_atom_list,
            allow_bond_list=self.allow_bond_list)
        return copy.deepcopy(self._valid_states), copy.deepcopy(self._valid_actions), \
               copy.deepcopy(self._valid_atom_list), copy.deepcopy(self._valid_bond_list), \
               copy.deepcopy(self._valid_mol_list)

    def visualize_state(self, state=None, **kwargs):
        """画出分子的当前状态.
        """
        if state is None:
            state = self._state
        if isinstance(state, str):
            state = Chem.MolFromSmiles(state)
        return Draw.MolToImage(state, **kwargs)

# # Test code
# save_fig_path = './data/save_fig'
# os.makedirs(save_fig_path, exist_ok=True)
# test_smiles = 'OCc1cn2c(n1)OC(Cl)=CC2'  # 'CC(O)c1cn2c(n1)OC(Cl)=CC2'
# test_mol = Chem.MolFromSmiles(test_smiles)
# add_atom_types = ['C', 'N', 'O']
# R_max = []
# RL_sequence_list = []
#
# for i in range(10):  # 一个分子进行10step修改
#     RL_sequence_list.append(test_mol)
#     FT_molecule = Molecule(atom_types=add_atom_types, init_mol=test_mol)
#     FT_molecule.initialize()
#     actions = list(FT_molecule._valid_actions)
#     r_value = []
#     for a in actions:
#         m = Chem.MolFromSmiles(a)
#         r_value.append(qed(m))
#     R_max.append(max(r_value))  # greedy-epslion
#     index = r_value.index(max(r_value))
#     action = actions[index]
#     test_mol = action
#     img = Draw.MolToImage(Chem.MolFromSmiles(test_mol))
#     img.save(os.path.join(save_fig_path, str(i) + '.png'))
