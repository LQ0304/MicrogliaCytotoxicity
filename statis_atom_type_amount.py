from rdkit import Chem
import pandas as pd
from tqdm import tqdm

df = pd.read_excel('./data/Data sets for microglia cytotoxicity.xlsx')
data_line = df.values

atom_type_dict = {}
for data in tqdm(data_line):
    smiles = data[0]
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetAtoms()
    for atom in atoms:
        atom_symbol = atom.GetSymbol()
        if atom_symbol not in atom_type_dict.keys():
            atom_type_dict[atom_symbol] = 0
        else:
            atom_type_dict[atom_symbol] = atom_type_dict[atom_symbol] + 1

sort_atom_type_dict = sorted(atom_type_dict.items(),
                             key=lambda x: x[1],
                             reverse=True)
print(len(sort_atom_type_dict))
print(sort_atom_type_dict)
