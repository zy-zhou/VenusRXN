import os
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, AllChem
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from rdkit import RDConfig
from .graph_utils.algos import to_dense_adj, compute_shortest_paths

fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

x_vocabs = {
    'atomic_num': list(range(119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'degree': list(range(6)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(9)),
    'num_radical_electrons': list(range(5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER'
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True]
}

e_vocabs = {
    'bond_type': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'HYDROGEN',
        'IONIC',
        'OTHER'
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}

for vocabs in (x_vocabs, e_vocabs):
    for feature, tokens in vocabs.items():
        vocabs[feature] = {token: idx for idx, token in enumerate(tokens)}

x_dim = sum(map(len, x_vocabs.values())) + 1 # with mask token
e_dim = sum(map(len, e_vocabs.values()))

def stoi(token, vocab):
    if token not in vocab:
        return len(vocab) - 1
    return vocab[token]

def get_atom_features(atom):
    # Calculate implicit valence before getting num hydrogens
    atom.UpdatePropertyCache()
    
    encoding = [
        stoi(atom.GetAtomicNum(), x_vocabs['atomic_num']),
        stoi(str(atom.GetChiralTag()), x_vocabs['chirality']),
        stoi(atom.GetDegree(), x_vocabs['degree']),
        stoi(atom.GetFormalCharge(), x_vocabs['formal_charge']),
        stoi(atom.GetTotalNumHs(), x_vocabs['num_hs']),
        stoi(atom.GetNumRadicalElectrons(), x_vocabs['num_radical_electrons']),
        stoi(str(atom.GetHybridization()), x_vocabs['hybridization']),
        stoi(atom.GetIsAromatic(), x_vocabs['is_aromatic']),
        stoi(atom.IsInRing(), x_vocabs['is_in_ring'])
    ]
    return encoding

def get_bond_features(bond):
    encoding = [
        stoi(str(bond.GetBondType()), e_vocabs['bond_type']),
        stoi(str(bond.GetStereo()), e_vocabs['stereo']),
        stoi(bond.GetIsConjugated(), e_vocabs['is_conjugated'])
    ]
    return encoding

def labels_to_one_hots(labels, vocabs, mask_token=False):
    vocab_sizes = map(len, vocabs.values())
    encoding = [F.one_hot(labels[:, i], num_classes=size) for i, size in enumerate(vocab_sizes)]
    encoding = torch.cat(encoding, dim=1) # num_nodes, total_vocab_size
    if mask_token:
        mask_col = torch.zeros((labels.size(0), 1), dtype=torch.long)
        encoding = torch.cat([encoding, mask_col], dim=1)
    return encoding.float()

def one_hots_to_labels(one_hots, vocabs):
    vocab_sizes = list(map(len, vocabs.values()))
    splits = torch.split(one_hots[:, :sum(vocab_sizes)], vocab_sizes, dim=1)
    labels = torch.stack([split.argmax(1) for split in splits], dim=1)
    return labels

def get_mapped_atoms(molecules):
    mapped_atoms = {}
    for mol in molecules:
        for atom in mol.GetAtoms():
            aam_num = atom.GetAtomMapNum()
            
            if aam_num > 0:
                if aam_num in mapped_atoms: # duplicate AtomMapNum, reset it
                    atom.SetAtomMapNum(0)
                else:
                    mapped_atoms[aam_num] = atom
    return mapped_atoms

def match_aams(reactants, products):
    r_mapped_atoms = get_mapped_atoms(reactants)
    p_mapped_atoms = get_mapped_atoms(products)

    r_aam_diff = r_mapped_atoms.keys() - p_mapped_atoms.keys()
    for aam_num in r_aam_diff:
        r_mapped_atoms[aam_num].SetAtomMapNum(0)
    
    p_aam_diff = p_mapped_atoms.keys() - r_mapped_atoms.keys()
    for aam_num in p_aam_diff:
        p_mapped_atoms[aam_num].SetAtomMapNum(0)

def get_atom_id(mol_idx, atom):
    '''
    If the atom is mapped, return AtomMapNum, else return a string of mol_idx.atom_idx
    '''
    aam_num = atom.GetAtomMapNum()
    if aam_num > 0:
        return aam_num
    else:
        return f'{mol_idx}.{atom.GetIdx()}'

def get_mol_features(molecules):
    '''
    Return two dicts of atom features with atom ids as keys.
    The first dict stores features of atoms with valid AtomMapNum, sorted by AtomMapNum.
    The second dict stores features of unmapped atoms.
    '''
    mapped_features, unmapped_features = {}, {}
    for mol_idx, mol in enumerate(molecules):
        for atom in mol.GetAtoms():
            atom_id = get_atom_id(mol_idx, atom)
            atom_features = get_atom_features(atom)

            if type(atom_id) is int:
                mapped_features[atom_id] = atom_features
            else:
                unmapped_features[atom_id] = atom_features
    
    sorted_features = {key: mapped_features[key] for key in sorted(mapped_features)}
    return sorted_features, unmapped_features

def build_mol_graph(molecules, mapped_features, unmapped_features):
    # combine features and get atom indices
    atom_ids = list(mapped_features.keys()) + list(unmapped_features.keys())
    atom_features = list(mapped_features.values()) + list(unmapped_features.values())
    id_to_index = {key: idx for idx, key in enumerate(atom_ids)}
    
    # obtain edge features
    edge_indices = []
    edge_attrs = []
    
    for mol_idx, mol in enumerate(molecules):
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            
            # edge index and features
            begin_idx = id_to_index[get_atom_id(mol_idx, begin_atom)]
            end_idx = id_to_index[get_atom_id(mol_idx, end_atom)]
            edge_attr = get_bond_features(bond)
            
            # edges are bidirectional
            edge_indices.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
            edge_attrs.extend([edge_attr, edge_attr])
    
    # to Tensor
    atom_features = torch.tensor(atom_features, dtype=torch.long)
    atom_features = labels_to_one_hots(atom_features, x_vocabs, mask_token=True)
    if not edge_indices:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, e_dim), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).T
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
        edge_attr = labels_to_one_hots(edge_attr, e_vocabs)
    
    data = Data(x=atom_features,
                num_nodes=atom_features.size(0),
                num_aam_nodes=len(mapped_features), 
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_edges=edge_index.size(1))
    return data

def build_cgr(reactants_graph, products_graph):
    '''
    Construct a CGR from reactants and products graph.
    '''
    reactants_adj = to_dense_adj(reactants_graph.num_nodes,
                                 reactants_graph.edge_index,
                                 reactants_graph.edge_attr)
    products_adj = to_dense_adj(products_graph.num_nodes,
                                products_graph.edge_index,
                                products_graph.edge_attr)
    # only keep mapped atoms
    N_AAM = reactants_graph.num_aam_nodes
    reactants_adj = reactants_adj[:N_AAM, :N_AAM]
    products_adj = products_adj[:N_AAM, :N_AAM]
    
    cgr_adj = torch.cat([reactants_adj, products_adj], dim=-1)
    cgr_adj_bool = cgr_adj.any(-1)
    cgr_edge_index, _ = dense_to_sparse(cgr_adj_bool)
    cgr = Data(x=torch.empty(N_AAM, 0, dtype=torch.float),
               num_nodes=N_AAM,
               edge_index=cgr_edge_index,
               edge_attr=cgr_adj[cgr_edge_index[0], cgr_edge_index[1]],
               num_edges=cgr_edge_index.size(1))
    return cgr

def build_rxn_graphs(rxn_smiles, max_dist=20, return_cgr=True):
    assert rxn_smiles != '>>', 'Empty mapped reaction'
    rxn = AllChem.ReactionFromSmarts(rxn_smiles)

    reactants = [Chem.MolFromSmiles(Chem.MolToSmiles(reactant), sanitize=False) \
                     for reactant in rxn.GetReactants()]
    products = [Chem.MolFromSmiles(Chem.MolToSmiles(product), sanitize=False) \
                    for product in rxn.GetProducts()]
    match_aams(reactants, products)
    reactants_features = get_mol_features(reactants)
    products_features = get_mol_features(products)
    
    assert len(reactants_features[0]) > 0, 'No mapped atom'
    reactants_graph = build_mol_graph(reactants, *reactants_features)
    compute_shortest_paths(reactants_graph, max_dist)
    products_graph = build_mol_graph(products, *products_features)
    compute_shortest_paths(products_graph, max_dist)
    
    if return_cgr:
        cgr = build_cgr(reactants_graph, products_graph)
        compute_shortest_paths(cgr, max_dist)
        return reactants_graph, products_graph, cgr
    else:
        return reactants_graph, products_graph

def get_reactive_center(reactants_graph, products_graph, cgr):
    N_AAM = reactants_graph.num_aam_nodes

    # changes in edges
    changed_edges = torch.ne(cgr.edge_attr[:, :e_dim],
                             cgr.edge_attr[:, e_dim:]).any(dim=1)
    changed_edges = cgr.edge_index[:, changed_edges]

    # changes in atom properties
    changed_nodes = torch.ne(reactants_graph.x[:N_AAM, len(x_vocabs['atomic_num']):-1],
                             products_graph.x[:N_AAM, len(x_vocabs['atomic_num']):-1]).any(dim=1)
    changed_nodes = torch.cat([changed_nodes.nonzero().squeeze(1),
                               changed_edges.reshape(-1)]).unique()
    
    # assert changed_nodes.size(0) > 0, 'No reactive center'
    return changed_nodes

def build_partial_rxn_graph(rxn_smiles, max_dist=20):
    '''
    For partially observed reactions that only have smiles for reactants or products.
    '''
    molecules = [Chem.MolFromSmiles(mol, sanitize=False) for mol in rxn_smiles.split('.')]
    mol_features = get_mol_features(molecules)
    mol_graph = build_mol_graph(molecules, *mol_features)
    compute_shortest_paths(mol_graph, max_dist)
    return mol_graph
