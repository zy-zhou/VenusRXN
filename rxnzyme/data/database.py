import os
import torch
import lmdb
import pickle
import pandas as pd
from tqdm import tqdm
from math import ceil
from collections import Counter
from .reaction import (
    x_vocabs,
    x_dim,
    e_dim,
    one_hots_to_labels,
    build_rxn_graphs,
    get_reactive_center,
    build_partial_rxn_graph
)

def matrix_to_triu(matrix):
    assert torch.all(matrix == matrix.T), 'Cannot compress asymmetric matrix using triu'
    rows, cols = torch.triu_indices(*matrix.size(), device=matrix.device)
    return matrix[rows, cols]

def triu_to_matrix(triu, N):
    rows, cols = torch.triu_indices(N, N, device=triu.device)
    matrix = torch.zeros((N, N), dtype=triu.dtype, device=triu.device)
    matrix[rows, cols] = triu
    matrix[cols, rows] = triu
    return matrix

def compress_int_tensor(tensor):
    min_value, max_value = tensor.min(), tensor.max()
    if min_value >= 0 and max_value <= 255:
        return tensor.to(dtype=torch.uint8)
    elif min_value >= -128 and max_value <= 127:
        return tensor.to(dtype=torch.int8)
    else:
        return tensor.short()

def compress_mol_graph(graph):
    if graph.x.numel() > 0:
        graph.x = compress_int_tensor(graph.x.nonzero())
    
    if graph.edge_attr.numel() > 0:
        graph.edge_attr = compress_int_tensor(graph.edge_attr.nonzero())
        graph.edge_index = compress_int_tensor(graph.edge_index)
        graph.path_indices = compress_int_tensor(graph.path_indices)
    
    graph.spatial_pos = compress_int_tensor(matrix_to_triu(graph.spatial_pos))
    return graph

def decompress_mol_graph(graph):
    if graph.x.numel() > 0:
        x = torch.zeros((graph.num_nodes, x_dim), dtype=torch.float)
        indices = graph.x.long()
        x[indices[:, 0], indices[:, 1]] = 1
        graph.x = x
        e_dim_ = e_dim
    else:
        e_dim_ = e_dim * 2
    
    if graph.edge_attr.numel() > 0:
        edge_attr = torch.zeros((graph.num_edges, e_dim_), dtype=torch.float)
        indices = graph.edge_attr.long()
        edge_attr[indices[:, 0], indices[:, 1]] = 1
        graph.edge_attr = edge_attr
        graph.edge_index = graph.edge_index.long()
        graph.path_indices = graph.path_indices.long()

    graph.spatial_pos = triu_to_matrix(graph.spatial_pos, graph.num_nodes).long()
    return graph

def get_db_size(db_dir):
    with lmdb.open(db_dir, readonly=True, lock=False) as env:
        map_size = env.info()['map_size']
    return map_size

def build_rxn_db(
        rxn_smiles_path,
        index_col,
        rxn_col,
        max_dist=20,
        db_dir='data/reaction_db',
        key_prefix=None,
        bytes_per_rxn=180000,
        chunk_size=100000,
        overwrite=True
    ):
    sep = '\t' if  rxn_smiles_path.endswith('.tsv') else ','
    all_rxns = pd.read_csv(
        rxn_smiles_path,
        sep=sep,
        index_col=index_col,
        usecols=[index_col, rxn_col] if index_col is not None else [rxn_col]
    )
    all_rxns = all_rxns[~all_rxns.index.duplicated()]
    all_rxns = all_rxns[rxn_col]
    is_partial = all_rxns.iloc[0].find('>>') == -1
    map_size = len(all_rxns) * bytes_per_rxn
    num_chunks = ceil(len(all_rxns) / chunk_size)
    if key_prefix:
        all_rxns.index = key_prefix + all_rxns.index.astype(str)
    
    os.makedirs(db_dir, exist_ok=True)
    data_path = os.path.join(db_dir, 'data.mdb')
    lock_path = os.path.join(db_dir, 'lock.mdb')
    metadata_path = os.path.join(db_dir, 'metadata.csv')
    if os.path.exists(data_path):
        if overwrite:
            os.remove(data_path)
            os.remove(lock_path)
        else:
            exist_map_size = get_db_size(db_dir)
            map_size += exist_map_size
    
    metadata = []
    env = lmdb.open(db_dir, map_size=map_size)
    print('Building reaction database...')

    for i in range(0, len(all_rxns), chunk_size):
        chunk = all_rxns.iloc[i: i + chunk_size]

        with env.begin(write=True) as txn:
            for rxn_id, rxn_smiles in tqdm(
                chunk.items(),
                desc=f'Processing chunk {i + 1} of {num_chunks}',
                total=len(chunk)
            ):
                if not is_partial:
                    try:
                        reactants_graph, products_graph, cgr = build_rxn_graphs(rxn_smiles, max_dist)
                        reactive_center = get_reactive_center(reactants_graph, products_graph, cgr)
                    except Exception as e:
                        print(f'Error processing reaction {rxn_id}: {e}')
                        continue
                    metadata.append(
                        dict(
                            rxn_id=rxn_id,
                            num_atoms=reactants_graph.num_nodes,
                            num_bonds=reactants_graph.num_edges,
                            path_length=cgr.path_indices.size(1),
                            center_size=reactive_center.size(0)
                        )
                    )
                    rxn_graphs = dict(
                        reactants=compress_mol_graph(reactants_graph),
                        products=compress_mol_graph(products_graph),
                        cgr=compress_mol_graph(cgr),
                        reactive_center=compress_int_tensor(reactive_center) \
                            if reactive_center.numel() > 0 else reactive_center
                    )
                else:
                    try:
                        mol_graph = build_partial_rxn_graph(rxn_smiles, max_dist)
                    except Exception as e:
                        print(f'Error processing reaction {rxn_id}: {e}')
                        continue
                    metadata.append(
                        dict(
                            rxn_id=rxn_id,
                            num_atoms=mol_graph.num_nodes,
                            num_bonds=mol_graph.num_edges,
                            path_length=mol_graph.path_indices.size(1),
                        )
                    )
                    rxn_graphs = compress_mol_graph(mol_graph)

                txn.put(str(rxn_id).encode(), pickle.dumps(rxn_graphs))

    env.close()
    metadata = pd.DataFrame(metadata).set_index('rxn_id')
    errors = len(all_rxns) - len(metadata)
    print(f'Done. Number of error reactions: {errors} ({errors / len(all_rxns) * 100:.2f}%)')

    if os.path.exists(metadata_path) and not overwrite:
        exist_metadata = pd.read_csv(metadata_path, index_col='rxn_id')
        metadata = pd.concat([exist_metadata, metadata])
    metadata.to_csv(metadata_path)

def count_labels(db_dir, attr_combs):
    rxn_ids = pd.read_csv(
        os.path.join(db_dir, 'metadata.csv'),
        index_col='rxn_id',
        usecols=['rxn_id']
    ).index

    counters = [Counter() for _ in attr_combs]
    attr_combs = [torch.tensor(attr_comb, dtype=torch.long) for attr_comb in attr_combs]
    map_size = get_db_size(db_dir)
    env = lmdb.open(db_dir, map_size=map_size, readonly=True, lock=False)

    with env.begin() as txn:
        for rxn_id in tqdm(rxn_ids, desc='Counting tokens'):
            rxn_graphs = txn.get(str(rxn_id).encode())
            rxn_graphs = pickle.loads(rxn_graphs)
            
            graph = decompress_mol_graph(rxn_graphs['reactants'])
            r_x = one_hots_to_labels(graph.x, x_vocabs)
            graph = decompress_mol_graph(rxn_graphs['products'])
            p_x = one_hots_to_labels(graph.x, x_vocabs)
            x = torch.cat([r_x, p_x])

            for i, attr_comb in enumerate(attr_combs):
                tokens = x[:, attr_comb].tolist()
                counters[i].update([tuple(token) for token in tokens])

    env.close()
    print('Done.')

    label_counts = []
    for attr_comb, counter in zip(attr_combs, counters):
        label_counts.append(dict(attr_comb=attr_comb, count=counter))
        print(f'Unique labels for attribute combination {attr_comb.tolist()}: {len(counter)}')
    torch.save(label_counts, os.path.join(db_dir, 'label_counts.pkl'))
    return counters
