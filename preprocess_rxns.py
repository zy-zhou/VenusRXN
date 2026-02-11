from rxnzyme.data.database import build_rxn_db

if __name__ == '__main__':
    build_rxn_db(
        rxn_smiles_path='data/reactions.tsv',
        index_col='rxn_id',
        rxn_col='mapped_rxn',
        max_dist=5,
        db_dir='data/rxn_db'
    )
