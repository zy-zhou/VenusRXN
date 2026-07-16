import argparse
from rxnzyme.data.reaction.database import build_rxn_db

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rxn_smiles_path', '-rsp', type=str, default='data/reactions.tsv')
    parser.add_argument('--db_dir', '-db', type=str, default='data/rxn_db')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    build_rxn_db(
        rxn_smiles_path=args.rxn_smiles_path,
        index_col='rxn_id',
        rxn_col='mapped_rxn',
        max_dist=5,
        db_dir=args.db_dir
    )
