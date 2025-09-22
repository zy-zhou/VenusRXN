import pandas as pd
from rxnzyme.data.database import build_rxn_db
from rxnzyme.utils import write_json

if __name__ == '__main__':
    build_rxn_db(
        rxn_smiles_path='data/retrieval_benchmark/test_pairs.tsv',
        index_col='rxn_id',
        rxn_col='mapped_rxn',
        max_dist=5,
        db_dir='data/retrieval_benchmark/rxn_db',
        key_prefix=None,
        overwrite=False
    )

    train_df = pd.read_csv(
        'data/retrieval_benchmark/train_pairs.tsv',
        sep='\t',
        usecols=['enz_id', 'Enzyme Seq']
    )
    test_df = pd.read_csv(
        'data/retrieval_benchmark/test_pairs.tsv',
        sep='\t',
        usecols=['enz_id', 'Enzyme Seq']
    )
    df = pd.concat([train_df, test_df]).drop_duplicates('enz_id').set_index('enz_id')
    enz_dict = df['Enzyme Seq'].to_dict()
    write_json(enz_dict, 'data/retrieval_benchmark/enz_db.json')
