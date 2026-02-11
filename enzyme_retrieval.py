import os
import argparse
import torch
import pandas as pd
from rxnzyme.data.datasets import ignore_label
from rxnzyme.models.loading import get_tokenizer, get_plm, get_prorxn
from rxnzyme.training.base import get_trainer
from rxnzyme.training.prorxn import LitProRxnForMM
from rxnzyme.extractor import plm_mean_pooling, DenseRetriever
from rxnzyme.data_modules.prorxn import ProRxnDataModule
from rxnzyme.data_modules.extractor import DenseRetrieverDataModule
from rxnzyme.utils import read_json, read_fasta, retrieval_metrics, screening_metrics

rxn_graphormer_config = read_json('configs/rxn_graphormer.json')
mol_graphormer_config = rxn_graphormer_config['mol_graphormer']
cgr_graphormer_config = rxn_graphormer_config['cgr_graphormer']
train_config = read_json('configs/prorxn_pretrain.json')

def get_pair_ids(rxn_db_dir, enz_db_path, ids_path):
    pair_ids = pd.read_csv(
        ids_path,
        sep='\t' if ids_path.endswith('.tsv') else ',',
        usecols=['rxn_id', 'enz_id']
    )

    valid_rxn_ids = pd.read_csv(
        os.path.join(rxn_db_dir, 'metadata.csv'),
        index_col='rxn_id',
        usecols=['rxn_id']
    ).index
    valid_enz_ids = read_fasta(enz_db_path).keys()

    pair_ids = pair_ids[pair_ids['rxn_id'].isin(valid_rxn_ids) & pair_ids['enz_id'].isin(valid_enz_ids)]
    return pair_ids

def get_query_ids(train_ids, test_ids):
    test_ids = test_ids.groupby('rxn_id', sort=False).filter(lambda x: len(x) > 1)
    test_ids['in_train'] = test_ids['enz_id'].isin(train_ids['enz_id'].unique())
    test_rxn_groups = dict(list(test_ids.groupby('rxn_id', sort=False)))
    print(f'Number of test reactions with multiple enzymes: {len(test_rxn_groups)}')

    query_ids = []
    for test_rxn_group in test_rxn_groups.values():
        unseen_enzs = test_rxn_group[~test_rxn_group['in_train']]['enz_id']
        if len(unseen_enzs) > 0:
            query_id = unseen_enzs.sample(n=1, random_state=42).iloc[0]
        else:
            query_id = test_rxn_group['enz_id'].sample(n=1, random_state=42).iloc[0]
        query_ids.append(query_id)
        
    query_ids = pd.DataFrame({'rxn_id': list(test_rxn_groups.keys()), 'enz_id': query_ids})
    test_ids.drop(columns=['in_train'], inplace=True)
    return test_ids, query_ids

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rxn_num_layers', '-l', type=int, default=6)
    parser.add_argument('--rxn_hidden_dim', '-d', type=int, default=512)
    parser.add_argument('--plm_name', '-plm', type=str, default='esmc_600m')
    parser.add_argument('--rxn_db_dir', '-rdb', type=str, default='data/rxn_db')
    parser.add_argument('--enz_db_path', '-edb', type=str, default='data/enzymes.fasta')
    parser.add_argument('--train_ids_path', '-tid', type=str, default='data/train_pairs.tsv')
    parser.add_argument('--test_ids_path', '-sid', type=str, default='data/test_pairs.tsv')
    parser.add_argument('--ref_enzymes', '-ref', action='store_true')
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=10)
    parser.add_argument('--precision', '-ps', type=str, default='bf16-true')
    parser.add_argument('--ckpt_path', '-ckpt', type=str, default=None)
    parser.add_argument('--overwrite', '-o', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    mol_graphormer_config['num_encoder_layers'] = args.rxn_num_layers
    mol_graphormer_config['embedding_dim'] = args.rxn_hidden_dim
    mol_graphormer_config['ffn_embedding_dim'] = args.rxn_hidden_dim
    cgr_graphormer_config['feature_dim'] = args.rxn_hidden_dim * 2
    cgr_graphormer_config['embedding_dim'] = args.rxn_hidden_dim
    cgr_graphormer_config['ffn_embedding_dim'] = args.rxn_hidden_dim
    train_config['eval_batch_size'] = args.eval_batch_size
    train_config['gamma'] = 1.0
    train_config['precision'] = args.precision

    train_ids = get_pair_ids(args.rxn_db_dir, args.enz_db_path, args.train_ids_path)
    test_ids = get_pair_ids(args.rxn_db_dir, args.enz_db_path, args.test_ids_path)
    
    if args.ref_enzymes:
        test_ids, test_query_ids = get_query_ids(train_ids, test_ids)
        dm = DenseRetrieverDataModule(
            enz_db_path=args.enz_db_path,
            tokenizer=get_tokenizer(args.plm_name),
            train_config=train_config,
            train_ids=train_ids,
            test_ids=test_ids,
            test_query_ids=test_query_ids
        )
    else:
        dm = ProRxnDataModule(
            rxn_db_dir=args.rxn_db_dir,
            enz_db_path=args.enz_db_path,
            tokenizer=get_tokenizer(args.plm_name),
            train_config=train_config,
            train_ids=train_ids,
            test_ids=test_ids
        )
    
    trainer = get_trainer(train_config)

    if args.ckpt_path:
        pred_path = os.path.join(*os.path.splitext(args.ckpt_path)[0].split('/')[1:])
        pred_path = 'predictions/' + pred_path + ('_ref.pkl' if args.ref_enzymes else '.pkl')
    else:
        assert args.ref_enzymes
        pred_path = 'predictions/{}/{}_ref.pkl'.format(
            args.plm_name,
            args.test_ids_path.split('/')[-2]
        )
    
    if not os.path.exists(pred_path) or args.overwrite: # run prediction
        if args.ckpt_path: # load pretrained prorxn
            prorxn = get_prorxn(
                args.plm_name,
                train_config,
                mol_graphormer_config,
                cgr_graphormer_config,
                pretrained_plm=False
            )
            lit_model = LitProRxnForMM.load_from_checkpoint(args.ckpt_path, prorxn=prorxn)
            if args.ref_enzymes: # use prorxn for enzyme-enzyme retrieval
                lit_model = DenseRetriever(
                    lit_model.model,
                    embed_fn=lambda model, batch: model.encode_enzymes(batch)
                )
        
        else: # load vanilla plm for enzyme-enzyme retrieval
            plm = get_plm(args.plm_name)
            lit_model = DenseRetriever(plm, embed_fn=plm_mean_pooling)

        trainer.test(lit_model, datamodule=dm)
        eval_preds = lit_model.eval_preds
        eval_labels = lit_model.eval_labels

        if trainer.is_global_zero:
            os.makedirs(os.path.dirname(pred_path), exist_ok=True)
            torch.save(eval_preds.cpu(), pred_path)
    
    else:
        lit_model = None
        dm.setup('test')
        eval_preds = torch.load(pred_path)
        eval_labels = dm.test_labels

        if torch.cuda.is_available():
            eval_preds = eval_preds.cuda()
            eval_labels = eval_labels.cuda()

    if trainer.is_global_zero:
        print('--------------------------Retrieval Performance--------------------------')
        for k in (1, 3, 5, 10, 20):
            metrics = retrieval_metrics(eval_preds, eval_labels, k=k, ignore_index=ignore_label)
            print('\t'.join([f'{name}: {value * 100:.2f}%' for name, value in metrics.items()]))
        
        metrics = screening_metrics(eval_preds, eval_labels, ignore_index=ignore_label)
        for name, value in metrics.items():
            if name.startswith('bedroc'):
                print(f'{name}: {value * 100:.2f}%')
            else:
                print(f'{name}: {value:.2f}')
