import os
import argparse
import torch
import pandas as pd
from rxnzyme.data.datasets.base import ignore_label
from rxnzyme.data.datasets.graphormer import is_partial_rxns
from rxnzyme.models.loading import get_tokenizer, get_plm, get_prorxn
from rxnzyme.training.base import get_trainer
from rxnzyme.training.prorxn import LitProRxnForMM
from rxnzyme.extractor import plm_mean_pooling, Enzyme2EnzymeRetriever
from rxnzyme.data.modules.prorxn import ProRxnDataModule
from rxnzyme.data.modules.extractor import Enzyme2EnzymeDataModule
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
    enz_ext = os.path.splitext(enz_db_path)[1]
    if enz_ext in {'.fasta', '.faa', '.fa'}:
        valid_enz_ids = read_fasta(enz_db_path).keys()
    elif enz_ext == '.json':
        valid_enz_ids = read_json(enz_db_path).keys()
    else:
        raise ValueError('Enzyme database must be a fasta or json file.')

    pair_ids = pair_ids[pair_ids['rxn_id'].isin(valid_rxn_ids) & pair_ids['enz_id'].isin(valid_enz_ids)]
    return pair_ids

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plm_name', '-plm', type=str, default='esmc_600m')
    parser.add_argument('--rxn_db_dir', '-rdb', type=str, default='data/rxn_db')
    parser.add_argument('--enz_db_path', '-edb', type=str, default='data/enzymes.fasta')
    parser.add_argument('--train_ids_path', '-tid', type=str)
    parser.add_argument('--test_ids_path', '-sid', type=str)
    parser.add_argument('--ref_enzymes', '-ref', action='store_true')
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=80)
    parser.add_argument('--precision', '-ps', type=str, default='bf16-true')
    parser.add_argument('--ckpt_path', '-ckpt', type=str, default=None)
    parser.add_argument('--overwrite', '-o', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train_config['eval_batch_size'] = args.eval_batch_size
    train_config['precision'] = args.precision

    train_ids = get_pair_ids(args.rxn_db_dir, args.enz_db_path, args.train_ids_path)
    test_ids = get_pair_ids(args.rxn_db_dir, args.enz_db_path, args.test_ids_path)
    
    if args.ref_enzymes:
        test_query_ids = pd.read_csv(
            os.path.join(os.path.split(args.test_ids_path)[0], 'templates.csv')
        )
        test_ids = test_ids[test_ids['rxn_id'].isin(test_query_ids['rxn_id'])]
        dm = Enzyme2EnzymeDataModule(
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

    trainer = None

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
                train_config,
                mol_graphormer_config,
                cgr_graphormer_config,
                partial_rxns=is_partial_rxns(args.rxn_db_dir),
                pretrained_plm=False
            )
            lit_model = LitProRxnForMM.load_from_checkpoint(
                args.ckpt_path, prorxn=prorxn, train_config=train_config
            )
            if args.ref_enzymes: # use prorxn for enzyme-enzyme retrieval
                lit_model = Enzyme2EnzymeRetriever(
                    lit_model.model,
                    embed_fn=lambda model, batch: model.encode_enzymes(batch)
                )
        
        else: # load vanilla plm for enzyme-enzyme retrieval
            plm = get_plm(args.plm_name)
            lit_model = Enzyme2EnzymeRetriever(plm, embed_fn=plm_mean_pooling)

        trainer = get_trainer(train_config)
        trainer.test(lit_model, datamodule=dm)
        eval_preds = lit_model.eval_preds
        eval_labels = lit_model.eval_labels

        if trainer.is_global_zero:
            os.makedirs(os.path.dirname(pred_path), exist_ok=True)
            torch.save(eval_preds, pred_path)
    
    else:
        dm.setup('test')
        eval_preds = torch.load(pred_path)
        eval_labels = dm.test_labels

    if trainer is None or trainer.is_global_zero:
        if torch.cuda.is_available() and '_extended' not in args.enz_db_path:
            eval_preds = eval_preds.cuda()
            eval_labels = eval_labels.cuda()
        
        print('--------------------------Retrieval Performance--------------------------')
        for k in (1, 3, 5, 10, 20):
            metrics = retrieval_metrics(eval_preds, eval_labels, k=k, ignore_index=ignore_label)
            print('\t'.join([f'{name}: {value * 100:.2f}%' for name, value in metrics.items()]))
        
        metrics = screening_metrics(
            eval_preds, eval_labels, alpha=160, fraction=0.001, ignore_index=ignore_label
        )
        for name, value in metrics.items():
            print(f'{name}: {value:.3f}')
