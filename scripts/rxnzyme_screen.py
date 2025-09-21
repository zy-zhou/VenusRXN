import os
import argparse
import torch
import pandas as pd
from rxnzyme.data.datasets import ignore_label
from rxnzyme.models.loading import get_tokenizer, get_plm, get_prorxn
from rxnzyme.training.base import get_trainer
from rxnzyme.training.prorxn import LitProRxnForMM
from rxnzyme.training.extractor import plm_mean_pooling, DenseRetriever
from rxnzyme.data_modules.prorxn import ProRxnDataModule
from rxnzyme.data_modules.extractor import DenseRetrieverDataModule
from rxnzyme.utils import read_json, retrieval_metrics, screening_metrics
from ..prorxn_pretrain import get_pair_ids

rxn_graphormer_config = read_json('configs/rxn_graphormer.json')
mol_graphormer_config = rxn_graphormer_config['mol_graphormer']
cgr_graphormer_config = rxn_graphormer_config['cgr_graphormer']
train_config = read_json('configs/prorxn_pretrain.json')

def get_query_ids(train_ids, test_ids):
    train_rxn_groups = dict(list(train_ids.groupby('rxn_id', sort=False)['enz_id']))
    test_rxn_groups = dict(list(test_ids.groupby('rxn_id', sort=False)['enz_id']))
    query_ids = [
        train_rxn_groups[rxn_id].iloc[0] if rxn_id in train_rxn_groups.keys() else test_rxn_group.iloc[0] \
            for rxn_id, test_rxn_group in test_rxn_groups.items()
    ]
    query_ids = pd.DataFrame({'rxn_id': list(test_rxn_groups.keys()), 'enz_id': query_ids})
    return query_ids

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rxn_num_layers', '-l', type=int, default=6)
    parser.add_argument('--rxn_hidden_dim', '-d', type=int, default=512)
    parser.add_argument('--plm_name', '-plm', type=str, default='esmc_600m')
    parser.add_argument('--cross_attn', '-ca', action='store_true')
    parser.add_argument('--rxn_db_dir', '-rdb', type=str, default='data/all_reactions')
    parser.add_argument('--enz_db_path', '-edb', type=str, default='data/pair_merged_data/enzyme_db.json')
    parser.add_argument('--train_ids_path', '-tid', type=str, default='data/pair_merged_data/rxn_split/train_reactions.tsv')
    parser.add_argument('--test_ids_path', '-sid', type=str, default='data/pair_merged_data/rxn_split/test_reactions.tsv')
    parser.add_argument('--ref_enzymes', '-ref', action='store_true')
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=200)
    parser.add_argument('--num_candidates', '-k', type=int, default=0)
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
    train_config['gamma'] = 1.0 if args.cross_attn else 0.0
    train_config['precision'] = args.precision

    train_ids = get_pair_ids(args.rxn_db_dir, args.enz_db_path, args.train_ids_path)
    test_ids = get_pair_ids(args.rxn_db_dir, args.enz_db_path, args.test_ids_path)
    
    if args.ref_enzymes:
        dm = DenseRetrieverDataModule(
            enz_db_path=args.enz_db_path,
            tokenizer=get_tokenizer(args.plm_name),
            train_config=train_config,
            train_ids=train_ids,
            test_ids=test_ids,
            test_query_ids=get_query_ids(train_ids, test_ids)
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

    if args.num_candidates > 0 and not args.ref_enzymes:
        assert args.cross_attn, 'Cross attention should be used for ranking.'
        if lit_model is None:
            prorxn = get_prorxn(
                args.plm_name,
                train_config,
                mol_graphormer_config,
                cgr_graphormer_config,
                pretrained_plm=False
            )
            lit_model = LitProRxnForMM.load_from_checkpoint(args.ckpt_path, prorxn=prorxn)

        eval_preds = eval_preds.where(eval_labels != ignore_label, -1000)
        eval_labels = eval_labels.where(eval_labels != ignore_label, 0)
        topk_indices = eval_preds.topk(args.num_candidates, dim=1).indices
        num_pos = eval_labels.sum(1)
        dm.test_cdts = topk_indices.cpu()
        
        trainer.test(lit_model, datamodule=dm)
        eval_preds = lit_model.eval_preds
        eval_labels = lit_model.eval_labels

        if trainer.is_global_zero:
            print('--------------------------Ranking Performance--------------------------')
            for k in (1, 3, 5, 10, 20):
                if k > args.num_candidates:
                    break
                metrics = retrieval_metrics(eval_preds, eval_labels, k=k, num_pos=num_pos)
                print('\t'.join([f'{name}: {value * 100:.2f}%' for name, value in metrics.items()]))
