import os
import sys
import re
import pandas as pd
from tqdm import tqdm
from rxnmapper import RXNMapper, BatchedMapper
from .enzymemap import helpers_map, helpers_rdkit

dir_name = os.path.dirname(os.path.abspath(__file__))
if dir_name not in sys.path:
    sys.path.insert(0, dir_name)
rules_path = os.path.join(dir_name, 'enzymemap', 'rules.pkl')

def std_rxn_smiles(rxn_smiles):
    std_parts = []
    for part in rxn_smiles.split('>>'): # may be partial reaction
        std_mols = []
        for mol in part.split('.'):
            smi = helpers_rdkit.get_smi(mol)
            if smi:
                std_mols.append(smi)
            else:
                print(f'Warning: invalid SMILES: {mol}')
        std_parts.append('.'.join(sorted(std_mols)))
    return '>>'.join(std_parts)

def run_rxnmapper(rxn_smiles, batch_size=None):
    if batch_size is None:
        rxn_mapper = RXNMapper()
        results = rxn_mapper.get_attention_guided_atom_maps(rxn_smiles)
    else:
        rxn_mapper = BatchedMapper(batch_size=batch_size)
        # results as strings
        results = []
        for mapped_rxn in tqdm(
            rxn_mapper.map_reactions(rxn_smiles), total=len(rxn_smiles), desc='Running RXNMapper...'
        ):
            results.append(mapped_rxn)
        # results as dictionaries
        # results = list(rxn_mapper.map_reactions_with_info(rxn_smiles))
    return results

def is_unbalanced_or_unmapped(rxn_smiles):
    if rxn_smiles == '>>' or not rxn_smiles:
        return True
    
    reactants, products = rxn_smiles.split('>>')
    # get the atom mapping numbers of reactants and products
    reactant_maps = set(int(m.group(1)) for m in re.finditer(r':(\d+)', reactants))
    product_maps = set(int(m.group(1)) for m in re.finditer(r':(\d+)', products))
    if reactant_maps != product_maps or len(reactant_maps) == 0:
        return True
    
    return False

def fix_unbalanced(df, rules, rxn_col, src_rxn_col):
    '''
    Fix unbalanced or unmapped reactions.
    modified from enzymemap.map_group()

    Args:
        df: dataframe with initially mapped reactions to be fixed
        rules: dataframe of rules to use for mapping
        rxn_col: column name of initially mapped reactions
        src_rxn_col: column name of raw reactions
    '''
    rxns_for_templates = df[rxn_col].dropna().tolist()
    if len(rxns_for_templates) == 0:
        return df
    
    templates, temp2h, temp2reac, template_s = \
        helpers_map.make_templates_for_suggestions(rxns_for_templates)

    # Per entry: suggest corrections -> map -> select best option -> write back
    num_broken = 0
    num_fixed = 0
    for i in tqdm(df.index, desc='Suggesting reactions...'):
        if not is_unbalanced_or_unmapped(df.loc[i, rxn_col]):
            continue

        num_broken += 1
        try:
            suggested_rxns = helpers_map.suggest_corrections(
                df.loc[i, src_rxn_col], templates, temp2h, temp2reac, template_s
            )
        except:
            suggested_rxns = []

        if len(suggested_rxns) == 0:
            continue
        suggested_rxns, suggested_rules, rule_ids, individuals = helpers_map.map(
            suggested_rxns, rules, single=True
        )

        if len(suggested_rxns) == 0:
            continue
        best_rxn, _, _, _ = helpers_rdkit.select_best(
            suggested_rxns, suggested_rules, rule_ids, individuals
        )
        df.loc[i, rxn_col] = best_rxn
        num_fixed += 1
    
    print(f'Number of unbalanced or unmapped reactions: {num_broken}')
    print(f'Number of reactions fixed by EnzymeMap: {num_fixed}')
    return df

def map_and_fix_rxns(
    rxn_smiles_path,
    rxn_col,
    src_rxn_col,
    batch_size=32,
    standardize=False,
    deduplicate=False,
    save_path=None
):
    '''
    Map and fix reactions using RXNMapper and EnzymeMap.
    
    Args:
        rxn_smiles_path: path to the reaction smiles file
        rxn_col: column name of the processed reactions (to add)
        src_rxn_col: column name of the raw reactions
        batch_size: batch size for RXNMapper
        standardize: whether to standardize the raw reactions before processing
        deduplicate: whether to deduplicate the processed reactions
        save_path: path to save the merged file
    '''
    df = pd.read_csv(
        rxn_smiles_path,
        sep='\t' if rxn_smiles_path.endswith('.tsv') else ','
    )
    if standardize:
        df[src_rxn_col] = df[src_rxn_col].apply(std_rxn_smiles)
    if deduplicate:
        df = df.drop_duplicates(subset=[src_rxn_col])
    if rxn_col not in df.columns:
        df[rxn_col] = None
    
    print('Mapping and fixing reactions...')
    print(f'Total reactions: {len(df)}')
    
    # map unmapped reactions
    mask = df[rxn_col].isna() | (df[rxn_col] == '>>')
    if mask.any():
        unmapped_rxns = df.loc[mask, src_rxn_col].tolist()
        mapped_rxns = run_rxnmapper(unmapped_rxns, batch_size=batch_size)
        df.loc[mask, rxn_col] = mapped_rxns
    mask = df[rxn_col].isna() | (df[rxn_col] == '>>')
    df.loc[mask, rxn_col] = None

    # fix unbalanced reactions after mapping, using the rules from enzymemap
    rules = pd.read_pickle(rules_path)
    df = fix_unbalanced(df, rules, rxn_col, src_rxn_col)

    # remove reactions that are still unmapped
    mask = df[rxn_col].isna() | (df[rxn_col] == '>>')
    df = df[~mask]
    print(f'Mapped reactions: {len(df)}')
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(
            save_path,
            sep='\t' if save_path.endswith('.tsv') else ',',
            index=False
        )
    return df
