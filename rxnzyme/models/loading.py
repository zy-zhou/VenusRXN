import re
import torch
from transformers import EsmConfig, EsmTokenizer
from peft import LoraConfig, inject_adapter_in_model
from esm.tokenization import get_esmc_model_tokenizers
from .modules import GraphormerGraphEncoder
from .graphormer import RxnGraphormer
from .prorxn import ProRxn
from .esm import EsmModel
from .esmc import ESMC

plm_dirs = {
    'esm1b': 'facebook/esm1b_t33_650M_UR50S',
    'esm1v-1': 'facebook/esm1v_t33_650M_UR90S_1',
    'esm1v-2': 'facebook/esm1v_t33_650M_UR90S_2',
    'esm1v-3': 'facebook/esm1v_t33_650M_UR90S_3',
    'esm1v-4': 'facebook/esm1v_t33_650M_UR90S_4',
    'esm1v-5': 'facebook/esm1v_t33_650M_UR90S_5',
    'esm2': 'facebook/esm2_t33_650M_UR50D',
    'esm2-35m': 'facebook/esm2_t12_35M_UR50D',
    'esm2-150m': 'facebook/esm2_t30_150M_UR50D',
    'esm2-3b': 'facebook/esm2_t36_3B_UR50D'
}

graphormer_lora_modules = ['q_proj', 'k_proj', 'v_proj']
esm_lora_modules = ['query', 'key', 'value']
esmc_lora_modules = ['layernorm_qkv.1', 'layernorm_q.1', 'kv']

def reduce_lit_ckpt(ckpt_path, module_name='model'):
    if not torch.cuda.is_available():
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    else:
        ckpt = torch.load(ckpt_path)
    prefix = f'{module_name}.'
    weights = {
        k.replace(prefix, ''): v for k, v in ckpt['state_dict'].items() if k.startswith(prefix)
    }
    return weights

def get_tokenizer(plm_name):
    if plm_name.startswith('esmc'):
        tokenizer = get_esmc_model_tokenizers()
    else:
        tokenizer = EsmTokenizer.from_pretrained(plm_dirs[plm_name])
    return tokenizer

def get_plm(
        plm_name,
        add_cross_attn=False,
        d_cross_attn=None,
        pretrained=True,
        grad_ckpt=False
    ):
    if plm_name.startswith('esmc'):
        d_model, n_heads, n_layers = (960, 15, 30) if plm_name == 'esmc_300m' else (1152, 18, 36)
        
        if pretrained:
            plm = ESMC.from_pretrained(
                plm_name,
                n_layers_cross_attn=n_layers // 2 if add_cross_attn else 0,
                d_cross_attn=d_cross_attn,
                add_pooling_layer=add_cross_attn
            )
        else:
            plm = ESMC(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                tokenizer=get_esmc_model_tokenizers(),
                n_layers_cross_attn=n_layers // 2 if add_cross_attn else 0,
                d_cross_attn=d_cross_attn,
                add_pooling_layer=add_cross_attn
            )
    
    else:
        config = EsmConfig.from_pretrained(
            plm_dirs[plm_name],
            add_cross_attention=add_cross_attn,
            cross_attention_hidden_size=d_cross_attn
        )
        if pretrained:
            plm = EsmModel.from_pretrained(
                plm_dirs[plm_name],
                config=config,
                add_pooling_layer=add_cross_attn
            )
        else:
            plm = EsmModel(config, add_pooling_layer=add_cross_attn)
        if grad_ckpt:
            plm.gradient_checkpointing_enable({'use_reentrant': False})

    return plm

def get_prorxn(
    plm_name,
    prorxn_config,
    mg_config,
    cg_config=None,
    grad_ckpt=False,
    pretrained_plm=True,
    mg_ckpt_path=None,
    ckpt_path=None
):
    '''
    Create a non-lightning `ProRxn` with specified configs and optionally load the pretrained weights.

    Args:
        mg_config: Mol-Graphormer config.
        cg_config: CGR-Graphormer config. `None` means not to use CGR in RXN-Graphormer. 
        rg_pooling: Pooling method for RXN-Graphormer when not using CGR-Graphormer.
        grad_ckpt: Whether to use gradient checkpointing. Only works for ESM-1 and 2.
        pretrained_plm: Whether to load the pretrained PLM weights.
        mg_ckpt_path: Path to the lightning checkpoint of `LitMolGraphormerForMLM`.
        ckpt_path: Path to the lightning checkpoint of `LitProRxnForMM`.
    '''
    plm = get_plm(
        plm_name,
        add_cross_attn=prorxn_config['gamma'] > 0,
        d_cross_attn=mg_config['embedding_dim'] if cg_config is None else cg_config['embedding_dim'],
        pretrained=pretrained_plm and not ckpt_path,
        grad_ckpt=grad_ckpt
    )
    
    mol_graphormer = GraphormerGraphEncoder(**mg_config)
    # load pretrained mol graphormer if provided
    if mg_ckpt_path and not ckpt_path:
        weights = reduce_lit_ckpt(mg_ckpt_path, module_name='model.graphormer')
        mol_graphormer.load_state_dict(weights)
    cgr_graphormer = GraphormerGraphEncoder(**cg_config) if cg_config is not None else None
    rxn_graphormer = RxnGraphormer(mol_graphormer, cgr_graphormer)
    
    prorxn = ProRxn(
        rxn_graphormer,
        plm,
        proj_dim=prorxn_config['proj_dim'],
        proj_type=prorxn_config['proj_type']
    )
    # load pretrained prorxn if provided
    if ckpt_path:
        weights = reduce_lit_ckpt(ckpt_path)
        prorxn.load_state_dict(weights)
    
    return prorxn

def inject_lora_to_prorxn(ltr_prorxn, r=16, dropout=0.1, trainable_heads=True):
    '''
    Inject LoRA parameters in-place into a `ProRxnForLTR`.

    Args:
        ltr_prorxn: Instance of `ProRxnForLTR`.
        trainable_heads: Whether to set the output layers trainable.
    '''
    plm_lora_modules = esmc_lora_modules if type(ltr_prorxn.prorxn.plm) is ESMC else esm_lora_modules
    target_modules = graphormer_lora_modules + plm_lora_modules
    lora_config = LoraConfig(
        r=r,
        target_modules=target_modules,
        lora_alpha=r,
        lora_dropout=dropout,
        bias='none'
    )
    inject_adapter_in_model(lora_config, ltr_prorxn)

    if not trainable_heads:
        return
    
    # manually unfreeze the heads since modules_to_save will not work
    pattern = r'^(prorxn\.(plm\.pooler|out_proj|rxn_proj|enz_proj)|ranking_head|t$)'
    for name, param in ltr_prorxn.named_parameters():
        if re.match(pattern, name):
            param.requires_grad = True
