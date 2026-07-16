import os
import json
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def read_fasta(path):
    records = SeqIO.parse(path, 'fasta')
    seqs = {record.id: str(record.seq).rstrip('*') for record in records}
    return seqs

def write_fasta(seqs, path):
    records = [SeqRecord(Seq(seq), id=seq_id, description='') for seq_id, seq in seqs.items()]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    SeqIO.write(records, path, 'fasta')

def classification_metrics(preds, labels):
    if type(preds) is not torch.Tensor:
        preds = torch.from_numpy(preds)
    if type(labels) is not torch.Tensor:
        labels = torch.from_numpy(labels)
    labels = labels.to(preds.device)
    
    # Calculate accuracy
    acc = torch.sum(labels == preds) / len(labels)
    
    # Calculate confusion matrix
    num_classes = max(labels.max(), preds.max()) + 1
    indices = labels * num_classes + preds
    conf_matrix = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    
    # Calculate confusion entropy (CEN)
    conf_matrix_norm = conf_matrix / conf_matrix.sum(dim=1, keepdim=True)
    conf_matrix_norm[torch.isnan(conf_matrix_norm)] = 0
    
    class_entropy = - torch.sum(conf_matrix_norm * torch.log2(conf_matrix_norm + 1e-10), dim=1)
    class_entropy = class_entropy / torch.log2(num_classes.float())
    cen = class_entropy.mean()
    
    # Calculate MCC
    tp = conf_matrix.diag()
    fp = conf_matrix.sum(0) - tp
    fn = conf_matrix.sum(1) - tp
    tn = conf_matrix.sum() - (tp + fp + fn)
    
    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    denominator[denominator == 0] = 1  # Avoid division by zero
    mcc = numerator / denominator
    mean_mcc = mcc.mean()
    
    return dict(
        acc=acc.item(),
        cen=cen.item(),
        mcc=mean_mcc.item()
    )

def retrieval_metrics(preds, labels, k=20, num_pos=None, reduce=True, ignore_index=-100):
    '''
    Compute retrieval metrics on different queries w.r.t their candidates.

    Args:
        preds: For each query, the predicted scores of all candidates. Shape: [num_src, num_tgt]
        labels: A binary tensor with the same shape as `preds` indicating the true candidates of each query.
        k: The number of top candidates to consider.
        num_pos: The number of positives for each query. If None, this is infered from the labels. Shape: [num_src]
        reduce: Whether to reduce the metrics to a single value.
        ignore_index: When computing the metrics, the candidates with this label value are ignored.
    '''
    if type(preds) is not torch.Tensor:
        preds = torch.from_numpy(preds)
    if type(labels) is not torch.Tensor:
        labels = torch.from_numpy(labels)
    labels = labels.to(preds.device)
    
    label_mask = labels != ignore_index
    preds = preds.where(label_mask, -1000)
    labels = labels.where(label_mask, 0).to(dtype=preds.dtype)

    # sort according to the predicted scores
    indices = preds.topk(k, dim=1).indices
    topk_labels = labels.gather(dim=1, index=indices)

    num_matches = topk_labels.sum(1)
    if num_pos is None:
        num_pos = labels.sum(1)
    else:
        num_pos = num_pos.to(device=preds.device, dtype=preds.dtype)
    success_rate = (num_matches > 0).to(dtype=preds.dtype)
    precision = num_matches / k
    recall = num_matches / num_pos
    if reduce:
        success_rate = success_rate.mean().item()
        precision = precision.mean().item()
        recall = recall.mean().item()

    return {
        f'sr@{k}': success_rate,
        f'acc@{k}': precision,
        f'recall@{k}': recall
    }

def calc_bedroc(preds, labels, alpha=85, reduce=True, ignore_index=-100):
    '''
    Compute BEDROC for already sorted scores and labels.
    '''
    alpha = float(alpha)

    label_mask = labels != ignore_index
    num_mol = label_mask.sum(dim=1, dtype=preds.dtype)
    safe_num_mol = torch.clamp(num_mol, min=1.0)
    active = labels.where(label_mask, 0).to(dtype=preds.dtype)
    num_actives = active.sum(dim=1)

    ranks = torch.arange(1, labels.size(1) + 1, device=labels.device, dtype=preds.dtype)
    weights = torch.exp(-(alpha * ranks.unsqueeze(0)) / safe_num_mol.unsqueeze(1))
    sum_exp = (active * weights).sum(dim=1)

    exp_neg_alpha = torch.exp(torch.tensor(-alpha, device=labels.device, dtype=preds.dtype))
    exp_pos_alpha = torch.exp(torch.tensor(alpha, device=labels.device, dtype=preds.dtype))
    denom = (1.0 / safe_num_mol) * ((1.0 - exp_neg_alpha) / (torch.exp(alpha / safe_num_mol) - 1.0))

    rie = torch.zeros_like(num_actives)
    has_actives = num_actives > 0
    rie[has_actives] = sum_exp[has_actives] / (num_actives[has_actives] * denom[has_actives])

    ratio = num_actives / safe_num_mol
    ratio = ratio.where(has_actives, torch.ones_like(ratio))
    rie_max = (1.0 - torch.exp(-alpha * ratio)) / (ratio * (1.0 - exp_neg_alpha))
    rie_min = (1.0 - torch.exp(alpha * ratio)) / (ratio * (1.0 - exp_pos_alpha))

    bedroc = torch.zeros_like(num_actives)
    normal = has_actives & (rie_max != rie_min)
    bedroc[normal] = (rie[normal] - rie_min[normal]) / (rie_max[normal] - rie_min[normal])
    bedroc[has_actives & ~normal] = 1.0

    return bedroc.mean().item() if reduce else bedroc

def calc_enrichment(preds, labels, fraction=0.02, reduce=True, normalize=False, ignore_index=-100):
    '''
    Compute enrichment factor for already sorted scores and labels.
    '''
    label_mask = labels != ignore_index
    num_mol = label_mask.sum(dim=1, dtype=preds.dtype)
    active = labels.where(label_mask, 0).to(dtype=preds.dtype)
    num_actives = active.sum(dim=1)
    num_top = torch.ceil(num_mol * fraction)
    ranks = torch.arange(1, labels.size(1) + 1, device=labels.device, dtype=preds.dtype)
    top_mask = ranks.unsqueeze(0) <= num_top.unsqueeze(1)
    num_top_actives = active.where(top_mask, 0).sum(dim=1)

    ef = torch.zeros_like(num_actives)
    if normalize:
        max_top_actives = torch.minimum(num_actives, num_top)
        has_actives = max_top_actives > 0
        ef[has_actives] = num_top_actives[has_actives] / max_top_actives[has_actives]
    else:
        has_actives = (num_actives > 0) & (num_top > 0)
        ef[has_actives] = (
            num_top_actives[has_actives]
            * num_mol[has_actives]
            / (num_top[has_actives] * num_actives[has_actives])
        )

    return ef.mean().item() if reduce else ef

def screening_metrics(preds, labels, alpha=85, fraction=0.02, normalize=False, reduce=True, ignore_index=-100):
    '''
    Compute virtual screening metrics on different queries w.r.t their candidates.

    Args:
        preds: For each query, the predicted scores of all candidates. Shape: [num_src, num_tgt]
        labels: A binary tensor with the same shape as `preds` indicating the true candidates of each query.
        alpha: Parameter for BEDROC.
        fraction: Parameter for enrichment factor.
        normalize: Whether to normalize EF by its theoretical maximum.
        reduce: Whether to reduce the metrics to a single value.
        ignore_index: When computing the metrics, the candidates with this label value are ignored.
    '''
    if type(preds) is not torch.Tensor:
        preds = torch.from_numpy(preds)
    if type(labels) is not torch.Tensor:
        labels = torch.from_numpy(labels)
    labels = labels.to(preds.device)
    
    label_mask = labels != ignore_index
    preds = preds.where(label_mask, -1000)
    
    # sort according to the predicted scores
    preds, indices = preds.sort(dim=1, descending=True)
    labels = labels.gather(dim=1, index=indices)
    
    bedroc = calc_bedroc(preds, labels, alpha, reduce, ignore_index)
    ef = calc_enrichment(preds, labels, fraction, reduce, normalize, ignore_index)

    return {
        f'bedroc@{alpha}': bedroc,
        f'ef@{fraction}': ef
    }

def ranking_metrics(preds, labels):
    if type(preds) is torch.Tensor:
        preds = preds.double().numpy(force=True)
    if type(labels) is torch.Tensor:
        labels = labels.double().numpy(force=True)
    
    return {
        'spearman': spearmanr(preds, labels).statistic,
        'pearson': pearsonr(preds, labels).statistic,
        'mse': mean_squared_error(labels, preds),
        'r2': r2_score(labels, preds)
    }
