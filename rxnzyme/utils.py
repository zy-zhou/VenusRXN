import os
import json
import torch
from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment # type: ignore
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

def pairwise_identity(seq1, seq2):
    best_aln = pairwise2.align.globalxx(seq1, seq2)[0]
    matches = sum(a == b for a, b in zip(best_aln.seqA, best_aln.seqB))
    return matches / len(best_aln.seqA)

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
    confusion_matrix = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    
    # Calculate confusion entropy (CEN)
    conf_matrix_norm = confusion_matrix / confusion_matrix.sum(dim=1, keepdim=True)
    conf_matrix_norm[torch.isnan(conf_matrix_norm)] = 0
    
    class_entropy = - torch.sum(conf_matrix_norm * torch.log2(conf_matrix_norm + 1e-10), dim=1)
    class_entropy = class_entropy / torch.log2(num_classes.float())
    cen = class_entropy.mean()
    
    # Calculate MCC
    tp = confusion_matrix.diag()
    fp = confusion_matrix.sum(0) - tp
    fn = confusion_matrix.sum(1) - tp
    tn = confusion_matrix.sum() - (tp + fp + fn)
    
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

def retrieval_metrics(preds, labels, k=20, num_pos=None, ignore_index=-100):
    '''
    Compute retrieval metrics on different queries w.r.t their candidates.

    Args:
        preds: For each query, the predicted scores of all candidates. Shape: [num_src, num_tgt]
        labels: A binary tensor with the same shape as `preds` indicating the true candidates of each query.
        k: The number of top candidates to consider.
        num_pos: The number of positives for each query. If None, this is infered from the labels. Shape: [num_src]
        ignore_index: When computing the metrics, the candidates with this label value are ignored.
    '''
    if type(preds) is not torch.Tensor:
        preds = torch.from_numpy(preds)
    if type(labels) is not torch.Tensor:
        labels = torch.from_numpy(labels)
    labels = labels.to(preds.device)
    
    label_mask = labels != ignore_index
    preds = preds.where(label_mask, -1000)
    labels = labels.where(label_mask, 0)

    # sort according to the predicted scores
    indices = preds.topk(k, dim=1).indices
    topk_labels = labels.gather(dim=1, index=indices)

    num_matches = topk_labels.sum(1)
    num_pos = labels.sum(1) if num_pos is None else num_pos
    success_rate = torch.sum(num_matches > 0) / num_matches.size(0)
    precision = torch.mean(num_matches / k)
    recall = torch.mean(num_matches / num_pos)

    return {
        f'sr@{k}': success_rate.item(),
        f'acc@{k}': precision.item(),
        f'recall@{k}': recall.item()
    }

def screening_metrics(preds, labels, alpha=85, fraction=0.02, ignore_index=-100):
    '''
    Compute virtual screening metrics on different queries w.r.t their candidates.

    Args:
        preds: For each query, the predicted scores of all candidates. Shape: [num_src, num_tgt]
        labels: A binary tensor with the same shape as `preds` indicating the true candidates of each query.
        alpha: Parameter for BEDROC.
        fraction: Parameter for enrichment factor.
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
    
    bedroc, ef = [], []
    scores = torch.stack([preds, labels], dim=-1).double().numpy(force=True) # num_src, num_tgt, 2
    num_cdts = label_mask.sum(1).tolist()
    for i, n in enumerate(num_cdts):
        bedroc.append(CalcBEDROC(scores[i][:n], col=1, alpha=alpha))
        ef.append(CalcEnrichment(scores[i][:n], col=1, fractions=[fraction])[0])
    N = preds.size(0)
    bedroc = sum(bedroc) / N
    ef = sum(ef) / N

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
