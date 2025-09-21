import os
import torch
import torch.distributed as dist
from lightning import LightningModule
from ..utils import retrieval_metrics
from ..data.datasets import ignore_label

def plm_mean_pooling(model, batch):
    hiddens = model(**batch).last_hidden_state
    hiddens = hiddens * batch['attention_mask'].unsqueeze(-1)
    embeddings = hiddens.sum(dim=1) / batch['attention_mask'].sum(dim=1, keepdim=True)
    return embeddings

def plm_max_pooling(model, batch):
    hiddens = model(**batch).last_hidden_state
    hiddens.masked_fill_(batch['attention_mask'].unsqueeze(-1) == 0, float('-inf'))
    embeddings = hiddens.max(dim=1).values
    return embeddings

class EmbeddingExtractor(LightningModule):
    def __init__(self, model, embed_fn=None, output_path=None):
        super().__init__()
        self.model = model
        if embed_fn is None:
            embed_fn = lambda model, batch: model(batch)
        self.embed_fn = embed_fn
        self.output_path = output_path
    
    def setup(self, stage='predict'):
        self.embeddings = []
        if stage == 'predict':
            self.total_size = len(self.trainer.datamodule.dataset)
    
    def forward(self, batch):
        embeddings = self.embed_fn(self.model, batch)
        self.embeddings.append(embeddings.cpu())
        return embeddings
    
    def gather_embeddings(self):
        embeddings = torch.cat(self.embeddings)
        self.embeddings.clear()
        if self.trainer.world_size == 1:
            return embeddings
        
        # save embeddings of rank > 0 to disk
        if not self.trainer.is_global_zero:
            torch.save(embeddings, f'embed_cache_{self.trainer.global_rank}.tmp')
        dist.barrier()
        
        # load embeddings on rank 0
        if self.trainer.is_global_zero:
            embeddings = [embeddings]
            for rank in range(1, self.trainer.world_size):
                embeddings.append(torch.load(f'embed_cache_{rank}.tmp'))
                os.remove(f'embed_cache_{rank}.tmp')
            embeddings = torch.stack(embeddings, dim=1)
        
            # restore original order and drop padded samples
            embed_dim = embeddings.size(-1)
            embeddings = embeddings.reshape(-1, embed_dim)
            embeddings = embeddings[:self.total_size]
            return embeddings
    
    def predict_step(self, batch, batch_idx):
        _ = self(batch)
    
    def on_predict_epoch_end(self):
        embeddings = self.gather_embeddings()
        self.embeddings = embeddings
        if self.output_path and self.trainer.is_global_zero:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            torch.save(embeddings, self.output_path)

class DenseRetriever(EmbeddingExtractor):
    def setup(self, stage='test'):
        self.embeddings = []
        if stage == 'test':
            self.query_indices = self.trainer.datamodule.test_query_indices,
            self.eval_labels = self.trainer.datamodule.test_labels
            self.total_size = self.eval_labels.size(1)
        elif stage == 'predict':
            self.query_indices = self.trainer.datamodule.pred_query_indices
            self.total_size = len(self.trainer.datamodule.pred_dataset)

    def gather_predictions(self):
        all_embeds = self.gather_embeddings()
        if self.trainer.is_global_zero:
            all_embeds = all_embeds / all_embeds.norm(dim=1, keepdim=True)
            query_embeds = all_embeds[self.query_indices]
            preds = query_embeds @ all_embeds.T
            return preds

    def test_step(self, batch, batch_idx):
        _ = self(batch)
    
    def on_test_epoch_end(self):
        preds = self.gather_predictions()
        if self.trainer.is_global_zero:
            metrics = retrieval_metrics(
                preds.to(self.device),
                self.eval_labels.to(self.device),
                k=min(20, self.total_size),
                ignore_index=ignore_label
            )
            self.log_dict(
                {f'test_{key}': value for key, value in metrics.items()},
                on_step=False,
                on_epoch=True,
                logger=True,
                rank_zero_only=True,
                batch_size=preds.size(0)
            )
        self.eval_preds = preds
    
    def on_predict_epoch_end(self):
        preds = self.gather_predictions()
        self.eval_preds = preds
        if self.output_path and self.trainer.is_global_zero:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            torch.save(preds, self.output_path)
