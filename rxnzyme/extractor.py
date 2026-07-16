import os
import shutil
import torch
import torch.distributed as dist
from lightning import LightningModule
from .utils import retrieval_metrics
from .data.datasets.base import ignore_label

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
    def __init__(
        self,
        model,
        embed_fn=None,
        l2_norm=False,
        output_path=None,
        tmp_dir=None
    ):
        super().__init__()
        self.model = model
        if embed_fn is None:
            embed_fn = lambda model, batch: model(batch)
        self.embed_fn = embed_fn
        self.l2_norm = l2_norm
        self.output_path = output_path
        if tmp_dir is not None:
            self.tmp_dir = tmp_dir
        elif output_path is not None:
            self.tmp_dir = os.path.splitext(output_path)[0] + '_tmp'
        else:
            self.tmp_dir = 'tmp'
    
    def setup(self, stage='predict'):
        assert stage == 'predict', 'Extractor only supports prediction.'
        self.embeddings = []
        self.total_size = len(self.trainer.datamodule.dataset)
    
    def forward(self, batch):
        embeddings = self.embed_fn(self.model, batch)
        if self.l2_norm:
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        self.embeddings.append(embeddings.cpu())
        return embeddings
    
    def gather_embeddings(self):
        embeddings = torch.cat(self.embeddings)
        self.embeddings.clear()
        if self.trainer.world_size == 1:
            return embeddings
        
        # save embeddings of rank > 0 to disk
        if self.trainer.is_global_zero:
            os.makedirs(self.tmp_dir)
        dist.barrier()
        if not self.trainer.is_global_zero:
            tmp_path = os.path.join(self.tmp_dir, f'embed_cache_{self.trainer.global_rank}.tmp')
            torch.save(embeddings, tmp_path)
        dist.barrier()
        
        # load embeddings on rank 0
        if self.trainer.is_global_zero:
            embeddings = [embeddings]
            for rank in range(1, self.trainer.world_size):
                tmp_path = os.path.join(self.tmp_dir, f'embed_cache_{rank}.tmp')
                embeddings.append(torch.load(tmp_path))
            shutil.rmtree(self.tmp_dir)

            # restore original order and drop padded samples
            embeddings = torch.stack(embeddings, dim=1).flatten(0, 1)
            return embeddings[:self.total_size]
    
    def predict_step(self, batch, batch_idx):
        _ = self(batch)
    
    def on_predict_epoch_end(self):
        self.embeddings = self.gather_embeddings()
        if self.output_path and self.trainer.is_global_zero:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            torch.save(self.embeddings, self.output_path)

class Enzyme2EnzymeRetriever(EmbeddingExtractor):
    def __init__(
        self,
        model,
        embed_fn=None,
        output_path=None,
        tmp_dir=None
    ):
        super().__init__(
            model,
            embed_fn,
            l2_norm=True,
            output_path=output_path,
            tmp_dir=tmp_dir
        )

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
        self.embeddings = self.gather_embeddings()
        if self.trainer.is_global_zero:
            query_embeds = self.embeddings[self.query_indices]
            preds = query_embeds @ self.embeddings.T
            return preds

    def test_step(self, batch, batch_idx):
        _ = self(batch)
    
    def on_test_epoch_end(self):
        self.eval_preds = self.gather_predictions()
        if self.trainer.is_global_zero:
            metrics = retrieval_metrics(
                self.eval_preds,
                self.eval_labels,
                k=min(20, self.total_size),
                ignore_index=ignore_label
            )
            self.log_dict(
                {f'test_{key}': value for key, value in metrics.items()},
                on_step=False,
                on_epoch=True,
                logger=True,
                rank_zero_only=True,
                batch_size=self.eval_preds.size(0)
            )
    
    def on_predict_epoch_end(self):
        self.eval_preds = self.gather_predictions()
        if self.output_path and self.trainer.is_global_zero:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            torch.save(self.eval_preds, self.output_path)
