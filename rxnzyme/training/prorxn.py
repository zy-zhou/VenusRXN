import torch
import torch.nn as nn
from .base import LightningModelBase, ContrastiveLoss, ListMLELoss
from ..utils import retrieval_metrics, ranking_metrics
from ..data.datasets import ignore_label

class LitProRxnForMM(LightningModelBase):
    def __init__(self, prorxn, train_config):
        super().__init__(prorxn, train_config)
        self.cl_loss = ContrastiveLoss(
            local_loss=train_config['local_loss'],
            gather_with_grad=train_config['gather_with_grad'],
            soft_weight=train_config['soft_weight'],
            return_logits=True
        )

        self.gamma = train_config['gamma']
        if self.gamma > 0:
            if self.model.plm.config.add_cross_attention:
                self.bce_loss = nn.BCEWithLogitsLoss()
            else:
                raise ValueError('The PLM should be configured with cross attention layers when `gamma` > 0.')
        elif self.model.plm.config.add_cross_attention:
            raise Warning('The cross attention layers in the PLM will not be used when `gamma` = 0.')

    def setup(self, stage='fit'):
        self.cl_loss.rank = self.trainer.global_rank
        self.cl_loss.world_size = self.trainer.world_size

        if stage in {'fit', 'validate'} and self.trainer.datamodule.val_rxn_dataset is not None:
            self.setup_evaluation(
                self.trainer.datamodule.val_labels.to(self.device),
                ranking=False
            )
        elif stage == 'test':
            self.setup_evaluation(
                self.trainer.datamodule.test_labels.to(self.device),
                ranking=self.trainer.datamodule.test_cdts is not None
            )
        elif stage == 'predict':
            self.setup_evaluation(
                eval_rxn_num=len(self.trainer.datamodule.pred_rxn_dataset),
                eval_enz_num=len(self.trainer.datamodule.pred_enz_dataset),
                ranking=False
            )

    def sample_hard_negatives(self, logits):
        with torch.no_grad():
            logits.fill_diagonal_(float('-inf')) # mask positives
            neg_probs = logits.softmax(dim=1)
            neg_indices = torch.multinomial(neg_probs, 1).squeeze(1)
        return neg_indices
    
    def compute_loss(self, batch):
        rxn_hiddens, rxn_reps, _, enz_reps = self.model(batch)
        batch_size = rxn_reps.size(0)
        
        # compute alignment loss and logits
        cl_loss, rxn_logits, enz_logits = self.cl_loss(rxn_reps, enz_reps)
        if self.gamma == 0:
            return cl_loss, dict(batch_size=batch_size)
        
        # prepare inputs for the PLM with cross attention
        # foward the postive pairs
        pos_reps = self.model.plm(
            **batch['enzymes'],
            encoder_hidden_states=rxn_hiddens,
            encoder_attention_mask=batch['rxns']['length_mask']
        ).pooler_output
        
        # foward the negative pairs
        sample_neg_for_rxn = torch.rand(1).item() < 0.5
        if sample_neg_for_rxn:
            neg_indices = self.sample_hard_negatives(rxn_logits)
            neg_reps = self.model.plm(
                input_ids=batch['enzymes'].input_ids[neg_indices],
                attention_mask=batch['enzymes'].attention_mask[neg_indices],
                encoder_hidden_states=rxn_hiddens,
                encoder_attention_mask=batch['rxns']['length_mask']
            ).pooler_output
        else:
            neg_indices = self.sample_hard_negatives(enz_logits)
            neg_reps = self.model.plm(
                input_ids=batch['enzymes'].input_ids,
                attention_mask=batch['enzymes'].attention_mask,
                encoder_hidden_states=rxn_hiddens[neg_indices],
                encoder_attention_mask=batch['rxns']['length_mask'][neg_indices]
            ).pooler_output
        
        rxn_enz_logits = self.model.out_proj(torch.cat([pos_reps, neg_reps])).squeeze(1)
        rxn_enz_labels = torch.cat([
            torch.ones_like(rxn_enz_logits[:batch_size]),
            torch.zeros_like(rxn_enz_logits[batch_size:])
        ])
        ce_loss = self.bce_loss(rxn_enz_logits, rxn_enz_labels)

        loss = cl_loss + self.gamma * ce_loss
        logs = dict(
            batch_size=batch_size,
            cl_loss=cl_loss.detach(),
            ce_loss=ce_loss.detach()
        )
        return loss, logs

    def setup_evaluation(self, eval_labels=None, eval_rxn_num=None, eval_enz_num=None, ranking=False):
        self.eval_labels = eval_labels
        self.eval_rxn_num = eval_labels.size(0) if eval_rxn_num is None else eval_rxn_num
        self.eval_enz_num = eval_labels.size(1) if eval_enz_num is None else eval_enz_num
        self.eval_ranking = ranking
        if ranking:
            self.eval_preds = []
        else:
            self.eval_rxn_reps = []
            self.eval_enz_reps = []
    
    def evaluation_step(self, batch, dataloader_idx=0):
        if self.eval_ranking:
            logits = self.model(batch, ranking=True)
            self.eval_preds.append(logits.sigmoid())
        
        else:
            if dataloader_idx == 0: # rxn batch
                rxn_reps = self.model.encode_rxns(batch)
                self.eval_rxn_reps.append(rxn_reps)
            else: # enzyme batch
                enz_reps = self.model.encode_enzymes(batch)
                self.eval_enz_reps.append(enz_reps)
    
    def gather_predictions(self):
        if self.eval_ranking:
            preds = torch.cat(self.eval_preds)
            self.eval_preds.clear()

            if self.trainer.world_size > 1:
                # gather predictions from all ranks
                preds = self.all_gather(preds)
                # restore original order and drop padded samples
                preds = preds.transpose(0, 1).reshape(-1)
                preds = preds[: self.eval_rxn_num * self.eval_enz_num]
            
            preds = preds.reshape(self.eval_rxn_num, self.eval_enz_num)
        
        else:
            rxn_reps = torch.cat(self.eval_rxn_reps)
            self.eval_rxn_reps.clear()
            enz_reps = torch.cat(self.eval_enz_reps)
            self.eval_enz_reps.clear()

            if self.trainer.world_size > 1:
                # gather rxn_reps from all ranks
                rxn_reps = self.all_gather(rxn_reps)
                # restore original order and drop padded samples
                hidden_dim = rxn_reps.size(-1)
                rxn_reps = rxn_reps.transpose(0, 1).reshape(-1, hidden_dim)
                rxn_reps = rxn_reps[:self.eval_rxn_num]
            
            rxn_reps = rxn_reps / rxn_reps.norm(dim=1, keepdim=True)
            enz_reps = enz_reps / enz_reps.norm(dim=1, keepdim=True)
            preds = rxn_reps @ enz_reps.T

            if self.trainer.world_size > 1:
                # gather predictions from all ranks
                preds = self.all_gather(preds)
                # restore original order and drop padded samples
                preds = preds.permute(1, 2, 0).reshape(self.eval_rxn_num, -1)
                preds = preds[:, :self.eval_enz_num]
        
        return preds
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.evaluation_step(batch, dataloader_idx)
    
    def on_validation_epoch_end(self):
        preds = self.gather_predictions()
        metrics = retrieval_metrics(
            preds, self.eval_labels, k=min(20, self.eval_enz_num), ignore_index=ignore_label
        )
        self.log_dict(
            {f'val_{key}': value for key, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=preds.size(0)
        )
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.evaluation_step(batch, dataloader_idx)
    
    def on_test_epoch_end(self):
        preds = self.gather_predictions()
        metrics = retrieval_metrics(
            preds, self.eval_labels, k=min(20, self.eval_enz_num), ignore_index=ignore_label
        )
        self.log_dict(
            {f'test_{key}': value for key, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=preds.size(0)
        )
        self.eval_preds = preds
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.evaluation_step(batch, dataloader_idx)
    
    def on_predict_epoch_end(self):
        preds = self.gather_predictions()
        self.eval_preds = preds

class LitProRxnForLTR(LightningModelBase):
    def __init__(self, prorxn, train_config):
        super().__init__(prorxn, train_config)
        if train_config['ranking_loss']:
            self.loss_fn = ListMLELoss(temperature=train_config['output_type'] == 'cosine')
        else:
            self.loss_fn = nn.MSELoss()
        self.strict_loading = False

    def state_dict(self):
        frozen_keys = {k for k, v in self.named_parameters() if not v.requires_grad}
        state_dict = super().state_dict()
        return {
            k: v for k, v in state_dict.items() if k not in frozen_keys
        }
    
    def setup(self, stage='fit'):
        if stage in {'fit', 'validate'} and self.trainer.datamodule.val_dataset is not None:
            self.setup_evaluation(
                torch.from_numpy(self.trainer.datamodule.val_labels['label'].values).to(self.device)
            )
        elif stage == 'test':
            self.setup_evaluation(
                torch.from_numpy(self.trainer.datamodule.test_labels['label'].values).to(self.device)
            )
        else:
            self.setup_evaluation(eval_size=len(self.trainer.datamodule.pred_dataset))

    def compute_loss(self, batch):
        preds = self.model(batch)
        loss = self.loss_fn(preds, batch['labels'])
        return loss, dict(batch_size=preds.size(0))
    
    def setup_evaluation(self, eval_labels=None, eval_size=None):
        self.eval_preds = []
        self.eval_labels = eval_labels
        self.eval_size = len(eval_labels) if eval_size is None else eval_size
    
    def evaluation_step(self, batch):
        preds = self.model(batch)
        self.eval_preds.append(preds)
    
    def gather_predictions(self):
        preds = torch.cat(self.eval_preds)
        self.eval_preds.clear()

        if self.trainer.world_size > 1:
            # gather predictions from all ranks
            preds = self.all_gather(preds)
            # restore original order and drop padded samples
            preds = preds.transpose(0, 1).reshape(-1)
            preds = preds[:self.eval_size]
        
        return preds
    
    def validation_step(self, batch, batch_idx):
        self.evaluation_step(batch)
    
    def on_validation_epoch_end(self):
        preds = self.gather_predictions()
        metrics = ranking_metrics(preds, self.eval_labels)
        self.log_dict(
            {f'val_{key}': value for key, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=preds.size(0)
        )
    
    def test_step(self, batch, batch_idx):
        self.evaluation_step(batch)
    
    def on_test_epoch_end(self):
        preds = self.gather_predictions()
        metrics = ranking_metrics(preds, self.eval_labels)
        self.log_dict(
            {f'test_{key}': value for key, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=preds.size(0)
        )
        self.eval_preds = preds

    def predict_step(self, batch, batch_idx):
        self.evaluation_step(batch)
    
    def on_predict_epoch_end(self):
        preds = self.gather_predictions()
        self.eval_preds = preds
