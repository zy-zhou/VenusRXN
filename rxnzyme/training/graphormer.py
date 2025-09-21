import torch
import torch.nn as nn
from .base import LightningModelBase, MultiAttrCELoss, ContrastiveLoss
from ..utils import classification_metrics
from ..data.datasets import ignore_label

class LitMolGraphormerForMLM(LightningModelBase):
    def __init__(self, mlm_graphormer, train_config):
        super().__init__(train_config)
        self.model = mlm_graphormer
        self.alpha = train_config['alpha'] # weight for reactive center prediction task
        self.beta = train_config['beta'] # weight for graph contrastive learning task
        self.ce_loss = MultiAttrCELoss(ignore_index=ignore_label)
        self.cl_loss = ContrastiveLoss(
            local_loss=train_config['local_loss'],
            gather_with_grad=train_config['gather_with_grad'],
            soft_weight=0.0
        )
        self.save_hyperparameters(ignore=['mlm_graphormer'])

    def setup(self, stage='fit'):
        self.cl_loss.rank = self.trainer.global_rank
        self.cl_loss.world_size = self.trainer.world_size

    def compute_loss(self, batch):
        r_mlm_logits, r_rcp_logits, r_reps, p_mlm_logits, p_rcp_logits, p_reps = self.model(batch)

        mlm_class_nums = self.model.mlm_class_nums
        rcp_class_nums = self.model.rcp_class_nums

        r_mlm_loss = self.ce_loss(r_mlm_logits, batch['r_mlm_labels'], mlm_class_nums)
        p_mlm_loss = self.ce_loss(p_mlm_logits, batch['p_mlm_labels'], mlm_class_nums)
        mlm_loss = (r_mlm_loss + p_mlm_loss) / 2

        r_rcp_loss = self.ce_loss(r_rcp_logits, batch['r_rcp_labels'], rcp_class_nums)
        p_rcp_loss = self.ce_loss(p_rcp_logits, batch['p_rcp_labels'], rcp_class_nums)
        rcp_loss = (r_rcp_loss + p_rcp_loss) / 2

        cl_loss = self.cl_loss(r_reps, p_reps)

        loss = mlm_loss + self.alpha * rcp_loss + self.beta * cl_loss
        logs = dict(
            batch_size=r_mlm_logits.size(0),
            mlm_loss=mlm_loss.detach(),
            rcp_loss=rcp_loss.detach(),
            cl_loss=cl_loss.detach()
        )
        return loss, logs

class LitRxnGraphormerForSCL(LightningModelBase):
    def __init__(self, scl_graphormer, train_config):
        super().__init__(train_config)
        self.model = scl_graphormer
        self.cl_loss = ContrastiveLoss(
            local_loss=train_config['local_loss'],
            gather_with_grad=train_config['gather_with_grad'],
            soft_weight=0.0
        )
        self.save_hyperparameters(ignore=['scl_graphormer'])
    
    def setup(self, stage='fit'):
        self.cl_loss.rank = self.trainer.global_rank
        self.cl_loss.world_size = self.trainer.world_size

        if stage in {'fit', 'validate'} and self.trainer.datamodule.val_dataset is not None:
            self.setup_evaluation(
                torch.from_numpy(self.trainer.datamodule.train_labels.values).to(self.device),
                torch.from_numpy(self.trainer.datamodule.val_labels.values).to(self.device)
            )
        elif stage == 'test':
            self.setup_evaluation(
                torch.from_numpy(self.trainer.datamodule.train_labels.values).to(self.device),
                torch.from_numpy(self.trainer.datamodule.test_labels.values).to(self.device)
            )

    def compute_loss(self, batch):
        rxn_reps = self.model(batch) # 2 * batch_size, hidden_dim
        rxn_reps1, rxn_reps2 = rxn_reps[::2], rxn_reps[1::2]
        loss = self.cl_loss(rxn_reps1, rxn_reps2)
        return loss, dict(batch_size=rxn_reps1.size(0))

    def setup_evaluation(self, train_labels, eval_labels=None, eval_size=None):
        self.eval_rxn_reps = []
        self.train_labels = train_labels
        self.eval_labels = eval_labels
        self.eval_size = len(eval_labels) if eval_size is None else eval_size
    
    def evaluation_step(self, batch):
        rxn_reps = self.model(batch)
        self.eval_rxn_reps.append(rxn_reps)

    def gather_predictions(self):
        rxn_reps = torch.cat(self.eval_rxn_reps)
        self.eval_rxn_reps.clear()
        
        if self.trainer.world_size > 1:
            # gather reps from all ranks, and all ranks will finally get the same (total) predictions
            rxn_reps = self.all_gather(rxn_reps)
            # restore original order and drop padded samples
            hidden_dim = rxn_reps.size(-1)
            rxn_reps = rxn_reps.transpose(0, 1).reshape(-1, hidden_dim)
            rxn_reps = rxn_reps[: len(self.train_labels) + self.eval_size]
        
        rxn_reps = rxn_reps / rxn_reps.norm(dim=1, keepdim=True)
        train_reps = rxn_reps[:len(self.train_labels)]
        eval_reps = rxn_reps[len(self.train_labels):]
        sims = eval_reps @ train_reps.T
        nns = sims.argmax(1)
        preds = self.train_labels[nns]
        return preds, nns

    def validation_step(self, batch, batch_idx):
        self.evaluation_step(batch)
    
    def on_validation_epoch_end(self):
        preds, _ = self.gather_predictions()
        metrics = classification_metrics(preds, self.eval_labels)
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
        preds, nns = self.gather_predictions()
        metrics = classification_metrics(preds, self.eval_labels)
        self.log_dict(
            {f'test_{key}': value for key, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=preds.size(0)
        )
        self.eval_preds = preds
        self.eval_nns = nns
    
class LitRxnGraphormerForCls(LightningModelBase):
    def __init__(self, cls_graphormer, train_config):
        super().__init__(train_config)
        self.model = cls_graphormer
        self.ce_loss = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=['cls_graphormer'])

    def setup(self, stage='fit'):
        if stage in {'fit', 'validate'} and self.trainer.datamodule.val_dataset is not None:
            self.setup_evaluation(
                torch.from_numpy(self.trainer.datamodule.val_labels.values).to(self.device)
            )
        elif stage == 'test':
            self.setup_evaluation(
                torch.from_numpy(self.trainer.datamodule.test_labels.values).to(self.device)
            )
    
    def compute_loss(self, batch):
        logits = self.model(batch)
        loss = self.ce_loss(logits, batch['labels'])
        return loss, dict(batch_size=logits.size(0))

    def setup_evaluation(self, eval_labels=None, eval_size=None):
        self.eval_preds = []
        self.eval_labels = eval_labels
        self.eval_size = len(eval_labels) if eval_size is None else eval_size
    
    def evaluation_step(self, batch):
        logits = self.model(batch)
        self.eval_preds.append(logits.argmax(dim=1))
    
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
        metrics = classification_metrics(preds, self.eval_labels)
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
        metrics = classification_metrics(preds, self.eval_labels)
        self.log_dict(
            {f'test_{key}': value for key, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=preds.size(0)
        )
        self.eval_preds = preds
