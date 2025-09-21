import torch
import torch.cuda as cuda
import torch.optim as optim
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch import nn
from torch.optim.lr_scheduler import StepLR
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

def get_optimizer(optimizer, lr, params):
    params = filter(lambda p: p.requires_grad, params)
    if optimizer == 'sgd':
        return optim.SGD(params, lr=lr)
    elif optimizer == 'nag':
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    elif optimizer == 'adagrad':
        return optim.Adagrad(params, lr=lr)
    elif optimizer == 'adadelta':
        return optim.Adadelta(params, lr=lr)
    elif optimizer == 'adam':
        return optim.Adam(params, lr=lr)
    elif optimizer == 'adamw':
        return optim.AdamW(params, lr=lr)
    else:
        raise ValueError('Unknown optimizer: ' + optimizer)

def get_trainer(train_config):
    if train_config['logger']:
        logger = CSVLogger(
            save_dir=train_config['ckpt_dir'],
            name=None,
            version=train_config['log_version']
        )
    else:
        logger = False

    ckpt_callback = ModelCheckpoint(
        save_top_k=train_config['save_top_k'],
        monitor=train_config['val_metric'],
        mode='min' if '_loss' in train_config['val_metric'] else 'max',
        dirpath=train_config['ckpt_dir'],
        filename=f'{{epoch}}-{{{train_config["val_metric"]}:.4f}}'
    )
    stop_callback = EarlyStopping(
        monitor=train_config['val_metric'],
        mode='min' if '_loss' in train_config['val_metric'] else 'max',
        min_delta=0.0,
        patience=train_config['patience'],
        check_on_train_epoch_end=False
    )
    
    strategy = 'auto' if cuda.device_count() <= 1 else 'ddp_find_unused_parameters_true'
    trainer = Trainer(
        strategy=strategy,
        precision=train_config['precision'],
        logger=logger,
        callbacks=[ckpt_callback, stop_callback],
        max_epochs=train_config['max_epochs'],
        check_val_every_n_epoch=train_config['val_interval'],
        num_sanity_val_steps=0,
        accumulate_grad_batches=train_config['grad_acc'],
        gradient_clip_val=train_config['grad_clip'],
        use_distributed_sampler=False
    )
    return trainer

class LightningModelBase(LightningModule):
    def __init__(self, train_config):
        super().__init__()
        self.model = None
        self.opt_name = train_config['optimizer']
        self.init_lr = train_config['lr']
        self.lr_decay = train_config['lr_decay']
        self.lr_decay_steps = train_config['lr_decay_steps']
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.opt_name, self.init_lr, self.parameters())
        config = {'optimizer': optimizer}
        
        if self.lr_decay > 0:
            config['lr_scheduler'] = StepLR(
                optimizer=optimizer,
                step_size=self.lr_decay_steps,
                gamma=self.lr_decay
            )
        return config
    
    def forward(self, batch):
        return self.model(batch)
    
    def compute_loss(self, batch):
        '''
        Return loss and a log_dict containing the metrics other than the loss to log.
        '''
        raise NotImplementedError('`compute_loss` is not implemented.')

    def on_train_epoch_start(self):
        if isinstance(self.trainer.train_dataloader, list):
            dataloaders = self.trainer.train_dataloader
        else:
            dataloaders = [self.trainer.train_dataloader]
        
        for dataloader in dataloaders: # check distributed batch sampler
            if hasattr(dataloader, 'batch_sampler') and hasattr(dataloader.batch_sampler, 'set_epoch'):
                if callable(dataloader.batch_sampler.set_epoch):
                    dataloader.batch_sampler.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        loss, logs = self.compute_loss(batch)
        batch_size = logs.pop('batch_size', None)
        
        self.log(
            'train_loss',
            loss.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size
        )
        self.log_dict(
            {f'train_{key}': value for key, value in logs.items()},
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size
        )
        return loss

    def on_validation_epoch_start(self):
        if isinstance(self.trainer.val_dataloaders, list):
            dataloaders = self.trainer.val_dataloaders
        else:
            dataloaders = [self.trainer.val_dataloaders]
        
        for dataloader in dataloaders: # check distributed batch sampler
            if hasattr(dataloader, 'batch_sampler') and hasattr(dataloader.batch_sampler, 'set_epoch'):
                if callable(dataloader.batch_sampler.set_epoch):
                    dataloader.batch_sampler.set_epoch(self.current_epoch)

    def validation_step(self, batch, batch_idx):
        loss, logs = self.compute_loss(batch)
        batch_size = logs.pop('batch_size', None)

        self.log(
            'val_loss',
            loss.detach(),
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size
        )
        self.log_dict(
            {f'val_{key}': value for key, value in logs.items()},
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size
        )
    
    def on_train_epoch_end(self):
        if self.trainer.is_global_zero:
            for key, value in self.trainer.logged_metrics.items():
                if not key.endswith('_step'):
                    print(f'{key}: {value:.4f}')

class MultiAttrCELoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
    
    def forward(self, logits, labels, class_nums):
        total_class, num_attr = logits.size(-1), labels.size(-1)
        logits = logits.reshape(-1, total_class).split(class_nums, dim=1)
        labels = labels.reshape(-1, num_attr)

        losses = [self.ce_loss(attr_logits, labels[:, attr]) \
                      for attr, attr_logits in enumerate(logits)]
        loss = sum(losses) / len(losses)
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(
            self,
            local_loss=True,
            gather_with_grad=False,
            rank=0,
            world_size=1,
            soft_weight=1.0,
            return_logits=False
        ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.soft_weight = soft_weight
        self.return_logits = return_logits
        
        self.ce_loss = nn.CrossEntropyLoss()
        if soft_weight > 0:
            self.kl_div = nn.KLDivLoss(reduction='batchmean', log_target=False)
        # learnable temperature parameter
        self.t = nn.Parameter(torch.tensor(1 / 0.07).log())

    def all_gather(self, tensor):
        if self.gather_with_grad and tensor.requires_grad:
            gather_list = list(dist_nn.all_gather(tensor))
        else:
            gather_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(gather_list, tensor)
            # ensure grads for local rank when all_* features don't have a gradient
            gather_list[self.rank] = tensor
        
        # move local tensor to the first position
        gather_list = [gather_list[self.rank]] + gather_list[:self.rank] + gather_list[self.rank + 1:]
        return torch.cat(gather_list)
    
    def get_logits(self, reps1, reps2):
        reps1 = reps1 / reps1.norm(dim=1, keepdim=True)
        reps2 = reps2 / reps2.norm(dim=1, keepdim=True)
        
        if self.world_size > 1:
            all_reps1 = self.all_gather(reps1)
            all_reps2 = self.all_gather(reps2)
            
            if self.local_loss:
                logits12 = self.t.exp() * reps1 @ all_reps2.T
                logits21 = self.t.exp() * reps2 @ all_reps1.T
                if self.soft_weight > 0:
                    logits11 = self.t.exp() * reps1 @ all_reps1.T
                    logits22 = self.t.exp() * reps2 @ all_reps2.T
            
            else:
                logits12 = self.t.exp() * all_reps1 @ all_reps2.T
                logits21 = logits12.T
                if self.soft_weight > 0:
                    logits11 = self.t.exp() * all_reps1 @ all_reps1.T
                    logits22 = self.t.exp() * all_reps2 @ all_reps2.T
        
        else:
            logits12 = self.t.exp() * reps1 @ reps2.T
            logits21 = logits12.T
            if self.soft_weight > 0:
                logits11 = self.t.exp() * reps1 @ reps1.T
                logits22 = self.t.exp() * reps2 @ reps2.T
        
        if self.soft_weight > 0:
            return logits12, logits21, logits11, logits22
        else:
            return logits12, logits21

    def forward(self, reps1, reps2):
        logits = self.get_logits(reps1, reps2)
        
        logits12, logits21 = logits[:2]
        labels = torch.arange(logits12.size(0), device=logits12.device)
        loss = (
            self.ce_loss(logits12, labels) + \
            self.ce_loss(logits21, labels)
        ) / 2
        
        if self.soft_weight > 0:
            logits11, logits22 = logits[2:]
            loss = loss + self.soft_weight * (
                self.kl_div(logits12.log_softmax(dim=1), logits11.softmax(dim=1)) + \
                self.kl_div(logits21.log_softmax(dim=1), logits22.softmax(dim=1))
            ) / 2

        if self.return_logits:
            # return local logits
            batch_size = reps1.size(0)
            logits12 = logits12[:batch_size, :batch_size]
            logits21 = logits21[:batch_size, :batch_size]
            return loss, logits12, logits21
        else:
            return loss

class PairwiseRankingLoss(nn.Module):
    def __init__(self, fn='hinge', margin=1.0):
        super().__init__()
        if fn not in {'hinge', 'exp', 'log'}:
            raise ValueError(f'Unknown loss function: {fn}')
        self.fn = fn
        if fn == 'hinge':
            self.margin_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, input1, input2, label1, label2):
        target = torch.where(label1 > label2, 1.0, -1.0)
        
        if self.fn == 'hinge':
            loss = self.margin_loss(input1, input2, target)
        elif self.fn == 'exp':
            loss = torch.exp(- target * (input1 - input2)).mean()
        else:
            loss = torch.log(1 + torch.exp(- target * (input1 - input2))).mean()
        return loss

class ListMLELoss(nn.Module):
    def __init__(self, temperature=False):
        super().__init__()
        if temperature:
            self.t = nn.Parameter(torch.tensor(1 / 0.07).log())
        else:
            self.t = None

    def forward(self, predicts, targets):
        if self.t is not None:
            predicts = self.t.exp() * predicts

        indices = targets.sort(dim=-1, descending=True).indices
        predicts = torch.gather(predicts, dim=-1, index=indices)
        cumsums = predicts.exp().flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        loss = torch.log(cumsums + 1e-10) - predicts
        return loss.mean(dim=-1).mean()
