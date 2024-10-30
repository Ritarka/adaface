import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.core import LightningModule
from torch.nn import CrossEntropyLoss
import evaluate_utils
import head
import net
import numpy as np
import utils


class Trainer(LightningModule):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__()
        self.save_hyperparameters()  # sets self.hparams

        self.class_num = utils.get_num_class(self.hparams)
        print('classnum: {}'.format(self.class_num))

        self.model = net.build_model(model_name=self.hparams.arch)
        self.head = head.build_head(head_type=self.hparams.head,
                                     embedding_size=512,
                                     class_num=self.class_num,
                                     m=self.hparams.m,
                                     h=self.hparams.h,
                                     t_alpha=self.hparams.t_alpha,
                                     s=self.hparams.s,
                                     )

        self.cross_entropy_loss = CrossEntropyLoss()

        if self.hparams.start_from_model_statedict:
            ckpt = torch.load(self.hparams.start_from_model_statedict)
            self.model.load_state_dict({key.replace('model.', ''):val
                                        for key,val in ckpt['state_dict'].items() if 'model.' in key})

    def get_current_lr(self):
        scheduler = None
        if scheduler is None:
            try:
                # pytorch lightning >= 1.8
                scheduler = self.trainer.lr_scheduler_configs[0].scheduler
            except:
                pass

        if scheduler is None:
            # pytorch lightning <=1.7
            try:
                scheduler = self.trainer.lr_schedulers[0]['scheduler']
            except:
                pass

        if scheduler is None:
            raise ValueError('lr calculation not successful')

        if isinstance(scheduler, lr_scheduler._LRScheduler):
            lr = scheduler.get_last_lr()[0]
        else:
            lr = scheduler.optimizer.param_groups[0]['lr']
        return lr


    def forward(self, images, labels):
        embeddings, norms = self.model(images)
        cos_thetas = self.head(embeddings, norms, labels)
        if isinstance(cos_thetas, tuple):
            cos_thetas, bad_grad = cos_thetas
            labels[bad_grad.squeeze(-1)] = -100 # ignore_index
        return cos_thetas, norms, embeddings, labels


    def training_step(self, batch, batch_idx):
        images, labels, _, _ = batch

        cos_thetas, norms, embeddings, labels = self.forward(images, labels)
        loss_train = self.cross_entropy_loss(cos_thetas, labels)
        lr = self.get_current_lr()
        # log
        self.log('lr', lr, on_step=True, on_epoch=True, logger=True)
        self.log('train_loss', loss_train, on_step=True, on_epoch=True, logger=True)

        return loss_train

    def training_epoch_end(self, outputs):
        return None

    def validation_step(self, batch, batch_idx):
        images, labels, dataname, image_index = batch
        embeddings, norms = self.model(images)

        fliped_images = torch.flip(images, dims=[3])
        flipped_embeddings, flipped_norms = self.model(fliped_images)
        stacked_embeddings = torch.stack([embeddings, flipped_embeddings], dim=0)
        stacked_norms = torch.stack([norms, flipped_norms], dim=0)
        embeddings, norms = utils.fuse_features_with_norm(stacked_embeddings, stacked_norms)

        if self.hparams.distributed_backend == 'ddp':
            # to save gpu memory
            return {
                'output': embeddings.to('cpu'),
                'norm': norms.to('cpu'),
                'target': labels.to('cpu'),
                'dataname': dataname.to('cpu'),
                'image_index': image_index.to('cpu')
            }
        else:
            # dp requires the tensor to be cuda
            return {
                'output': embeddings,
                'norm': norms,
                'target': labels,
                'dataname': dataname,
                'image_index': image_index
            }

    def validation_epoch_end(self, outputs):
        # Gather outputs across devices if running in distributed mode
        all_output_tensor, all_norm_tensor, all_target_tensor, all_dataname_tensor = self.gather_outputs(outputs)

        # Dictionary to hold logging metrics
        val_logs = {}

        # Process dataset outputs and calculate evaluation metrics
        embeddings = all_output_tensor.to('cpu').numpy()
        labels = all_target_tensor.to('cpu').numpy()
        issame = labels[0::2]

        # Evaluate embeddings
        tpr, fpr, accuracy, best_thresholds = evaluate_utils.evaluate(embeddings, issame, nrof_folds=10)
        acc, best_threshold = accuracy.mean(), best_thresholds.mean()

        # Prepare values for logging
        num_val_samples = torch.tensor(len(embeddings), dtype=torch.float32)
        epoch = torch.tensor(self.current_epoch, dtype=torch.float32)
        val_acc = torch.tensor(acc, dtype=torch.float32)
        best_threshold = torch.tensor(best_threshold, dtype=torch.float32)

        # Store values in val_logs dictionary
        val_logs['best_threshold'] = best_threshold
        val_logs['num_val_samples'] = num_val_samples
        val_logs['val_acc'] = val_acc
        val_logs['epoch'] = epoch

        for k, v in val_logs.items():
            self.log(name=k, value=v)

        return None


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        # Gather outputs across devices if running in distributed mode
        all_output_tensor, all_norm_tensor, all_target_tensor, all_dataname_tensor = self.gather_outputs(outputs)

        # Dictionary to hold logging metrics
        val_logs = {}

        # Process dataset outputs and calculate evaluation metrics
        embeddings = all_output_tensor.to('cpu').numpy()
        labels = all_target_tensor.to('cpu').numpy()
        issame = labels[0::2]

        # Evaluate embeddings
        tpr, fpr, accuracy, best_thresholds = evaluate_utils.evaluate(embeddings, issame, nrof_folds=10)
        acc, best_threshold = accuracy.mean(), best_thresholds.mean()

        # Prepare values for logging
        num_val_samples = torch.tensor(len(embeddings), dtype=torch.float32)
        epoch = torch.tensor(self.current_epoch, dtype=torch.float32)
        val_acc = torch.tensor(acc, dtype=torch.float32)
        best_threshold = torch.tensor(best_threshold, dtype=torch.float32)

        # Store values in val_logs dictionary
        val_logs['best_threshold'] = best_threshold
        val_logs['num_val_samples'] = num_val_samples
        val_logs['val_acc'] = val_acc
        val_logs['epoch'] = epoch

        for k, v in val_logs.items():
            self.log(name=k, value=v)

        return None

    def gather_outputs(self, outputs):
        if self.hparams.distributed_backend == 'ddp':
            # gather outputs across gpu
            outputs_list = []
            _outputs_list = utils.all_gather(outputs)
            for _outputs in _outputs_list:
                outputs_list.extend(_outputs)
        else:
            outputs_list = outputs

        # if self.trainer.is_global_zero:
        all_output_tensor = torch.cat([out['output'] for out in outputs_list], axis=0).to('cpu')
        all_norm_tensor = torch.cat([out['norm'] for out in outputs_list], axis=0).to('cpu')
        all_target_tensor = torch.cat([out['target'] for out in outputs_list], axis=0).to('cpu')
        all_dataname_tensor = torch.cat([out['dataname'] for out in outputs_list], axis=0).to('cpu')
        all_image_index = torch.cat([out['image_index'] for out in outputs_list], axis=0).to('cpu')
                
        # print(len(all_output_tensor))
        # print(len(all_norm_tensor))
        # print(len(all_target_tensor))
        # print(len(all_dataname_tensor))
        # print(len(all_image_index))

        # get rid of duplicate index outputs
        unique_dict = {}
        for _out, _nor, _tar, _dat, _idx in zip(all_output_tensor, all_norm_tensor, all_target_tensor,
                                                all_dataname_tensor, all_image_index):
            unique_dict[_idx.item()] = {'output': _out, 'norm': _nor, 'target': _tar, 'dataname': _dat}
        unique_keys = sorted(unique_dict.keys())
        all_output_tensor = torch.stack([unique_dict[key]['output'] for key in unique_keys], axis=0)
        all_norm_tensor = torch.stack([unique_dict[key]['norm'] for key in unique_keys], axis=0)
        all_target_tensor = torch.stack([unique_dict[key]['target'] for key in unique_keys], axis=0)
        all_dataname_tensor = torch.stack([unique_dict[key]['dataname'] for key in unique_keys], axis=0)

        return all_output_tensor, all_norm_tensor, all_target_tensor, all_dataname_tensor

    def configure_optimizers(self):

        # paras_only_bn, paras_wo_bn = self.separate_bn_paras(self.model)
        paras_wo_bn, paras_only_bn = self.split_parameters(self.model)

        optimizer = optim.SGD([{
            'params': paras_wo_bn + [self.head.kernel],
            'weight_decay': 5e-4
        }, {
            'params': paras_only_bn
        }],
                                lr=self.hparams.lr,
                                momentum=self.hparams.momentum)

        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=self.hparams.lr_milestones,
                                             gamma=self.hparams.lr_gamma)

        return [optimizer], [scheduler]

    def split_parameters(self, module):
        params_decay = []
        params_no_decay = []
        for m in module.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                params_no_decay.extend([*m.parameters()])
            elif len(list(m.children())) == 0:
                params_decay.extend([*m.parameters()])
        assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
        return params_decay, params_no_decay
