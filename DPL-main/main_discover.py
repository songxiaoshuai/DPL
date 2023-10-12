import os
import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import Accuracy

from utils.contrast_loss import InstanceLoss
from utils.data import get_datamodule
from utils.eval import ClusterMetrics
from utils.nets import MultiHeadBERT
from utils.sinkhorn_knopp import SinkhornKnopp

parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization.")
# dataset
parser.add_argument("--dataset", required=True, type=str, help="dataset")
parser.add_argument("--data_dir", default="datasets", type=str, help="data directory")
parser.add_argument("--num_labeled_classes", required=True, type=int, help="number of labeled classes")
parser.add_argument("--num_unlabeled_classes", required=True, type=int, help="number of unlabeled classes")
parser.add_argument("--divide_seed", required=True, type=int, help="seed for IND and OOD classes division")

# model
parser.add_argument("--arch", default="bert-base-uncased", type=str, help="backbone architecture")
parser.add_argument("--num_heads", default=1, type=int, help="number of heads for clustering")
parser.add_argument("--pretrained", type=str, help="pretrained checkpoint path")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")
parser.add_argument("--num_workers", default=10, type=int, help="number of workers")

# train
parser.add_argument("--batch_size", default=256, type=int, help="batch size")
parser.add_argument("--base_lr", default=0.02, type=float, help="base learning rate")
parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_opt", default=1.5e-4, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
parser.add_argument("--instance_temperature", default=0.5, type=float, help="instance contrastive learning temperature")
parser.add_argument("--num_views", default=2, type=int, help="number of large crops")
parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the momving average of prototypes')

# sinkhorn algorithm
parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")

# output
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--comment", default='', type=str)
parser.add_argument("--save_results_path", type=str, default='results', help="The path to save results.")


def set_seed(seed, divide_seed):
    random.seed(seed)
    np.random.seed(divide_seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Discoverer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})

        # build model
        self.model = MultiHeadBERT.from_pretrained(
            self.hparams.arch,
            self.hparams.num_labeled_classes,
            self.hparams.num_unlabeled_classes,
            overcluster_factor=3,
            num_heads=self.hparams.num_heads,
            num_hidden_layers=self.hparams.num_hidden_layers,
            contrast_head=1
        )

        state_dict = torch.load(self.hparams.pretrained, map_location=self.device)
        state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}
        self.model.load_state_dict(state_dict, strict=False)
        print("loading pretrain model:", self.hparams.pretrained)
        self.freeze_parameters(self.model)
        self.best_head = 0

        # Sinkorn-Knopp
        self.sk = SinkhornKnopp(
            num_iters=self.hparams.num_iters_sk, epsilon=self.hparams.epsilon_sk
        )

        self.total_features = torch.empty((0, 768)).to(self.device)
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        self.confidence_pseudo_view_1 = None
        self.confidence_pseudo_view_2 = None
        self.cur_epoch = -1

        self.metrics_inc = torch.nn.ModuleList(
            [
                Accuracy(),
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
            ]
        )

        self.metrics_inc_test = torch.nn.ModuleList(
            [
                Accuracy(),
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
            ]
        )

        self.test_results = {}

        # buffer for best head tracking
        self.register_buffer("loss_per_head", torch.zeros(self.hparams.num_heads))
        self.criterion_instance = InstanceLoss(self.hparams.batch_size, self.hparams.instance_temperature,
                                               self.device).to(self.device)

    def freeze_parameters(self, model):
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum_opt,
            weight_decay=self.hparams.weight_decay_opt,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr,
        )
        return [optimizer], [scheduler]

    def cross_entropy_loss(self, preds, targets):
        preds = F.log_softmax(preds / self.hparams.temperature, dim=-1)
        return -torch.mean(torch.sum(targets * preds, dim=-1))

    def swapped_prediction(self, logits, targets):
        loss = 0
        for view in range(self.hparams.num_views):
            for other_view in np.delete(range(self.hparams.num_views), view):
                loss += self.cross_entropy_loss(logits[other_view], targets[view])
        return loss / (self.hparams.num_views * (self.hparams.num_views - 1))

    def on_epoch_start(self):
        self.loss_per_head = torch.zeros_like(self.loss_per_head)

    def unpack_batch(self, batch):
        input_ids, input_mask, segment_ids, label_ids = batch
        mask_lab = label_ids < self.hparams.num_labeled_classes
        return input_ids, input_mask, segment_ids, label_ids, mask_lab

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.cur_epoch += 1

        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, mask_lab = self.unpack_batch(batch)
        nlc = self.hparams.num_labeled_classes
        self.model.normalize_prototypes()  # normalize prototypes

        # Prototypical Contrastive Learning
        # 1.forward and gather the logits
        outputs, outputs_contrast = self.model(input_ids, input_mask, segment_ids, mode="discovery")
        outputs["logits_lab"] = (outputs["logits_lab"].unsqueeze(1).expand(-1, self.hparams.num_heads, -1, -1))
        logits = torch.cat([outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)

        # 2.construct targets for sample and prototype alignment
        prototypes = self.model.prototypes.clone().detach()
        targets = torch.zeros_like(logits)
        targets_lab = (F.one_hot(label_ids[mask_lab], num_classes=nlc).float().to(self.device))
        for v in range(self.hparams.num_views):
            for h in range(self.hparams.num_heads):
                targets[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets[v, h, ~mask_lab, nlc:] = self.sk(outputs["logits_unlab"][v, h, ~mask_lab]).type_as(targets)

        # 3.instance-level contrastive learning
        z_i, z_j = outputs_contrast[0]["instance_features"], outputs_contrast[1]["instance_features"]
        loss_ins = self.criterion_instance(z_i, z_j)

        # 4.prototypical contrastive learning
        logits_prot_view1 = torch.mm(z_i, prototypes.t())
        logits_prot_view2 = torch.mm(z_j, prototypes.t())
        loss_pcl_1 = self.cross_entropy_loss(logits_prot_view1, targets[1, 0, :, :])
        loss_pcl_2 = self.cross_entropy_loss(logits_prot_view2, targets[0, 0, :, :])
        loss_pcl = (loss_pcl_1 + loss_pcl_2) / 2

        # 5.update prototype embedding
        _, target_label_ids = targets.max(dim=-1)
        for feat, label, ground_truth in zip(z_i, target_label_ids[0, 0, :], label_ids):
            self.model.prototypes[label] = self.model.prototypes[label] * args.proto_m + (1 - args.proto_m) * feat
        for feat, label, ground_truth in zip(z_j, target_label_ids[1, 0, :], label_ids):
            self.model.prototypes[label] = self.model.prototypes[label] * args.proto_m + (1 - args.proto_m) * feat
        self.model.prototypes = F.normalize(self.model.prototypes, dim=1)

        # Prototype-based Label Disambiguation
        # 1.find the closest prototype of each unlabeled sample
        _, proto_label_view1 = logits_prot_view1[~mask_lab, nlc:].max(dim=1)
        _, proto_label_view2 = logits_prot_view2[~mask_lab, nlc:].max(dim=1)

        # 2.construct pseudo labels
        proto_targets_unlab_view1 = (F.one_hot(proto_label_view1, num_classes=self.hparams.num_unlabeled_classes)
                                     .float().to(self.device))
        proto_targets_unlab_view2 = (F.one_hot(proto_label_view2, num_classes=self.hparams.num_unlabeled_classes)
                                     .float().to(self.device))

        # 3.construct targets for classifier output
        proto_targets = torch.zeros_like(logits)
        for v in range(self.hparams.num_views):
            for h in range(self.hparams.num_heads):
                proto_targets[v, h, mask_lab, :nlc] = targets_lab.type_as(proto_targets)
                if v == 0:
                    proto_targets[v, h, ~mask_lab, nlc:] = proto_targets_unlab_view1.type_as(proto_targets)
                else:
                    proto_targets[v, h, ~mask_lab, nlc:] = proto_targets_unlab_view2.type_as(proto_targets)

        loss_ce = self.swapped_prediction(logits, proto_targets)  # cross entropy loss
        loss = (loss_ce + loss_pcl + loss_ins) / 3  # total loss
        self.loss_per_head += loss_ce.clone().detach()  # update best head tracker

        return loss

    def validation_step(self, batch, batch_idx, dl_idx):
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        tag = self.trainer.datamodule.dataloader_mapping[dl_idx]

        # forward
        outputs = self.model(input_ids, input_mask, segment_ids, mode="eval")
        if "OOD" in tag:  # use clustering head
            preds_inc = torch.cat(
                [
                    outputs["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1),
                    outputs["logits_unlab"],
                ],
                dim=-1,
            )
        elif "IND" in tag:  # use supervised classifier
            best_head = torch.argmin(self.loss_per_head)
            preds_inc = torch.cat(
                [outputs["logits_lab"], outputs["logits_unlab"][best_head]], dim=-1
            )
        else:
            preds_inc = torch.cat(
                [
                    outputs["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1),
                    outputs["logits_unlab"],
                ],
                dim=-1,
            )
        preds_inc = preds_inc.max(dim=-1)[1]

        if dl_idx == 2:
            self.metrics_inc[dl_idx].update(preds_inc, label_ids, outputs["feats"])
        else:
            self.metrics_inc[dl_idx].update(preds_inc, label_ids)

    def validation_epoch_end(self, _):
        results_inc = [m.compute() for m in self.metrics_inc]
        best_head = results_inc[2]["SC"].index(max(results_inc[2]["SC"]))
        self.best_head = best_head
        val_acc = results_inc[2]["SC"][best_head]
        # log
        val_results = {
            "val/acc": val_acc,
        }
        print(results_inc)
        self.log_dict(val_results, on_step=False, on_epoch=True)
        self.log("val_acc", val_acc)
        return val_results

    def test_step(self, batch, batch_idx, dl_idx):
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        tag = self.trainer.datamodule.dataloader_mapping[dl_idx]

        # forward
        outputs = self.model(input_ids, input_mask, segment_ids, mode="eval")

        if "OOD" in tag:  # use clustering head
            preds_inc = torch.cat(
                [
                    outputs["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1),
                    outputs["logits_unlab"],
                ],
                dim=-1,
            )
        elif "IND" in tag:  # use supervised classifier
            best_head = torch.argmin(self.loss_per_head)
            preds_inc = torch.cat(
                [outputs["logits_lab"], outputs["logits_unlab"][best_head]], dim=-1
            )
        else:
            preds_inc = torch.cat(
                [
                    outputs["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1),
                    outputs["logits_unlab"],
                ],
                dim=-1,
            )
        preds_inc = preds_inc.max(dim=-1)[1]

        self.metrics_inc_test[dl_idx].update(preds_inc, label_ids)

    def test_epoch_end(self, _):
        results_inc = [m.compute() for m in self.metrics_inc_test]

        IND_acc = results_inc[0].cpu().numpy()
        OOD_acc = results_inc[1]["acc"][self.best_head].cpu().numpy()
        ALL_acc = results_inc[2]["acc"][self.best_head].cpu().numpy()
        OOD_f1 = results_inc[1]["f1"][self.best_head].cpu().numpy()
        ALL_f1 = results_inc[2]["f1"][self.best_head].cpu().numpy()

        self.test_results["IND_acc"] = IND_acc
        self.test_results["OOD_acc"] = OOD_acc
        self.test_results["ALL_acc"] = ALL_acc
        self.test_results["OOD_F1"] = OOD_f1
        self.test_results["ALL_F1"] = ALL_f1

        return self.test_results


def save_results(args, test_results):
    if not os.path.exists(args.save_results_path):
        os.makedirs(args.save_results_path)

    var = [args.dataset, args.num_labeled_classes, args.num_unlabeled_classes, args.divide_seed, args.batch_size,
           args.max_epochs]
    names = ['dataset', 'num_labeled_classes', 'num_unlabeled_classes', 'divide_seed', 'batch_size', 'max_epochs']
    vars_dict = {k: v for k, v in zip(names, var)}
    results = dict(test_results, **vars_dict)
    keys = list(results.keys())
    values = list(results.values())

    file_name = 'discovery_results.csv'
    results_path = os.path.join(args.save_results_path, file_name)

    if not os.path.exists(results_path):
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori, columns=keys)
        df1.to_csv(results_path, index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results, index=[1])
        df1 = df1.append(new, ignore_index=True)
        df1.to_csv(results_path, index=False)
    data_diagram = pd.read_csv(results_path)

    print('test_results', data_diagram)


def main(args):
    set_seed(args.seed, args.divide_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    global dm
    dm = get_datamodule(args, "discover")

    root_path = "./discovery_checkpoints"
    print(root_path)

    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max',
                                          dirpath=root_path)

    model = Discoverer(**args.__dict__)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(model, dm)
    test_model = Discoverer.load_from_checkpoint(checkpoint_path=trainer.checkpoint_callback.best_model_path)
    trainer.test(model=test_model, datamodule=dm)
    save_results(args, test_model.test_results)


if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    args.comment = 'dseed' + str(args.divide_seed) + '-' + str(args.num_labeled_classes) + '_' + str(
        args.num_unlabeled_classes)
    args.pretrained=os.path.join('pretrained_models',args.dataset,"-".join(["pretrain", args.arch, args.dataset, args.comment]))+ ".cp"
    # args.pretrained=os.path.join('pretrained_models',"-".join(["pretrain", args.arch, args.dataset, args.comment]))+ ".cp"
    main(args)
