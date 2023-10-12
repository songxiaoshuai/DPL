import os
import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.metrics import Accuracy

from utils.callbacks import PretrainCheckpointCallback
from utils.data import get_datamodule
from utils.nets import MultiHeadBERT

parser = ArgumentParser()
# dataset
parser.add_argument("--dataset", required=True, type=str, help="dataset")
parser.add_argument("--data_dir", default="datasets", type=str, help="data directory")
parser.add_argument("--num_labeled_classes", required=True, type=int, help="number of labeled classes")
parser.add_argument("--num_unlabeled_classes", required=True, type=int, help="number of unlabeled classes")
parser.add_argument("--divide_seed", required=True, type=int, help="seed for IND and OOD classes division")

# model
parser.add_argument("--arch", default="bert-base-uncased", type=str, help="backbone architecture")
parser.add_argument("--pretrained", type=str, default=None, help="pretrained checkpoint path")

# pretrain
parser.add_argument("--batch_size", default=256, type=int, help="batch size")
parser.add_argument("--base_lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
parser.add_argument("--weight_decay_opt", default=1.0e-4, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")

# output
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--checkpoint_dir", default="pretrained_models", type=str, help="pretrained checkpoint dir")
parser.add_argument("--comment", default='', type=str)
parser.add_argument("--save_results_path", type=str, default='results', help="The path to save csv results.")


def set_seed(seed, divide_seed):
    random.seed(seed)
    np.random.seed(divide_seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Pretrainer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})

        # build model
        self.model = MultiHeadBERT.from_pretrained(
            self.hparams.arch,
            self.hparams.num_labeled_classes,
            self.hparams.num_unlabeled_classes,
            num_heads=None,
        )
        self.freeze_parameters(self.model)

        if self.hparams.pretrained is not None:
            state_dict = torch.load(self.hparams.pretrained)
            self.model.load_state_dict(state_dict, strict=False)

        # metrics
        self.accuracy = Accuracy()

        self.test_results = 0
        self.t_step = 0

        self.OOD_features = torch.empty((0, 768)).to(self.device)
        self.OOD_labels = torch.empty(0, dtype=torch.long).to(self.device)

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

    def training_step(self, batch, batch_idx):
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # normalize prototypes
        self.model.normalize_prototypes()

        # forward
        outputs = self.model(input_ids, input_mask, segment_ids, mode="pretrain")

        # supervised loss
        loss_supervised = F.cross_entropy(outputs["logits_lab"] / self.hparams.temperature, label_ids)

        return loss_supervised

    def validation_step(self, batch, batch_idx):
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # forward
        logits = self.model(input_ids, input_mask, segment_ids, mode="eval")["logits_lab"]
        _, preds = logits.max(dim=-1)

        # calculate loss and accuracy
        loss_supervised = F.cross_entropy(logits, label_ids)
        acc = self.accuracy(preds, label_ids)

        results = {
            "val/loss_supervised": loss_supervised,
            "val/acc": acc,
        }
        return results

    def test_step(self, batch, batch_idx):
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        batch_num = label_ids.shape[0]

        # forward
        logits = self.model(input_ids, input_mask, segment_ids, mode="eval")["logits_lab"]
        _, preds = logits.max(dim=-1)

        # calculate loss and accuracy
        loss_supervised = F.cross_entropy(logits, label_ids)
        acc = self.accuracy(preds, label_ids)

        results = {
            "test/loss_supervised": loss_supervised,
            "test/acc": acc,
        }
        print(results)
        self.t_step += batch_num
        self.test_results += results['test/acc'] * batch_num
        return results


def test(args):
    set_seed(0, args.divide_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # build datamodule
    dm = get_datamodule(args, "pretrain")

    test_model = Pretrainer(**args.__dict__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_file = os.path.join(args.checkpoint_dir,args.dataset,
                                 "-".join(["pretrain", args.arch, args.dataset, args.comment]) + ".cp")
    state_dict = torch.load(pretrain_file, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}
    test_model.model.load_state_dict(state_dict, strict=False)

    trainer = pl.Trainer(gpus=1, precision=16)
    trainer.test(model=test_model, datamodule=dm)

    t_acc = test_model.test_results / test_model.t_step
    print("testing results:", t_acc)
    test_results = {}
    test_results["pretrain_acc"] = t_acc.cpu().numpy()

    save_results(args, test_results)


def save_results(args, test_results):
    if not os.path.exists(args.save_results_path):
        os.makedirs(args.save_results_path)

    var = [args.dataset, args.num_labeled_classes, args.num_unlabeled_classes, args.batch_size, args.max_epochs]
    names = ['dataset', 'num_labeled_classes', 'num_unlabeled_classes', 'batch_size', 'max_epochs']
    vars_dict = {k: v for k, v in zip(names, var)}
    results = dict(test_results, **vars_dict)
    keys = list(results.keys())
    values = list(results.values())

    file_name = 'results_pretrain.csv'
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
    set_seed(0, args.divide_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # build datamodule
    dm = get_datamodule(args, "pretrain")



    model = Pretrainer(**args.__dict__)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[PretrainCheckpointCallback()]
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.comment = 'dseed' + str(args.divide_seed) + '-' + str(args.num_labeled_classes) + '_' + str(
                args.num_unlabeled_classes)
    main(args)
    test(args)
