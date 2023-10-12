import csv
import os

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data import Dataset

from utils.util import cross_domain_division


class OriginSamples(Dataset):
    def __init__(self, train_x, train_y):
        assert len(train_y) == len(train_x)
        self.train_x = train_x
        self.train_y = train_y


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    content_list = examples.train_x
    label_list = examples.train_y

    for i in range(len(content_list)):
        tokens_a = tokenizer.tokenize(content_list[i])

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[label_list[i]]

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def get_datamodule(args, mode):
    if mode == "pretrain":
        if args.dataset == "GID-SD":
            return PretrainBankingDataModule(args)
        elif args.dataset == "GID-MD":
            return PretrainClincDataModule(args)
        elif args.dataset == "GID-CD":
            return PretrainClincDataModule(args)
        else:
            raise ValueError()
    elif mode == "discover":
        if args.dataset == "GID-SD":
            return DiscoverBankingDataModule(args)
        elif args.dataset == "GID-MD":
            return DiscoverClincDataModule(args)
        elif args.dataset == "GID-CD":
            return DiscoverClincDataModule(args)
        else:
            raise ValueError()


class PretrainClincDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.bert_model = args.arch
        self.max_seq_length = 30
        self.all_label_list = self.get_labels(self.data_dir)

        if args.dataset == "GID-MD":
            self.IND_class = list(
                np.random.choice(np.array(self.all_label_list), self.num_labeled_classes, replace=False))   
            self.OOD_class = list(set(self.all_label_list).difference(set(self.IND_class))) 
        elif args.dataset == "GID-CD":
            if args.num_unlabeled_classes == 30:
                if args.divide_seed==10:
                    self.IND_class, self.OOD_class = cross_domain_division(
                        IND_domains=["credit_cards", "travel", "home", "meta", "utility", "small_talk", "auto_and_commute", "work"],
                        OOD_domain=["kitchen_and_dining", "banking"])
                elif args.divide_seed==20:
                    self.IND_class, self.OOD_class = cross_domain_division(
                        IND_domains=["credit_cards", "travel", "home", "meta", "utility", "small_talk", "kitchen_and_dining", "banking"],
                        OOD_domain=["work", "auto_and_commute"])
                elif args.divide_seed==30:
                    self.IND_class, self.OOD_class = cross_domain_division(
                        IND_domains=["credit_cards", "meta", "utility", "small_talk", "kitchen_and_dining", "banking", "work", "auto_and_commute"],
                        OOD_domain=["travel","home"])
                else:
                    raise ValueError()

            if args.num_unlabeled_classes == 60:
                if args.divide_seed==10:
                    self.IND_class, self.OOD_class = cross_domain_division(
                        IND_domains=["credit_cards", "meta", "utility", "small_talk", "work", "auto_and_commute"],
                        OOD_domain=["travel", "home", "kitchen_and_dining", "banking"])
                elif args.divide_seed==20:
                    self.IND_class, self.OOD_class = cross_domain_division(
                        IND_domains=["credit_cards", "meta", "utility", "small_talk", "kitchen_and_dining", "banking"],
                        OOD_domain=["travel", "home", "auto_and_commute", "work"])
                elif args.divide_seed==30:
                    self.IND_class, self.OOD_class = cross_domain_division(
                        IND_domains=["travel", "home", "meta", "utility", "small_talk", "kitchen_and_dining"],
                        OOD_domain=["credit_cards", "banking", "auto_and_commute", "work"])
                else:
                    raise ValueError()

            if args.num_unlabeled_classes == 90:
                if args.divide_seed==10:
                    self.IND_class, self.OOD_class = cross_domain_division(
                        IND_domains=["credit_cards", "meta", "utility", "small_talk"],
                        OOD_domain=["travel", "home", "kitchen_and_dining", "banking","auto_and_commute","work"])
                else:
                    raise ValueError()
        else:
            raise ValueError()



        assert len(self.IND_class) == self.num_labeled_classes
        assert len(self.OOD_class) == self.num_unlabeled_classes

        self.all_label_list = []
        self.all_label_list.extend(self.IND_class)
        self.all_label_list.extend(self.OOD_class)

    def get_labels(self, data_dir):
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def get_datasets(self, data_dir, quotechar=None):
        with open(data_dir, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i = 0
            for line in reader:
                if i == 0:
                    i += 1
                    continue
                line[0] = line[0].strip()
                lines.append(line)
            return lines

    def divide_datasets(self, origin_data):
        labeled_examples, unlabeled_examples = [], []
        for example in origin_data:
            if example[-1] in self.IND_class:
                labeled_examples.append(example)
            elif example[-1] in self.OOD_class:
                unlabeled_examples.append(example)
        return labeled_examples, unlabeled_examples

    def get_samples(self, labelled_examples):
        content_list, labels_list = [], []
        for example in labelled_examples:
            text = example[0]
            label = example[-1]
            content_list.append(text)
            labels_list.append(label)

        data = OriginSamples(content_list, labels_list)

        return data

    def get_loader(self, labelled_examples, label_list, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)
        features = convert_examples_to_features(labelled_examples, label_list, self.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == "train":
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "validation":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "test":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train.tsv")
        val_data_dir = os.path.join(self.data_dir, "eval.tsv")
        test_data_dir = os.path.join(self.data_dir, "test.tsv")

        train_set = self.get_datasets(train_data_dir)
        val_set = self.get_datasets(val_data_dir)
        test_set = self.get_datasets(test_data_dir)

        train_ind, train_ood = self.divide_datasets(train_set)
        val_ind, val_ood = self.divide_datasets(val_set)
        test_ind, test_ood = self.divide_datasets(test_set)


        print("the numbers of all train samples: ", len(train_set))
        print("the numbers of all validation samples: ", len(val_set))
        print("the numbers of all test samples: ", len(test_set))
        print("the numbers of IND/OOD train samples: ", len(train_ind), len(train_ood))
        print("the numbers of IND/OOD validation samples: ", len(val_ind), len(val_ood))
        print("the numbers of IND/OOD test samples: ", len(test_ind), len(test_ood))

        self.train_ind = self.get_samples(train_ind)
        self.val_ind = self.get_samples(val_ind)
        self.test_ind = self.get_samples(test_ind)

    def train_dataloader(self):
        return self.get_loader(self.train_ind, self.IND_class, mode="train")

    def val_dataloader(self):
        return self.get_loader(self.val_ind, self.IND_class, mode="validation")

    def test_dataloader(self):
        return self.get_loader(self.test_ind, self.IND_class, mode="test")


class DiscoverClincDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.bert_model = args.arch
        self.max_seq_length = 30
        self.all_label_list = self.get_labels(self.data_dir)

        if args.dataset == "GID-MD":
            self.IND_class = list(
                np.random.choice(np.array(self.all_label_list), self.num_labeled_classes, replace=False))
            self.OOD_class = list(set(self.all_label_list).difference(set(self.IND_class)))

        elif args.dataset == "GID-CD":
            if args.num_unlabeled_classes == 30:
                if args.divide_seed==10:
                    self.IND_class, self.OOD_class = cross_domain_division(
                        IND_domains=["credit_cards", "travel", "home", "meta", "utility", "small_talk", "auto_and_commute", "work"],
                        OOD_domain=["kitchen_and_dining", "banking"])
                elif args.divide_seed==20:
                    self.IND_class, self.OOD_class = cross_domain_division(
                        IND_domains=["credit_cards", "travel", "home", "meta", "utility", "small_talk", "kitchen_and_dining", "banking"],
                        OOD_domain=["work", "auto_and_commute"])
                elif args.divide_seed==30:
                    self.IND_class, self.OOD_class = cross_domain_division(
                        IND_domains=["credit_cards", "meta", "utility", "small_talk", "kitchen_and_dining", "banking", "work", "auto_and_commute"],
                        OOD_domain=["travel","home"])
                else:
                    raise ValueError()

            if args.num_unlabeled_classes == 60:
                if args.divide_seed==10:
                    self.IND_class, self.OOD_class = cross_domain_division(
                        IND_domains=["credit_cards", "meta", "utility", "small_talk", "work", "auto_and_commute"],
                        OOD_domain=["travel", "home", "kitchen_and_dining", "banking"])
                elif args.divide_seed==20:
                    self.IND_class, self.OOD_class = cross_domain_division(
                        IND_domains=["credit_cards", "meta", "utility", "small_talk", "kitchen_and_dining", "banking"],
                        OOD_domain=["travel", "home", "auto_and_commute", "work"])
                elif args.divide_seed==30:
                    self.IND_class, self.OOD_class = cross_domain_division(
                        IND_domains=["travel", "home", "meta", "utility", "small_talk", "kitchen_and_dining"],
                        OOD_domain=["credit_cards", "banking", "auto_and_commute", "work"])
                else:
                    raise ValueError()

            if args.num_unlabeled_classes == 90:
                if args.divide_seed==10:
                    self.IND_class, self.OOD_class = cross_domain_division(
                        IND_domains=["credit_cards", "meta", "utility", "small_talk"],
                        OOD_domain=["travel", "home", "kitchen_and_dining", "banking","auto_and_commute","work"])
                else:
                    raise ValueError()
        else:
            raise ValueError()

        assert len(self.IND_class) == self.num_labeled_classes
        assert len(self.OOD_class) == self.num_unlabeled_classes


        self.all_label_list = []
        self.all_label_list.extend(self.IND_class)
        self.all_label_list.extend(self.OOD_class)

    def get_labels(self, data_dir):
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels


    def get_datasets(self, data_dir, quotechar=None):
        with open(data_dir, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i = 0
            for line in reader:
                if i == 0:
                    i += 1
                    continue
                line[0] = line[0].strip()

                if line[-1] in self.all_label_list:
                    lines.append(line)
            return lines

    def divide_datasets(self, origin_data):
        labeled_examples, unlabeled_examples = [], []
        for example in origin_data:
            if example[-1] in self.IND_class:
                labeled_examples.append(example)
            elif example[-1] in self.OOD_class:
                unlabeled_examples.append(example)
        return labeled_examples, unlabeled_examples

    def get_samples(self, labelled_examples):
        content_list, labels_list = [], []
        for example in labelled_examples:
            text = example[0]
            label = example[-1]
            content_list.append(text)
            labels_list.append(label)

        data = OriginSamples(content_list, labels_list)

        return data

    def get_loader(self, labelled_examples, label_list, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)
        features = convert_examples_to_features(labelled_examples, label_list, self.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == "train":
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "validation":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "test":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train.tsv")
        val_data_dir = os.path.join(self.data_dir, "eval.tsv")
        test_data_dir = os.path.join(self.data_dir, "test.tsv")

        train_set = self.get_datasets(train_data_dir)
        val_set = self.get_datasets(val_data_dir)
        test_set = self.get_datasets(test_data_dir)

        train_ind, train_ood = self.divide_datasets(train_set)
        val_ind, val_ood = self.divide_datasets(val_set)
        test_ind, test_ood = self.divide_datasets(test_set)

        train_set = []
        train_set.extend(train_ind)
        train_set.extend(train_ood)

        print("the numbers of all train samples: ", len(train_set))
        print("the numbers of all validation samples: ", len(val_set))
        print("the numbers of all test samples: ", len(test_set))
        print("the numbers of IND/OOD train samples: ", len(train_ind), len(train_ood))
        print("the numbers of IND/OOD validation samples: ", len(val_ind), len(val_ood))
        print("the numbers of IND/OOD test samples: ", len(test_ind), len(test_ood))

        self.train_all = self.get_samples(train_set)

        self.val_ind = self.get_samples(val_ind)
        self.val_ood = self.get_samples(val_ood)
        self.val_all = self.get_samples(val_set)

        self.test_ind = self.get_samples(test_ind)
        self.test_ood = self.get_samples(test_ood)
        self.test_all = self.get_samples(test_set)

    @property
    def dataloader_mapping(self):
        return {0: "IND", 1: "OOD", 2: "ALL"}

    def train_dataloader(self):
        return self.get_loader(self.train_all, self.all_label_list, mode="train")

    def val_dataloader(self):
        val_ind_loader = self.get_loader(self.val_ind, self.IND_class, mode="validation")
        val_ood_loader = self.get_loader(self.val_ood, self.OOD_class, mode="validation")
        val_loader = self.get_loader(self.val_all, self.all_label_list, mode="validation")
        return [val_ind_loader, val_ood_loader, val_loader]

    def test_dataloader(self):
        test_ind_loader = self.get_loader(self.test_ind, self.IND_class, mode="validation")
        test_ood_loader = self.get_loader(self.test_ood, self.OOD_class, mode="validation")
        test_loader = self.get_loader(self.test_all, self.all_label_list, mode="test")
        return [test_ind_loader, test_ood_loader, test_loader]


class PretrainBankingDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_labeled_classes = args.num_labeled_classes  # number of IND classes
        self.num_unlabeled_classes = args.num_unlabeled_classes  # number of OOD classes
        self.bert_model = args.arch  # BERT backbone
        self.max_seq_length = 55
        self.all_label_list = self.get_labels(self.data_dir)  # Get all classes labels

        self.IND_class = list(
            np.random.choice(np.array(self.all_label_list), self.num_labeled_classes, replace=False))  # list of IND classes
        self.OOD_class = list(set(self.all_label_list).difference(set(self.IND_class)))  # list of OOD classes

        assert len(self.IND_class) == self.num_labeled_classes
        assert len(self.OOD_class) == self.num_unlabeled_classes

    def get_labels(self, data_dir):
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def get_datasets(self, data_dir, quotechar=None):
        with open(data_dir, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i = 0
            for line in reader:
                if i == 0:
                    i += 1
                    continue
                line[0] = line[0].strip()
                lines.append(line)
            return lines

    def divide_datasets(self, origin_data):
        labeled_examples, unlabeled_examples = [], []
        for example in origin_data:
            if example[-1] in self.IND_class:
                labeled_examples.append(example)
            elif example[-1] in self.OOD_class:
                unlabeled_examples.append(example)
        return labeled_examples, unlabeled_examples

    def get_samples(self, labelled_examples):
        content_list, labels_list = [], []
        for example in labelled_examples:
            text = example[0]
            label = example[-1]
            content_list.append(text)
            labels_list.append(label)

        data = OriginSamples(content_list, labels_list)

        return data

    def get_loader(self, labelled_examples, label_list, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)
        features = convert_examples_to_features(labelled_examples, label_list, self.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == "train":
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "validation":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "test":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train.tsv")
        val_data_dir = os.path.join(self.data_dir, "eval.tsv")
        test_data_dir = os.path.join(self.data_dir, "test.tsv")

        train_set = self.get_datasets(train_data_dir)
        val_set = self.get_datasets(val_data_dir)
        test_set = self.get_datasets(test_data_dir)

        train_ind, train_ood = self.divide_datasets(train_set)
        val_ind, val_ood = self.divide_datasets(val_set)
        test_ind, test_ood = self.divide_datasets(test_set)
        print("the numbers of all train samples: ", len(train_set))
        print("the numbers of all validation samples: ", len(val_set))
        print("the numbers of all test samples: ", len(test_set))
        print("the numbers of IND/OOD train samples: ", len(train_ind), len(train_ood))
        print("the numbers of IND/OOD validation samples: ", len(val_ind), len(val_ood))
        print("the numbers of IND/OOD test samples: ", len(test_ind), len(test_ood))

        self.train_ind = self.get_samples(train_ind)
        self.val_ind = self.get_samples(val_ind)
        self.test_ind = self.get_samples(test_ind)

    def train_dataloader(self):
        return self.get_loader(self.train_ind, self.IND_class, mode="train")

    def val_dataloader(self):
        return self.get_loader(self.val_ind, self.IND_class, mode="validation")

    def test_dataloader(self):
        return self.get_loader(self.test_ind, self.IND_class, mode="test")


class DiscoverBankingDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.bert_model = args.arch
        self.max_seq_length = 55
        self.all_label_list = self.get_labels(self.data_dir)

        self.IND_class = list(
            np.random.choice(np.array(self.all_label_list), self.num_labeled_classes, replace=False))  
        self.OOD_class = list(set(self.all_label_list).difference(set(self.IND_class))) 

        self.all_label_list = []
        self.all_label_list.extend(self.IND_class)
        self.all_label_list.extend(self.OOD_class)

        assert len(self.IND_class) == self.num_labeled_classes
        assert len(self.OOD_class) == self.num_unlabeled_classes

    def get_loader_pre(self):
        train_data_dir = os.path.join(self.data_dir, "train.tsv")
        train_set = self.get_datasets(train_data_dir)
        train_all = self.get_samples(train_set)
        return self.get_loader(train_all, self.all_label_list, mode="train")

    def get_labels(self, data_dir):
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def get_datasets(self, data_dir, quotechar=None):
        with open(data_dir, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i = 0
            for line in reader:
                if i == 0:
                    i += 1
                    continue
                line[0] = line[0].strip()
                lines.append(line)
            return lines

    def divide_datasets(self, origin_data):
        labeled_examples, unlabeled_examples = [], []
        for example in origin_data:
            if example[-1] in self.IND_class:
                labeled_examples.append(example)
            elif example[-1] in self.OOD_class:
                unlabeled_examples.append(example)
        return labeled_examples, unlabeled_examples

    def get_samples(self, labelled_examples):
        content_list, labels_list = [], []
        for example in labelled_examples:
            text = example[0]
            label = example[-1]
            content_list.append(text)
            labels_list.append(label)

        data = OriginSamples(content_list, labels_list)

        return data

    def get_loader(self, labelled_examples, label_list, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)
        features = convert_examples_to_features(labelled_examples, label_list, self.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        if mode == "train":
            data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        else:
            data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == "train":
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "validation":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "test":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train.tsv")
        val_data_dir = os.path.join(self.data_dir, "eval.tsv")
        test_data_dir = os.path.join(self.data_dir, "test.tsv")

        train_set = self.get_datasets(train_data_dir)
        val_set = self.get_datasets(val_data_dir)
        test_set = self.get_datasets(test_data_dir)

        train_ind, train_ood = self.divide_datasets(train_set)
        val_ind, val_ood = self.divide_datasets(val_set)
        test_ind, test_ood = self.divide_datasets(test_set)
        print("the numbers of all train samples: ", len(train_set))
        print("the numbers of all validation samples: ", len(val_set))
        print("the numbers of all test samples: ", len(test_set))
        print("the numbers of IND/OOD train samples: ", len(train_ind), len(train_ood))
        print("the numbers of IND/OOD validation samples: ", len(val_ind), len(val_ood))
        print("the numbers of IND/OOD test samples: ", len(test_ind), len(test_ood))

        self.train_all = self.get_samples(train_set)

        self.val_ind = self.get_samples(val_ind)
        self.val_ood = self.get_samples(val_ood)
        self.val_all = self.get_samples(val_set)

        self.test_ind = self.get_samples(test_ind)
        self.test_ood = self.get_samples(test_ood)
        self.test_all = self.get_samples(test_set)

    @property
    def dataloader_mapping(self):
        return {0: "IND", 1: "OOD", 2: "ALL"}

    def train_dataloader(self):
        return self.get_loader(self.train_all, self.all_label_list, mode="train")

    def val_dataloader(self):
        val_ind_loader = self.get_loader(self.val_ind, self.IND_class, mode="validation")
        val_ood_loader = self.get_loader(self.val_ood, self.OOD_class, mode="validation")
        val_loader = self.get_loader(self.val_all, self.all_label_list, mode="validation")
        return [val_ind_loader, val_ood_loader, val_loader]

    def test_dataloader(self):
        test_ind_loader = self.get_loader(self.test_ind, self.IND_class, mode="validation")
        test_ood_loader = self.get_loader(self.test_ood, self.OOD_class, mode="validation")
        test_loader = self.get_loader(self.test_all, self.all_label_list, mode="test")
        return [test_ind_loader, test_ood_loader, test_loader]
