#!/usr/bin/env python
# coding: utf-8

from data_loaders import DATASETS, dataset_loader


if __name__ == "__main__":
    for dataset in DATASETS:
        dataset_loader(dataset)
