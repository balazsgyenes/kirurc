from __future__ import annotations

import json
import multiprocessing as mp
import os.path as osp
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
from packaging import version
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import Compose

if version.parse(torch.__version__) >= version.parse("2.6.0"):
    TORCH_LOAD_ARGS = {"weights_only": False}
else:
    TORCH_LOAD_ARGS = {}


class KirurcDataset(Dataset):
    _processed_file_names: list[str]

    def __init__(
        self,
        root: Path,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        take_first_n: int | None = None,
        num_workers: int | None = None,
        force_reprocess: bool = False,
        debug_transforms: bool = False,
    ) -> None:
        self.take_first_n = take_first_n
        self.num_workers = num_workers
        self.force_reprocess = force_reprocess

        if debug_transforms and pre_transform is not None:
            if transform is not None:
                transform = Compose([pre_transform, transform])
            else:
                transform = pre_transform
            pre_transform = None
        self.debug_transforms = debug_transforms

        super().__init__(str(root), transform, pre_transform, pre_filter)

        try:
            with open(self.counts_file, "r") as f:
                class_counts = json.load(f)

            class_counts = {int(cls): count for cls, count in class_counts.items()}
            self._n_classes = max(class_counts) + 1
            self._counts = torch.zeros((self._n_classes,), dtype=torch.long)
            for cls, count in class_counts.items():
                self._counts[cls] = count

        except FileNotFoundError:
            # maybe we don't need this information
            # in this case, throw an error if we try to access self.counts or
            # self.n_classes
            pass

    def _verify_files(self) -> None:
        raise NotImplementedError

    def len(self) -> int:
        return len(self.processed_file_names)

    def _process(self):
        # delete processed data and force reprocess, if requested
        if self.force_reprocess and osp.exists(self.processed_dir):
            shutil.rmtree(self.processed_dir)

        # before starting processing, create list of processed_file_names
        self._verify_files()

        return super()._process()

    def process(self) -> None:
        if self.debug_transforms:
            return

        total_class_counts = defaultdict(int)
        num_workers = mp.cpu_count() if self.num_workers is None else self.num_workers

        if num_workers > 0:
            ctx = mp.get_context("spawn")
            chunksize = max(1, self.len() // num_workers)
            with ctx.Pool(num_workers) as pool:
                class_counts_per_file = pool.imap_unordered(
                    self._process_one,
                    self.indices(),
                    chunksize=chunksize,
                )

                # TODO: fix naming here
                for class_counts in class_counts_per_file:
                    for cls, count in class_counts.items():
                        total_class_counts[cls] += count

        else:
            for idx in self.indices():
                class_counts = self._process_one(idx)
                for cls, count in class_counts.items():
                    total_class_counts[cls] += count

        with open(self.counts_file, "w") as f:
            json.dump(total_class_counts, f)

    def _process_one(self, idx: int) -> dict[int, int]:
        sample = self._load_one(idx)

        if self.pre_filter is not None and self.pre_filter(sample):
            return {}

        if self.pre_transform is not None:
            sample = self.pre_transform(sample)

        classes, counts = torch.unique(sample.y, return_counts=True)
        class_counts = {int(cls): int(count) for cls, count in zip(classes, counts)}

        # compress y to reduce space on disk
        if "y" in sample:
            sample["y"] = sample["y"].to(torch.uint8)

        # remove filename which is redundant
        sample.pop("file", None)

        # save to pt file using pickle
        torch.save(sample, osp.join(self.processed_dir, self.processed_file_names[idx]))

        return class_counts

    def _load_one(self, idx: int) -> Data:
        raise NotImplementedError

    @property
    def counts_file(self) -> str:
        return osp.join(self.processed_dir, "class_counts.json")

    @property
    def processed_file_names(self) -> list[str]:
        return self._processed_file_names

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def class_counts(self) -> torch.Tensor:
        return self._counts

    def get(self, idx: int) -> Data:
        if self.debug_transforms:
            return self._load_one(idx)

        file = Path(self.processed_dir) / self.processed_file_names[idx]

        sample = torch.load(file, **TORCH_LOAD_ARGS)

        sample["file"] = file

        if "y" in sample:
            sample["y"] = sample["y"].to(torch.long)
        return sample
