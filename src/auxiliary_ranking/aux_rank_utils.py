import numpy as np
from dataclasses import dataclass
from typing import Iterable, Optional, Union, NamedTuple

from itertools import zip_longest

import torch
from torch import Tensor, from_numpy
from torch.utils.data import DataLoader

from chemprop.data.molgraph import MolGraph
from chemprop.data.collate import BatchMolGraph
from chemprop.data.datasets import MoleculeDataset
from chemprop.data.samplers import ClassBalanceSampler
from chemprop.data.datapoints import _DatapointMixin, _MoleculeDatapointMixin

class Datum(NamedTuple):
    """a singular training data point"""
    mg: MolGraph
    V_d: np.ndarray | None
    x_d: np.ndarray | None
    y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None
    aux_mask: np.ndarray | None


class TrainingBatch(NamedTuple):
    bmg: BatchMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None
    aux_mask: Tensor | None


@dataclass
class CustomMoleculeDataset(MoleculeDataset):
    def __getitem__(self, idx: int) -> Datum:
        d = self.data[idx]
        mg = self.mg_cache[idx]
        return Datum(mg, self.V_ds[idx], self.X_d[idx], self.Y[idx], d.weight, d.lt_mask, d.gt_mask, d.aux_mask)


def custom_collate_batch(batch: Iterable[Datum]) -> TrainingBatch:
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks, aux_masks = zip(*batch)
    return TrainingBatch(BatchMolGraph(mgs), 
                         None if V_ds[0] is None else from_numpy(np.concatenate(V_ds)).float(),
                         None if x_ds[0] is None else from_numpy(np.array(x_ds)).float(),
                         None if ys[0] is None else from_numpy(np.array(ys)).float(), torch.tensor(weights, dtype=torch.float).unsqueeze(1),
                         None if lt_masks[0] is None else from_numpy(np.array(lt_masks)),
                         None if gt_masks[0] is None else from_numpy(np.array(gt_masks)),
                         None if aux_masks[0] is None else from_numpy(np.array(aux_masks)),
                        )


@dataclass
class MoleculeDatapoint(_DatapointMixin, _MoleculeDatapointMixin):
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets."""

    V_f: Optional[np.ndarray] = None
    """A numpy array of shape ``V x d_vf``, where ``V`` is the number of atoms in the molecule, and
    ``d_vf`` is the number of additional features that will be concatenated to atom-level features
    *before* message passing"""
    E_f: Optional[np.ndarray] = None
    """A numpy array of shape ``E x d_ef``, where ``E`` is the number of bonds in the molecule, and
    ``d_ef`` is the number of additional features  containing additional features that will be
    concatenated to bond-level features *before* message passing"""
    V_d: Optional[np.ndarray] = None
    """A numpy array of shape ``V x d_vd``, where ``V`` is the number of atoms in the molecule, and
    ``d_vd`` is the number of additional descriptors that will be concatenated to atom-level
    descriptors *after* message passing"""
    aux_mask: Optional[np.ndarray] = None
    # """Indicates whether the datapoint should be rank 1 or rank 0 in pairwise ranking, two 1s need to be checked with true value, 0s are all inactives and should only be pairwise compared to 1s"""

    def __post_init__(self):
        NAN_TOKEN = 0
        if self.V_f is not None:
            self.V_f[np.isnan(self.V_f)] = NAN_TOKEN
        if self.E_f is not None:
            self.E_f[np.isnan(self.E_f)] = NAN_TOKEN
        if self.V_d is not None:
            self.V_d[np.isnan(self.V_d)] = NAN_TOKEN
        # if self.aux_mask is not None:
        #     self.aux_mask[np.isnan(self.aux_mask)] = NAN_TOKEN
        super().__post_init__()

    def __len__(self) -> int:
        return 1

'''
class ClassBalanceSampler(Sampler):
    """Samples data from a dataset such that classes are sampled according to specified ratios.

    Parameters
    ----------
    Y : np.ndarray
        1D array of class labels, shape (N,), OR a 2D one-hot/multi-label array,
        shape (N, C), where each row belongs to exactly one class (its argmax column).
    seed : int, optional
        random seed used for shuffling (only used when `shuffle=True`)
    shuffle : bool, default=False
        whether to shuffle indices within each class before sampling
    ratios : dict[label, float] or list[float], optional
        relative sampling ratio per class. If None, all classes are weighted equally (1:1:...:1).
        Ratios don't need to sum to 1 — they're relative weights, normalized internally
        against whichever class runs out first.
    """

    def __init__(
        self,
        Y: np.ndarray,
        seed: Optional[int] = None,
        shuffle: bool = False,
        ratios: Optional[Union[dict, list, tuple]] = None,
    ):
        self.shuffle = shuffle
        self.rg = np.random.default_rng(seed)

        Y = np.asarray(Y)
        idxs = np.arange(len(Y))
        labels = Y if Y.ndim == 1 else Y.argmax(1)

        self.classes = np.unique(labels)
        self.class_idxs = {c: idxs[labels == c] for c in self.classes}

        if ratios is None:
            ratios = {c: 1.0 for c in self.classes}
        elif isinstance(ratios, (list, tuple)):
            if len(ratios) != len(self.classes):
                raise ValueError(
                    f"length of `ratios` ({len(ratios)}) must match number of classes ({len(self.classes)})"
                )
            ratios = dict(zip(self.classes, ratios))
        else:
            missing = set(self.classes) - set(ratios)
            if missing:
                raise ValueError(f"`ratios` is missing entries for classes: {missing}")

        self.ratios = ratios

        # find the largest k such that k * ratio[c] <= available samples, for every class
        k = min(len(self.class_idxs[c]) / self.ratios[c] for c in self.classes)
        self.samples_per_class = {c: int(np.floor(k * self.ratios[c])) for c in self.classes}
        self.length = sum(self.samples_per_class.values())

    def __iter__(self) -> Iterator[int]:
        """an iterator over indices to sample."""
        per_class = {}
        for c in self.classes:
            idxs = self.class_idxs[c]
            if self.shuffle:
                idxs = idxs.copy()
                self.rg.shuffle(idxs)
            per_class[c] = idxs[: self.samples_per_class[c]]

        # round-robin interleave across classes so batches stay mixed rather than
        # block-ordered by class
        groups = [per_class[c] for c in self.classes]
        return iter(idx for group in zip_longest(*groups) for idx in group if idx is not None)

    def __len__(self) -> int:
        """the number of indices that will be sampled."""
        return self.length
'''        

def custom_build_dataloader(
    dataset: CustomMoleculeDataset,
    batch_size: int = 64,
    num_workers: int = 0,
    class_balance: bool = False,
    seed: int | None = None,
    shuffle: bool = True,
    drop_last: bool | None = None,
    **kwargs,):
    """Return a :obj:`~torch.utils.data.DataLoader` for :class:`MolGraphDataset`\s

    Parameters
    ----------
    dataset : MoleculeDataset | ReactionDataset | MulticomponentDataset
        The dataset containing the molecules or reactions to load.
    batch_size : int, default=64
        the batch size to load.
    num_workers : int, default=0
        the number of workers used to build batches.
    class_balance : bool, default=False
        Whether to perform class balancing (i.e., use an equal number of positive and negative
        molecules), in this custom builder adapted to force an equal number of regression and classification-only datapoints. 
    seed : int, default=None
        the random seed to use for shuffling (only used when `shuffle` is `True`).
    shuffle : bool, default=False
        whether to shuffle the data during sampling.
    drop_last : bool, default=None
        Whether to drop the last batch if it is of size 1 (needed if using batchnorm during training).
        If None, this will be set automatically.
    """
    if class_balance:
        if dataset.aux_mask is not None:
            # balancing_class = np.isnan(dataset.y) # balancing based on whether datapoints have value for regression loss calculation or not, aim for a 50/50 split of regression+ranking v ranking datapoints only. 
            balancing_class = np.isnan(dataset.aux_mask[:, 1]) & np.isnan(dataset.aux_mask[:, 2])
            
            sampler = ClassBalanceSampler(balancing_class.astype(float), seed, shuffle)
    else:
        sampler = None

    collate_fn = custom_collate_batch

    if drop_last is None:
        if len(dataset) % batch_size == 1:
            logger.warning(
                f"Dropping last batch of size 1 to avoid issues with batch normalization \
    (dataset size = {len(dataset)}, batch_size = {batch_size})"
            )
            drop_last = True
        else:
            drop_last = False

    return DataLoader(
        dataset,
        batch_size,
        sampler is None and shuffle,
        sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        **kwargs,
    )