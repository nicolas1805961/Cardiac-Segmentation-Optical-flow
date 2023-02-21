#!/usr/env/bin python3.9

import io
import re
import pickle
import random
from pathlib import Path
from itertools import repeat
from operator import itemgetter, mul
from functools import partial, reduce
from typing import Callable, BinaryIO, Match, Pattern, Tuple, Union, Optional

import torch
import numpy as np
from PIL import Image, ImageOps
from torch import Tensor
from torchvision import transforms
from skimage.transform import resize
from torch.utils.data import Dataset, Sampler

from boundary_utils import map_, class2one_hot, one_hot2dist, id_
from boundary_utils import one_hot, depth

F = Union[Path, BinaryIO]
D = Union[Image.Image, np.ndarray, Tensor]


resizing_fn = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)


def png_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda img: img.convert('L'),
                lambda img: np.array(img)[np.newaxis, ...],
                lambda nd: nd / 255,  # max <= 1
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])


def equalized_png(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda img: img.convert('L'),
                lambda img: ImageOps.equalize(img),
                lambda img: np.array(img)[np.newaxis, ...],
                lambda nd: nd / 255,  # max <= 1
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])


def png_transform_npy(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda img: img.convert('L'),
                lambda img: np.array(img)[np.newaxis, ...],
                lambda nd: nd / 255,  # max <= 1
        ])


def npy_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda npy: np.array(npy)[np.newaxis, ...],
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])


def raw_npy_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda npy: np.array(npy),
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])


def from_numpy_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda nd: torch.tensor(nd)
        ])


def tensor_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])


def gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda img: np.array(img)[...],
                lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
                partial(class2one_hot, K=K),
                itemgetter(0)  # Then pop the element to go back to img shape
        ])


def dummy_gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda img: np.array(img)[...],
                lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
                lambda t: torch.zeros_like(t),
                partial(class2one_hot, K=K),
                itemgetter(0)  # Then pop the element to go back to img shape
        ])


def dist_map_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                gt_transform(resolution, K),
                lambda t: t.cpu().numpy(),
                partial(one_hot2dist, resolution=resolution),
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])


def unet_loss_weights_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        w_0: float = 10
        sigma: float = 5

        def closure(in_: D) -> Tensor:
                gt: Tensor = gt_transform(resolution, K)(in_)

                signed_dist_map: Tensor = dist_map_transform(resolution, K)(in_)
                dist_map: Tensor = torch.abs(signed_dist_map).type(torch.float32)

                w_c: Tensor = torch.einsum("k...->k", gt) / reduce(mul, gt.shape[1:])
                filled_w_c: Tensor = torch.einsum("k,k...->k...", w_c.type(torch.float32), torch.ones_like(dist_map))

                w: Tensor = filled_w_c + w_0 * torch.exp(- dist_map**2 / (2 * sigma**2))
                assert (K, *in_.shape) == w.shape == gt.shape, (in_.shape, w.shape, gt.shape)

                final: Tensor = torch.einsum("k...,k...->k...", gt.type(torch.float32), w)

                return final

        return closure

class SliceDataset(Dataset):
        def __init__(self, filenames: list[str], folders: list[Path], are_hots: list[bool],
                     transforms: list[Callable], debug=False, quiet=False,
                     K=4, in_memory: bool = False, spacing_dict: dict[str, Tuple[float, ...]] = None,
                     augment: Optional[Callable] = None, ignore_norm: bool = False,
                     dimensions: int = 2, debug_size: int = 10, no_assert: bool = False) -> None:
                self.folders: list[Path] = folders
                self.transforms: list[Callable[[Tuple, int], Callable[[D], Tensor]]] = transforms
                assert len(self.transforms) == len(self.folders)

                self.are_hots: list[bool] = are_hots
                self.filenames: list[str] = filenames
                self.debug = debug
                self.K: int = K  # Number of classes
                self.in_memory: bool = in_memory
                self.quiet: bool = quiet
                self.spacing_dict: Optional[dict[str, Tuple[float, ...]]] = spacing_dict
                if self.spacing_dict:
                        assert len(self.spacing_dict) == len(self.filenames)
                        print("> Spacing dictionnary loaded correctly")

                self.augment: Optional[Callable] = augment
                self.ignore_norm: bool = ignore_norm
                self.dimensions: int = dimensions

                self.no_assert: bool = no_assert

                if self.debug:
                        self.filenames = self.filenames[:debug_size]

                assert self.check_files()  # Make sure all file exists

                if not self.quiet:
                        print(f">> Initializing {self.__class__.__name__} with {len(self.filenames)} images")
                        print(f"> {self.dimensions=}")
                        if self.augment:
                                print("> Will augment data online")

                # Load things in memory if needed
                self.files: list[list[F]] = SliceDataset.load_images(self.folders, self.filenames, self.in_memory)
                assert len(self.files) == len(self.folders)
                for files in self.files:
                        assert len(files) == len(self.filenames)

        def check_files(self) -> bool:
                for folder in self.folders:
                        if not Path(folder).exists():
                                return False

                        for f_n in self.filenames:
                                if not Path(folder, f_n).exists():
                                        return False

                return True

        @staticmethod
        def load_images(folders: list[Path], filenames: list[str], in_memory: bool, quiet=False) -> list[list[F]]:
                def load(folder: Path, filename: str) -> F:
                        p: Path = Path(folder, filename)
                        if in_memory:
                                with open(p, 'rb') as data:
                                        res = io.BytesIO(data.read())
                                return res
                        return p
                if in_memory and not quiet:
                        print("> Loading the data in memory...")

                files: list[list[F]] = [[load(f, im) for im in filenames] for f in folders]

                return files

        def __len__(self):
                return len(self.filenames)

        def __getitem__(self, index: int) -> dict[str, Union[str,
                                                             int,
                                                             Tensor,
                                                             list[Tensor],
                                                             list[Tuple[slice, ...]],
                                                             list[Tuple[Tensor, Tensor]]]]:
                filename: str = self.filenames[index]
                path_name: Path = Path(filename)
                images: list[D]

                if path_name.suffix == ".png":
                        images = [Image.open(files[index]) for files in self.files]
                elif path_name.suffix == ".npy":
                        images = [np.load(files[index]) for files in self.files]
                else:
                        raise ValueError(filename)

                resolution: Tuple[float, ...]
                if self.spacing_dict:
                        resolution = self.spacing_dict[path_name.stem]
                else:
                        resolution = tuple([1] * self.dimensions)

                # Final transforms and assertions
                assert len(images) == len(self.folders) == len(self.transforms)
                t_tensors: list[Tensor] = [tr(resolution, self.K)(e) for (tr, e) in zip(self.transforms, images)]
                _, *img_shape = t_tensors[0].shape

                final_tensors: list[Tensor]
                if self.augment:
                        final_tensors = self.augment(*t_tensors)
                else:
                        final_tensors = t_tensors
                del t_tensors

                if not self.no_assert:
                        # main image is between 0 and 1
                        if not self.ignore_norm:
                                assert 0 <= final_tensors[0].min() and final_tensors[0].max() <= 1, \
                                        (final_tensors[0].min(), final_tensors[0].max())

                        for ttensor in final_tensors[1:]:  # Things should be one-hot or at least have the shape
                                assert ttensor.shape == (self.K, *img_shape), (ttensor.shape, self.K, *img_shape)

                        for ttensor, is_hot in zip(final_tensors, self.are_hots):  # All masks (ground truths) are class encoded
                                if is_hot:
                                        assert one_hot(ttensor, axis=0), torch.einsum("k...->...", ttensor)

                img, gt = final_tensors[:2]

                return {'filenames': filename,
                        'images': final_tensors[0],
                        'gt': final_tensors[1],
                        'labels': final_tensors[2:],
                        'spacings': torch.tensor(resolution),
                        'index': index}


_use_shared_memory = True


class PatientSampler(Sampler):
        def __init__(self, dataset: SliceDataset, grp_regex, shuffle=False, quiet=False) -> None:
                filenames: list[str] = dataset.filenames
                # Might be needed in case of escape sequence fuckups
                # self.grp_regex = bytes(grp_regex, "utf-8").decode('unicode_escape')
                assert grp_regex is not None
                self.grp_regex = grp_regex

                # Configure the shuffling function
                self.shuffle: bool = shuffle
                self.shuffle_fn: Callable = (lambda x: random.sample(x, len(x))) if self.shuffle else id_

                # print(f"Grouping using {self.grp_regex} regex")
                # assert grp_regex == "(patient\d+_\d+)_\d+"
                # grouping_regex: Pattern = re.compile("grp_regex")
                grouping_regex: Pattern = re.compile(self.grp_regex)

                stems: list[str] = [Path(filename).stem for filename in filenames]  # avoid matching the extension
                matches: list[Match] = map_(grouping_regex.match, stems)
                patients: list[str] = [match.group(1) for match in matches]

                unique_patients: list[str] = list(set(patients))
                assert len(unique_patients) < len(filenames)
                if not quiet:
                        print(f"Found {len(unique_patients)} unique patients out of {len(filenames)} images ; regex: {self.grp_regex}")

                self.idx_map: dict[str, list[int]] = dict(zip(unique_patients, repeat(None)))
                for i, patient in enumerate(patients):
                        if not self.idx_map[patient]:
                                self.idx_map[patient] = []

                        self.idx_map[patient] += [i]
                # print(self.idx_map)
                assert sum(len(self.idx_map[k]) for k in unique_patients) == len(filenames)

                for pid in self.idx_map.keys():
                        self.idx_map[pid] = sorted(self.idx_map[pid], key=lambda i: filenames[i])

                # print("Patient to slices mapping done")

        def __len__(self):
                return len(self.idx_map.keys())

        def __iter__(self):
                values = list(self.idx_map.values())
                shuffled = self.shuffle_fn(values)
                return iter(shuffled)