import warnings
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.nn.functional import pad as pad_pt

from monai.config import USE_COMPILED, DtypeLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.utils import compute_shape_offset, to_affine_nd, zoom_affine
from monai.networks.layers import AffineTransform, GaussianFilter, grid_pull
from monai.transforms.croppad.array import CenterSpatialCrop
from monai.transforms.transform import Randomizable, RandomizableTransform, ThreadUnsafe, Transform
from monai.transforms.utils import (
    convert_pad_mode,
    create_control_grid,
    create_grid,
    create_rotate,
    create_scale,
    create_shear,
    create_translate,
    map_spatial_axes,
)
from monai.transforms.utils_pytorch_numpy_unification import concatenate
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    PytorchPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    issequenceiterable,
    optional_import,
)
from monai.utils.deprecate_utils import deprecated_arg
from monai.utils.enums import TransformBackends
from monai.utils.module import look_up_option
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type


class Pad(Transform):
    """
    Perform padding for a given an amount of padding in each dimension.
    If input is `torch.Tensor`, `torch.nn.functional.pad` will be used, otherwise, `np.pad` will be used.

    Args:
        to_pad: the amount to be padded in each dimension [(low_H, high_H), (low_W, high_W), ...].
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        to_pad: List[Tuple[int, int]],
        mode: Union[NumpyPadMode, PytorchPadMode, str] = NumpyPadMode.CONSTANT,
        **kwargs,
    ) -> None:
        self.to_pad = to_pad
        self.mode = mode
        self.kwargs = kwargs

    @staticmethod
    def _np_pad(img: np.ndarray, all_pad_width, mode, **kwargs) -> np.ndarray:
        return np.pad(img, all_pad_width, mode=mode, **kwargs)  # type: ignore

    @staticmethod
    def _pt_pad(img: torch.Tensor, all_pad_width, mode, **kwargs) -> torch.Tensor:
        pt_pad_width = [val for sublist in all_pad_width[1:] for val in sublist[::-1]][::-1]
        # torch.pad expects `[B, C, H, W, [D]]` shape
        return pad_pt(img.unsqueeze(0), pt_pad_width, mode=mode, **kwargs).squeeze(0)

    def __call__(
        self, img: NdarrayOrTensor, mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = None
    ) -> NdarrayOrTensor:
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first and
                padding doesn't apply to the channel dim.
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"`` or ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to `self.mode`.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

        """
        if not np.asarray(self.to_pad).any():
            # all zeros, skip padding
            return img
        mode = convert_pad_mode(dst=img, mode=mode or self.mode).value
        pad = self._pt_pad if isinstance(img, torch.Tensor) else self._np_pad
        return pad(img, self.to_pad, mode, **self.kwargs)  # type: ignore

class Zoom(Transform):
    """
    Zooms an ND image using :py:class:`torch.nn.functional.interpolate`.
    For details, please see https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html.

    Different from :py:class:`monai.transforms.resize`, this transform takes scaling factors
    as input, and provides an option of preserving the input spatial size.

    Args:
        zoom: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        keep_size: Should keep original size (padding/slicing if needed), default is True.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        zoom: Union[Sequence[float], float],
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        padding_mode: Union[NumpyPadMode, PytorchPadMode, str] = NumpyPadMode.EDGE,
        align_corners: Optional[bool] = None,
        keep_size: bool = True,
        pad_value=0.0,
        **kwargs,
    ) -> None:
        self.zoom = zoom
        self.mode: InterpolateMode = InterpolateMode(mode)
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.keep_size = keep_size
        self.pad_value=pad_value
        self.kwargs = kwargs

    def __call__(
        self,
        img: NdarrayOrTensor,
        mode: Optional[Union[InterpolateMode, str]] = None,
        padding_mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = None,
        align_corners: Optional[bool] = None,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
            mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                The mode to pad data after zooming.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html

        """
        img_t: torch.Tensor
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float32)  # type: ignore

        _zoom = ensure_tuple_rep(self.zoom, img.ndim - 1)  # match the spatial image dim
        zoomed: NdarrayOrTensor = torch.nn.functional.interpolate(  # type: ignore
            recompute_scale_factor=True,
            input=img_t.unsqueeze(0),
            scale_factor=list(_zoom),
            mode=look_up_option(self.mode if mode is None else mode, InterpolateMode).value,
            align_corners=self.align_corners if align_corners is None else align_corners,
        )
        zoomed = zoomed.squeeze(0)

        if self.keep_size and not np.allclose(img_t.shape, zoomed.shape):

            pad_vec = [(0.0, 0.0)] * len(img_t.shape)
            slice_vec = [slice(None)] * len(img_t.shape)
            for idx, (od, zd) in enumerate(zip(img_t.shape, zoomed.shape)):
                diff = od - zd
                half = abs(diff) // 2
                if diff > 0:  # need padding
                    pad_vec[idx] = (half, diff - half)
                elif diff < 0:  # need slicing
                    slice_vec[idx] = slice(half, half + od)

            padder = Pad(pad_vec, padding_mode or self.padding_mode, value=self.pad_value)
            zoomed = padder(zoomed)
            zoomed = zoomed[tuple(slice_vec)]

        out, *_ = convert_to_dst_type(zoomed, dst=img)
        return out