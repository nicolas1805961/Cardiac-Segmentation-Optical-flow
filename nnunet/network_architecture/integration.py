import torch.nn as nn
import torch
from torch.nn.functional import grid_sample

class SpatialTransformerContour(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        self.shape = size

    def forward(self, new_locs, original, mode='bilinear'):
        '''new_locs: B, 2, H, W
        original: B, C, H, W'''
        # new locations

        # need to normalize grid values to [-1, 1] for resampler
        #print(new_locs.mean())
        for i in range(2):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (self.shape[~i] - 1) - 0.5)
            #new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (self.shape[i] - 1) - 0.5)

        #print(new_locs.mean())
        #print('********************************')
        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        new_locs = new_locs.permute(0, 2, 3, 1)
        #new_locs = new_locs[..., [1, 0]]

        return grid_sample(original, new_locs, align_corners=True, mode=self.mode)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, flow, original, mode='bilinear'):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return grid_sample(original, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec