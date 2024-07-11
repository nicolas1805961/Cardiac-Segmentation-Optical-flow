from math import isnan
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from .boundary_utils import one_hot, simplex
from torch import batch_norm, einsum, Tensor
from typing import List, cast
import numpy as np
import sys
from kornia.filters import spatial_gradient3d, spatial_gradient
import matplotlib


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=9, eps=1e-3, reduction='mean'):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win
        self.reduction = reduction

    def forward(self, I, J):

        if I.dim() > 4:
            T, B, C, H, W = I.shape
            I = I.view(T * B, C, H, W)
            J = J.view(T * B, C, H, W)

        ndims = 2
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 1, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        # prepare conv kernel
        conv_fn = getattr(F, 'conv%dd' % ndims)
        # conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        # win_size = np.prod(self.win)
        win_size = torch.from_numpy(np.array([np.prod(self.win)])).float()
        win_size = win_size.cuda()
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc0 = cross * cross / (I_var * J_var + self.eps)
        cc = torch.clamp(cc0, 0.001, 0.999)

        # return negative cc.
        if self.reduction == 'mean':
            return 1-torch.mean(cc)
        else:
            cc = cc.view(T, B, C, H, W)
            return 1-cc

    


class SpatialSmoothingLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SpatialSmoothingLoss, self).__init__()
        self.epsilon = torch.Tensor([0.01]).float().to('cuda')
        self.reduction = reduction

    def forward(self, flow):
        if flow.dim() == 4:
            B, C, H, W = flow.shape
            flow = flow[None]
        elif flow.dim() == 5:
            T, B, C, H, W = flow.shape

        #gradient_list = []
        #for t in range(len(flow)):
        #    current_flow = flow[t]
        #    gradient = spatial_gradient(current_flow)
        #    assert gradient.shape[1] == 2
        #    assert gradient.shape[2] == 2
        #    gradient_list.append(gradient)
        #gradient = torch.stack(gradient_list, dim=0)

        flow = flow.permute(1, 2, 0, 3, 4).contiguous() # B, C, T, H, W
        gradient = spatial_gradient3d(flow)
        assert gradient.shape[2] == 3
        assert gradient.shape[1] == 2

        #if iter_nb > 240:
        #    matplotlib.use('qtagg')
        #    fig, ax = plt.subplots(1, 3)
        #    ax[0].imshow(gradient[0, 0, 0, 0].detach().cpu(), cmap='gray')
        #    ax[1].imshow(gradient[0, 0, 1, 0].detach().cpu(), cmap='gray')
        #    ax[2].imshow(gradient[0, 0, 2, 0].detach().cpu(), cmap='gray')
        #    plt.show()

        gradient = gradient.pow(2)

        if self.reduction == 'mean':
            gradient_xy = gradient[:, :, :2].mean()
        else:
            gradient_xy = gradient[:, :, :2].mean(1).mean(1)
            gradient_xy = gradient_xy.permute(1, 0, 2, 3).contiguous()[:, :, None, :, :]


        #huber_xy = torch.sqrt(self.epsilon + torch.sum(gradient[:, :, 0].pow(2) + gradient[:, :, 1].pow(2)))
        #huber_z = torch.sqrt(self.epsilon + torch.sum(gradient[:, :, 2].pow(2)))
        #return self.w_xy * huber_xy + self.w_z * huber_z
        
        return gradient_xy

class TemporalSmoothingLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(TemporalSmoothingLoss, self).__init__()
        self.epsilon = torch.Tensor([0.01]).float().to('cuda')
        self.reduction = reduction

    def forward(self, flow):
        if flow.dim() == 4:
            B, C, H, W = flow.shape
            flow = flow[None]
        elif flow.dim() == 5:
            T, B, C, H, W = flow.shape

        #gradient_list = []
        #for t in range(len(flow)):
        #    current_flow = flow[t]
        #    gradient = spatial_gradient(current_flow)
        #    assert gradient.shape[1] == 2
        #    assert gradient.shape[2] == 2
        #    gradient_list.append(gradient)
        #gradient = torch.stack(gradient_list, dim=0)

        flow = flow.permute(1, 2, 0, 3, 4).contiguous() # B, C, T, H, W
        gradient = spatial_gradient3d(flow)
        assert gradient.shape[2] == 3
        assert gradient.shape[1] == 2

        #if iter_nb > 240:
        #    matplotlib.use('qtagg')
        #    fig, ax = plt.subplots(1, 3)
        #    ax[0].imshow(gradient[0, 0, 0, 0].detach().cpu(), cmap='gray')
        #    ax[1].imshow(gradient[0, 0, 1, 0].detach().cpu(), cmap='gray')
        #    ax[2].imshow(gradient[0, 0, 2, 0].detach().cpu(), cmap='gray')
        #    plt.show()

        gradient = gradient.pow(2)

        if self.reduction == 'mean':
            gradient_z = gradient[:, :, 2].mean()
        else:
            gradient_z = gradient[:, :, 2].mean(1)
            gradient_z = gradient_z.permute(1, 0, 2, 3).contiguous()[:, :, None, :, :]


        #huber_xy = torch.sqrt(self.epsilon + torch.sum(gradient[:, :, 0].pow(2) + gradient[:, :, 1].pow(2)))
        #huber_z = torch.sqrt(self.epsilon + torch.sum(gradient[:, :, 2].pow(2)))
        #return self.w_xy * huber_xy + self.w_z * huber_z
        
        return gradient_z

class ContrastiveLoss(nn.Module):
    def __init__(self, temp):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, positive_sim, negative_sim):
        positive_sim = torch.flatten(positive_sim, start_dim=1)
        negative_sim = torch.flatten(negative_sim, start_dim=1)

        out = torch.exp(self.temp * negative_sim) / torch.exp(self.temp * positive_sim)
        dist_to_one = self.mse(out, torch.ones_like(out))
        k = int(0.001 * dist_to_one.shape[-1])
        _, indices = torch.topk(dist_to_one, k=k, largest=True)
        out = torch.gather(out, dim=1, index=indices)
        return out.mean()

class AverageDistanceLoss(nn.Module):
    def __init__(self):
        super(AverageDistanceLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, x, y):
        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        diff = self.mse(x, y)
        k = int(0.05 * diff.shape[-1])
        values, _ = torch.topk(diff, k=k, dim=-1, largest=True)
        return values.mean()

class MaximizeDistanceLoss(nn.Module):
    def __init__(self):
        super(MaximizeDistanceLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, x, y):
        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        diff = self.mse(x, y)
        k = int(0.05 * diff.shape[-1])
        _, indices = torch.topk(diff, k=k, dim=-1, largest=True)
        x_out = torch.gather(x, dim=1, index=indices)
        y_out = torch.gather(y, dim=1, index=indices)
        return (x_out + y_out).mean()

class RelationLoss2(nn.Module):
    def __init__(self):
        super(RelationLoss2, self).__init__()
        pass

    def rescale(self, x):
        assert x.shape[1] == 4
        my_max = torch.max(torch.flatten(x, start_dim=2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        my_min = torch.min(torch.flatten(x, start_dim=2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        x = torch.nn.functional.interpolate(x, scale_factor=(1/8), mode='bicubic', antialias=True)
        x = torch.clamp(x, my_min, my_max)
        return x

    def get_similarity_ready(self, x):
        norm = torch.linalg.norm(x, dim=0, keepdim=True)
        x = x / torch.max(norm, torch.tensor([1e-8], device=x.device))
        return x

    def get_similarity(self, x1, x2):
        C, L = x1.shape
        C, L = x2.shape

        x1 = self.get_similarity_ready(x1)
        x2 = self.get_similarity_ready(x2)

        return torch.matmul(torch.transpose(x1, dim0=0, dim1=1), x2)

    def get_mask(self, x):
        x = torch.softmax(x, dim=1)
        x = torch.argmax(x, dim=1).long()
        x = torch.nn.functional.one_hot(x, num_classes=4).permute(0, 3, 1, 2)
        x = torch.flatten(x, start_dim=2).bool()
        return x

    def forward(self, batch1, batch2):
        batch1 = self.rescale(batch1)
        batch2 = self.rescale(batch2)
        mask1 = self.get_mask(batch1)
        mask2 = self.get_mask(batch2)
        batch_res_list = []
        for i in range(len(batch1)):
            class_res_list = []
            for j in range(4):
                current_mask1 = mask1[i, j]
                current_mask2 = mask2[i, j]
                if (torch.count_nonzero(current_mask1) == 0 
                or torch.count_nonzero(current_mask2) == 0
                or torch.count_nonzero(current_mask1) == torch.numel(current_mask1)
                or torch.count_nonzero(current_mask2) == torch.numel(current_mask2)):
                    continue
                payload1 = torch.flatten(batch1[i, :], start_dim=1)
                payload2 = torch.flatten(batch2[i, :], start_dim=1)
                filtered_payload1 = payload1[:, current_mask1]
                filtered_payload2 = payload2[:, current_mask2]
                non_payload = payload2[:, ~current_mask2]
                payload_sim = self.get_similarity(filtered_payload1, filtered_payload2)
                non_payload_sim = self.get_similarity(filtered_payload1, non_payload)
                positives = (torch.mean(payload_sim, dim=1) > torch.max(non_payload_sim, dim=1)[0]).float()
                class_res_list.append(positives.mean())
            if not class_res_list:
                continue
            batch_res_list.append(torch.tensor(class_res_list).mean())
        if not batch_res_list:
            return None
        out = torch.tensor(batch_res_list).mean()
        return 1 - out


class DistanceLoss(nn.Module):
    def __init__(self, temp):
        super(DistanceLoss, self).__init__()
        self.temp = temp
    
    def get_similarity(self, p1, p2):
        B, C = p1.shape
        B, C = p2.shape
        norm1 = torch.linalg.norm(p1, dim=1, keepdim=True)
        norm2 = torch.linalg.norm(p2, dim=1, keepdim=True)

        p1 = p1 / torch.max(norm1, torch.tensor([1e-8], device=p1.device))
        p2 = p2 / torch.max(norm2, torch.tensor([1e-8], device=p1.device))

        return torch.matmul(p1, torch.transpose(p2, dim0=0, dim1=1))
    
    def forward(self, batch1, batch2):
        loss_list = []
        positive_sims = self.get_similarity(batch1, batch1)
        negative_sims = self.get_similarity(batch1, batch2)
        #positive_sims = torch.tril(positive_sims).fill_diagonal_(0)
        #positive_sims[positive_sims != 0]
        for idx, i in enumerate(range(1, len(positive_sims))):
            for j in range(idx + 1):
                positive_sim = positive_sims[i, j] / self.temp
                negative_sim = negative_sims[i] / self.temp
                numerator = torch.exp(positive_sim)
                denominator = torch.exp(negative_sim).sum()
                loss_list.append(-torch.log(numerator / denominator))
        return torch.tensor(loss_list).mean()

class SeparabilityLoss(nn.Module):
    def __init__(self):
        super(SeparabilityLoss, self).__init__()

    def forward(self, x):
        B, C, C = x.shape
        mask = torch.eye(C, dtype=bool).unsqueeze(0)
        mask = mask.repeat(B, 1, 1)
        #variances = x[mask]
        covariances = x[~mask]

        return covariances.mean()
        #return covariances.mean() - variances.mean()

def contour(x):
    '''
    Differenciable aproximation of contour extraction
    
    '''   
    min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
    max_min_pool_x = torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1)
    contour = torch.nn.functional.relu(max_min_pool_x - min_pool_x)
    return contour

class PerimeterLoss(nn.Module):
    def __init__(self, device):
        super(PerimeterLoss, self).__init__()
        self.device = device
 
    def forward(self, inputs, targets):
        B, C, H, W = targets.shape
        inputs = inputs[:, 1:, ...].type(torch.float32)
        targets = targets[:, 1:, ...].type(torch.float32)
        input_contours = contour(inputs)
        target_contours = contour(targets)
        contour_loss = (input_contours.sum(dim=(2, 3)) - target_contours.sum(dim=(2, 3)))**2
        contour_loss = contour_loss / (H*W)
        contour_loss = contour_loss.mean()
        return contour_loss

class AngleLoss(object):
    def __init__(self):
        pass
    def __call__(self, input_angle, target_angle):
        angle_distance = input_angle - target_angle
        angle_distance = (angle_distance + np.pi) % (2*np.pi) - np.pi
        angle_loss = (angle_distance ** 2).mean()
        return angle_loss

class DirectionalFieldLoss(object):
    def __init__(self, weights, writer):
        self.weights = weights
        self.writer = writer

    def __call__(self, pred, y_df, y_seg, iter_nb, do_backprop):
        assert not simplex(y_seg)

        dot = torch.einsum('bchw,bchw->bhw', pred, y_df)
        pred_norm = torch.linalg.norm(pred, dim=1, ord=2)
        y_df_norm = torch.linalg.norm(y_df, dim=1, ord=2)
        sim = dot / torch.maximum(pred_norm * y_df_norm, torch.tensor([1e-8], device=pred.device))

        #sim = torch.nn.functional.cosine_similarity(pred, y_df, dim=1, eps=1e-08)
        sim = torch.clamp(sim, -1.0 + 1e-7, 1.0 - 1e-7)
        #angle_distance_loss = torch.acos(sim) ** 2
        angle_distance_loss = torch.acos(sim) / np.pi

        l2_distance_loss = torch.linalg.norm(pred - y_df, dim=1, ord=2)

        weight_mask = torch.zeros_like(l2_distance_loss)
        for idx, weight in enumerate(self.weights):
            mask = (y_seg == idx).float() * weight
            weight_mask += mask

        #plt.imshow(weight_mask[0].cpu())
        #plt.show()
        #
        if do_backprop:
            self.writer.add_scalar('Iteration/l2_df_loss', l2_distance_loss.mean(), iter_nb)
            self.writer.add_scalar('Iteration/angle_df_loss', angle_distance_loss.mean(), iter_nb)

        assert weight_mask.shape == l2_distance_loss.shape == angle_distance_loss.shape
        #loss = (weight_mask * (l2_distance_loss + angle_distance_loss)).mean()
        loss = (weight_mask * (l2_distance_loss + angle_distance_loss))
        loss = (loss / weight_mask.sum()).sum()
        return loss


class LocalizationLoss(object):
    def __init__(self, weight, writer):
        self.mse_loss = nn.MSELoss()
        self.writer = writer
        self.weight = weight
    
    def __call__(self, inputs, metadata, iter_nb):

        computed_losses = {}

        angle_distance = inputs['angle'] - metadata['angle']
        angle_distance = (angle_distance + np.pi) % (2*np.pi) - np.pi

        loss_tx = self.mse_loss(inputs['tx'], metadata['tx'])
        loss_ty = self.mse_loss(inputs['ty'], metadata['ty'])
        loss_scale = self.mse_loss(inputs['scale'], metadata['scale'])
        loss_angle = (angle_distance ** 2).mean()

        computed_losses['tx_loss'] = loss_tx
        computed_losses['ty_loss'] = loss_ty
        computed_losses['scale_loss'] = loss_scale
        computed_losses['angle_loss'] = loss_angle
        
        self.writer.add_scalars('Iteration/Individual localization losses', computed_losses, iter_nb)
        
        out =  [x.float() for x in computed_losses.values()]
        out = self.weight * sum(out)
        return out

class Loss(object):
    def __init__(self, losses, writer, description):
        self.losses = losses
        self.writer = writer
        if any([isinstance(x['loss'], SurfaceLoss) for x in losses]):
            self.use_dist_maps = True
        else:
            self.use_dist_maps = False
        self.description = description
    
    def __str__(self):
        return self.description + ' losses'
    
    def __call__(self, inputs, targets, iter_nb):
        assert not simplex(inputs)
        assert simplex(targets)

        computed_losses = {}
        for loss in self.losses:
            l = loss['weight'] * loss['loss'](inputs, targets)
            #self.writer.add_scalar('Iteration/' + str(loss['loss']), l, iter_nb)
            computed_losses[str(loss['loss'])] = l
        
        out =  [x for x in computed_losses.values()]
        return sum(out)
    
    def ce_bootstrap(self, inputs, targets, ds_last, bootstrap):
        targets = torch.argmax(targets, dim=1)
        ce_image_loss = self.losses[0]['weight'] * self.losses[0]['loss'](inputs, targets)
        ds_last = (torch.max(ds_last, dim=1)[0]).view(ds_last.shape[0], -1)
        k = int(ds_last.shape[-1] * bootstrap)
        _, indices = torch.topk(ds_last, k=k, dim=-1, largest=False, sorted=False)
        out = torch.gather(ce_image_loss.view(ce_image_loss.shape[0], -1), dim=1, index=indices)
        return out.mean()

    def update_weight(self):
        for loss_data in self.losses:
            loss_data['weight'] = loss_data['weight'] + loss_data['add']
    
    def get_loss_weight(self):
        out = {}
        for loss in self.losses:
            out[str(loss['loss'])] = loss['weight']
        return out

class TopkLoss():
    def __init__(self, class_weights, percent):
        self.class_weights = class_weights
        self.percent = percent
    
    def __str__(self) -> str:
        return 'Topk_loss'

    def __call__(self, inputs, targets):
        targets = torch.argmax(targets, dim=1)
        N, H, W = targets.shape

        weight_tensor = torch.zeros_like(targets).float()
        for idx, weight in enumerate(self.class_weights):
            weight_tensor[targets == idx] = weight
        weight_tensor = weight_tensor.view(N, -1)

        ce_image_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none').view(N, -1)
        k = int(H * W * self.percent)
        _, indices = torch.topk(ce_image_loss, k=k, dim=-1, largest=True, sorted=False)
        ce_image_loss = torch.gather(ce_image_loss, dim=1, index=indices)
        weight_tensor = torch.gather(weight_tensor, dim=1, index=indices)

        out = (ce_image_loss / weight_tensor.sum()).sum()

        return out

class ScaleLoss():
    def __init__(self, class_weights, slope):
        self.class_weights = class_weights
        self.slope = slope

    def __str__(self) -> str:
        return 'Scale_loss'
    
    def __call__(self, inputs, targets):
        B, C, H, W = targets.shape
        targets = torch.argmax(targets, dim=1)

        binary_target = torch.flatten(torch.clone(targets), start_dim=1)
        binary_target[binary_target > 0] = 1
        r = (torch.count_nonzero(binary_target, dim=-1) / (H * W)).unsqueeze(-1)

        w = B * torch.nn.functional.softmax(-r / 0.25, dim=0)

        #w = self.slope * r + 1
        #w = B * (w / w.sum())

        weight_tensor = torch.zeros_like(targets).float()
        for idx, weight in enumerate(self.class_weights):
            weight_tensor[targets == idx] = weight

        ce_image_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none').view(B, -1)
        ce_image_loss = w * ce_image_loss
        ce_image_loss = (ce_image_loss / weight_tensor.sum()).sum()
        #ce_image_loss2 = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='mean')
        #return ce_image_loss2
        return ce_image_loss

class TopkLoss3D():
    def __init__(self, class_weights, percent):
        self.class_weights = class_weights
        self.percent = percent
    
    def __str__(self) -> str:
        return 'topk_loss_3d'

    def __call__(self, inputs, targets, dist_maps=None):
        targets = torch.argmax(targets, dim=1)
        N, D, H, W = targets.shape
        ce_image_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none').view(N, -1)
        k = int(D * H * W * self.percent)
        _, indices = torch.topk(ce_image_loss, k=k, dim=-1, largest=True, sorted=False)
        out = torch.gather(ce_image_loss, dim=1, index=indices)
        return out.mean()

class MyCrossEntropy():
    def __init__(self, class_weights, ignore_index):
        self.class_weights = class_weights
        self.ignore_index = ignore_index
    
    def __str__(self) -> str:
        return 'cross_entropy_loss'

    def __call__(self, inputs, targets):
        assert targets.shape[1] == 4
        assert not simplex(inputs)
        
        targets = torch.argmax(targets, dim=1)
        return F.cross_entropy(inputs, targets, weight=self.class_weights, ignore_index=self.ignore_index, reduction='mean')

#class LossBoundary(Loss):
#    def __init__(self, losses):
#        super().__init__(losses)
#    
#    def __call__(self, inputs, targets, dist_maps):
#        inputs = inputs[:, 1:, ...].type(torch.float32)
#        dist_maps = dist_maps[:, 1:, ...].type(torch.float32)
#
#        x1 = self.losses[0]['weight'] * self.losses[0]['loss'](inputs, targets)
#        x2 = self.losses[1]['weight'] * self.losses[1]['loss'](inputs, dist_maps)
#        return x1 + x2


class SurfaceLoss():
    def __init__(self, **kwargs):
        pass
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        #self.idc = kwargs["idc"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")
    
    def __str__(self) -> str:
        return 'boundary_loss'

    def __call__(self, probs, targets, dist_maps):
        #plt.imshow(dist_maps.cpu().detach().numpy()[0, 1, :, :])
        #plt.show()
        assert not one_hot(dist_maps)

        probs = probs[:, 1:, ...].type(torch.float32)
        dist_maps = dist_maps[:, 1:, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", probs, dist_maps)

        loss = multipled.mean()

        return loss

#def simplex(t: Tensor, axis=1) -> bool:
#    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
#    print(torch.unique(_sum))
#    _ones = torch.ones_like(_sum, dtype=torch.float32)
#    print(torch.unique(_ones))
#    print('************************************************')
#    return torch.allclose(_sum, _ones)

class GeneralizedDice():
    def __init__(self, **kwargs):
        pass
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        #self.idc: List[int] = kwargs["idc"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")
    
    def __str__(self) -> str:
        return 'generalized_dice_loss'

    def __call__(self, probs: Tensor, target: Tensor, dist_maps=None) -> Tensor:

        probs = probs[:, 1:, ...].type(torch.float32)
        target = target[:, 1:, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bkwh->bk", target).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bkwh,bkwh->bk", probs, target)
        union: Tensor = w * (einsum("bkwh->bk", probs) + einsum("bkwh->bk", target))

        divided: Tensor = 1 - 2 * (einsum("bk->b", intersection) + 1e-10) / (einsum("bk->b", union) + 1e-10)

        loss = divided.mean()

        return loss

#class MTLWrapper(nn.Module):
#    def __init__(self, model):
#        super(MTLWrapper, self).__init__()
#        self.sigma = nn.Parameter(torch.ones(2))
#        self.model = model
#
#    def __call__(self, x) -> Tensor:
#        return self.model(x)
#
#    def get_mtl_loss(self, segmentation_loss_value, reconstruction_loss_value) -> Tensor:
#        l = 0.5 * loss_values / self.sigma**2
#        l = l.sum() + torch.log(self.sigma.prod())
#        return l
#    
#    def get_loss_weights(self):
#        return 1 / (2 * self.sigma**2)

class DiceLoss():
    def __init__(self, **kwargs):
        pass
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        #self.idc: List[int] = kwargs["idc"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")
    
    def __str__(self) -> str:
        return 'dice_loss'

    def __call__(self, probs: Tensor, target: Tensor, dist_maps=None) -> Tensor:

        probs = probs[:, 1:, ...].type(torch.float32)
        target = target[:, 1:, ...].type(torch.float32)

        intersection: Tensor = einsum("bcwh,bcwh->bc", probs, target)
        union: Tensor = (einsum("bkwh->bk", probs) + einsum("bkwh->bk", target))

        divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)

        loss = divided.mean()

        return loss

class DiceLoss3D():
    def __init__(self, **kwargs):
        pass
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        #self.idc: List[int] = kwargs["idc"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")
    
    def __str__(self) -> str:
        return 'dice_loss'

    def __call__(self, probs: Tensor, target: Tensor, dist_maps=None) -> Tensor:

        probs = probs[:, 1:, ...].type(torch.float32)
        target = target[:, 1:, ...].type(torch.float32)

        intersection: Tensor = einsum("bcdwh,bcdwh->bc", probs, target)
        union: Tensor = (einsum("bkdwh->bk", probs) + einsum("bkdwh->bk", target))

        divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)

        loss = divided.mean()

        return loss

def logisticGradientPenalty(real_input, discriminator, weight):
    r"""
    Gradient penalty described in "Which training method of GANs actually
    converge
    https://arxiv.org/pdf/1801.04406.pdf
    Args:
        - input (Tensor): batch of real data
        - discrimator (nn.Module): discriminator network
        - weight (float): weight to apply to the penalty term
        - backward (bool): loss backpropagation
    """
    
    real_img = torch.autograd.Variable(real_input, requires_grad=True)
    real_logit  = discriminator(real_img)
    gradients = torch.autograd.grad(outputs=real_logit.sum(), inputs=real_img, create_graph=True)[0]

    gradients = gradients.reshape(real_img.size(0), -1)
    gradients = torch.sum(torch.mul(gradients, gradients))
    #gradients = (gradients * gradients).sum(dim=1).mean()

    gradient_penalty = gradients * weight
    return gradient_penalty

class FocalLoss():
    def __init__(self, class_weights, gamma=2):
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight=class_weights, reduction='none')

    def __str__(self) -> str:
        return 'focal_loss'

    def __call__(self, inputs, targets, dist_maps=None):
        N, C, H, W = inputs.shape
        targets = torch.argmax(targets, dim=1)
        # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, C)
        # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
        targets = targets.view(-1)
        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(inputs, dim=-1)
        ce = self.nll_loss(log_p, targets)

        # get true class column from each row
        all_rows = torch.arange(len(inputs))
        log_pt = log_p[all_rows, targets]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce
        loss = loss.mean()

        return loss






















#class DiceLoss(nn.Module):
#    def __init__(self, device, weights, smoothing=0.0):
#        super(DiceLoss, self).__init__()
#        self.device = device
#        self.smoothing = smoothing
#        self.weights = weights
# 
#    def forward(self, inputs, targets):
#        if self.smoothing > 0.0:
#            with torch.no_grad():
#                new_targets = torch.full(targets.shape, self.smoothing/(inputs.size(1)-1), device=self.device)
#                targets = new_targets.scatter_(1, torch.argmax(targets, dim=1, keepdim=True), 1.-self.smoothing, reduce='add')
#        inputs = inputs[:, 1:, :, :]
#        targets = targets[:, 1:, :, :]
#        return compute_dice(inputs, targets, self.weights)
#
#def compute_dice(inputs, targets, weights):
#    numerator = 2 * (inputs * targets).sum(dim=(0, 2, 3))
#    denominator = inputs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
#    dice_loss = (numerator / (denominator + 1e-10))
#    dice_loss = dice_loss * weights
#    dice_loss = dice_loss.mean()
#    #loss = 1 - dice_loss
#    return 1 - dice_loss


def focal_loss(inputs, targets, weights=None, gamma=2, reduction='mean'):
    if weights == None:
        _, count = torch.unique(targets, return_counts=True)
        f = count / torch.numel(targets)
        weights = 1 / f
    ce_loss = F.cross_entropy(inputs, targets, reduction=reduction, weight=weights)
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
    return focal_loss

def active_contour_loss(y_pred, y_true, weight=10):
    '''
    y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
    weight: scalar, length term weight.
    '''
    #y_true = F.one_hot(targets, num_classes=4).permute((0, 3, 1, 2))
    #y_pred = F.one_hot(y_pred, num_classes=4).permute((0, 3, 1, 2))

    # length term
    delta_r = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal gradient (B, C, H-1, W) 
    delta_c = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1] # vertical gradient   (B, C, H,   W-1)
    
    delta_r    = delta_r[:,:,1:,:-2]**2  # (B, C, H-2, W-2)
    delta_c    = delta_c[:,:,:-2,1:]**2  # (B, C, H-2, W-2)
    delta_pred = torch.abs(delta_r + delta_c) 

    epsilon = 1e-8 # where is a parameter to avoid square root is zero in practice.
    lenth = torch.mean(torch.sqrt(delta_pred + epsilon)) # eq.(11) in the paper, mean is used instead of sum.
    
    # region term
    c_in  = torch.ones_like(y_pred)
    c_out = torch.zeros_like(y_pred)

    region_in  = torch.mean(y_pred * (y_true - c_in )**2 ) # equ.(12) in the paper, mean is used instead of sum.
    region_out = torch.mean((1-y_pred) * (y_true - c_out)**2) 
    region = region_in + region_out
    
    loss =  weight * lenth + region

    return loss