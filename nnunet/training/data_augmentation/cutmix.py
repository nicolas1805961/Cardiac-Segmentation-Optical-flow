import numpy as np

def batched_rand_bbox(size):
    B = size[0]
    W = size[2]
    H = size[3]

    lam = np.random.beta(1, 1, size=(B,))

    cut_rat = np.sqrt(1. - lam)
    cut_w = (W * cut_rat).astype(np.int32)
    cut_h = (H * cut_rat).astype(np.int32)

    # uniform
    cx = np.random.randint(W, size=(B,))
    cy = np.random.randint(H, size=(B,))

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(img1, img2, mask_coords):
    if isinstance(img1, list):
        for j in range(len(img1)):
            current_mask_coords = [x//(2**j) for x in mask_coords]
            for i in range(len(mask_coords[0])):
                img1[j][i, :, current_mask_coords[0][i]:current_mask_coords[2][i], current_mask_coords[1][i]:current_mask_coords[3][i]] = img2[j][i, :, current_mask_coords[0][i]:current_mask_coords[2][i], current_mask_coords[1][i]:current_mask_coords[3][i]]
    else:
        for i in range(len(mask_coords[0])):
            img1[i, :, mask_coords[0][i]:mask_coords[2][i], mask_coords[1][i]:mask_coords[3][i]] = img2[i, :, mask_coords[0][i]:mask_coords[2][i], mask_coords[1][i]:mask_coords[3][i]]
    return img1