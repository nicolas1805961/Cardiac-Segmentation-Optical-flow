import albumentations as A
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from monai.transforms import NormalizeIntensity
import cv2 as cv

class Augmenter(object):
    """
    Use this for debugging custom transforms. It does not use a background thread and you can therefore easily debug
    into your augmentations. This should not be used for training. If you want a generator that uses (a) background
    process(es), use MultiThreadedAugmenter.
    Args:
        data_loader (generator or DataLoaderBase instance): Your data loader. Must have a .next() function and return
        a dict that complies with our data structure

        transform (Transform instance): Any of our transformations. If you want to use multiple transformations then
        use our Compose transform! Can be None (in that case no transform will be applied)
    """
    def __init__(self, data_loader, isval, pretraining, deep_supervision, image_size):
        self.data_loader = data_loader
        self.pretraining = pretraining
        self.deep_supervision = deep_supervision
        self.image_size = image_size
        if isval:
            self.transform = None
        else:
            self.transform = self.set_up_augmentation_pipeline()
    
    def normalize(self, x):
        my_min = x.min()
        my_max = x.max()
        x = (x - my_min) / (my_max - my_min)
        return x, my_min, my_max
    
    def unnormalize(self, x, my_min, my_max):
        x = (x * (my_max - my_min)) + my_min
        return x

    def augment_flow(self, data_dict):
        labeled = data_dict['labeled']
        target = data_dict['seg']
        unlabeled = data_dict['unlabeled']

        batch_labeled_list = []
        batch_target_list = []
        batch_unlabeled_list = []
        for b in range(len(labeled)):
            current_labeled = labeled[b, 0]
            current_target = target[b, 0]

            temp = np.copy(current_labeled)

            transformed = self.transform(image=current_labeled, mask=current_target)
            current_labeled = transformed['image']
            current_target = transformed['mask']

            matplotlib.use('QtAgg')
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(temp, cmap='gray')
            ax[1].imshow(current_labeled, cmap='gray')
            ax[2].imshow(current_target, cmap='gray')
            plt.show()

            unlabeled_list = []
            for t in range(len(unlabeled)):
                current_unlabeled = unlabeled[t, b, 0]
                replayed = A.ReplayCompose.replay(transformed['replay'], image=current_unlabeled)
                current_unlabeled = replayed['image']
                unlabeled_list.append(current_unlabeled)
            unlabeled_temporal = np.stack(unlabeled_list, axis=0) # T, H, W

            mu = unlabeled_temporal.mean()
            sigma = unlabeled_temporal.std()
            unlabeled_temporal = NormalizeIntensity(subtrahend=mu, divisor=sigma, )(unlabeled_temporal)
            current_labeled = NormalizeIntensity(subtrahend=mu, divisor=sigma, )(current_labeled)

            batch_labeled_list.append(current_labeled)
            batch_target_list.append(current_target)
            batch_unlabeled_list.append(unlabeled_temporal)
        
        labeled = np.stack(batch_labeled_list, axis=0)[:, None]
        target = np.stack(batch_target_list, axis=0)[:, None]
        unlabeled = np.stack(batch_unlabeled_list, axis=1)[:, :, None]

        matplotlib.use('QtAgg')
        fig, ax = plt.subplots(2, 4)
        for i in range(len(labeled)):
            for j in range(len(unlabeled)):
                ax[i, j].imshow(unlabeled[j, i, 0], cmap='gray')
            ax[i, 3].imshow(labeled[i, 0], cmap='gray')
        plt.show()

        return {'labeled': labeled, 'target': target, 'unlabeled': unlabeled}


    def augment_binary(self, data_dict):
        labeled = data_dict['labeled']
        target = data_dict['seg']

        nb_scales = 3 if self.deep_supervision else 1

        batch_labeled_list = []
        batch_target_list = []
        for b in range(len(labeled)):
            current_labeled = labeled[b, 0]
            current_target = target[b, 0]

            temp = np.copy(current_labeled)

            transformed = self.transform(image=current_labeled, mask=current_target)
            current_labeled = transformed['image']
            current_target = transformed['mask']

            matplotlib.use('QtAgg')
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(temp, cmap='gray')
            ax[1].imshow(current_labeled, cmap='gray')
            ax[2].imshow(current_target, cmap='gray')
            plt.show()

            mu = current_labeled.mean()
            sigma = current_labeled.std()
            current_labeled = NormalizeIntensity(subtrahend=mu, divisor=sigma, )(current_labeled)

            resized_labeled_list = []
            resized_target_list = []
            for i in range(nb_scales):
                resized_labeled = cv.resize(current_labeled, (self.image_size // 2**i, self.image_size // 2**i), interpolation=cv.INTER_AREA)
                resized_target = cv.resize(current_target, (self.image_size // 2**i, self.image_size // 2**i), interpolation=cv.INTER_NEAREST)
                resized_labeled_list.append(resized_labeled)
                resized_target_list.append(resized_target)

            batch_labeled_list.append(resized_labeled_list)
            batch_target_list.append(resized_target_list)

        out_labeled = []
        out_target = []
        for j in range(nb_scales):
            in_labeled = []
            in_target = []
            for x in batch_labeled_list:
                in_labeled.append(x[j])
            out_labeled.append(np.stack(in_labeled, axis=0)[:, None])
            for x in batch_target_list:
                in_target.append(x[j])
            out_target.append(np.stack(in_labeled, axis=0)[:, None])

        print(len(out_labeled))
        print(out_labeled[0].shape)
        print(out_labeled[1].shape)
        print(out_labeled[2].shape)

        return {'data': out_labeled, 'target': out_target}
    
    
    def set_up_augmentation_pipeline(self):
        if self.pretraining:
            transform = A.Compose(
                                    [
                                        #A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=0, border_mode=0, value=0, p=0.2),
                                        #A.ShiftScaleRotate(shift_limit=0, scale_limit=0.5, rotate_limit=0, border_mode=0, value=0, p=0.2),
                                        #A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=(-100, -100), border_mode=0, value=0, p=0.2),
                                        #A.GaussNoise(var_limit=(0.0001, 0.0005), p=0.2),
                                        #A.RandomGamma(gamma_limit=(70, 150), p=0.2),
                                        #A.Flip(p=0.5),
                                        #A.GaussianBlur(blur_limit=1, p=0.2),
                                        #A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.2),
                                        #A.Downscale(scale_min=0.50, scale_max=0.75, p=0.2),
                                        #A.ImageCompression(quality_lower=50, quality_upper=75, p=0.2),
                                        #A.Sharpen(alpha=(0.01, 0.05), p=0.2)
                                    ]
                                )
        else:
            transform = A.ReplayCompose([])

        return transform

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.data_loader)
        if self.transform is not None:
            if self.pretraining:
                item = self.augment_binary(item)
            else:
                item = self.augment_flow(item)
        return item

    def next(self):
        return self.__next__()