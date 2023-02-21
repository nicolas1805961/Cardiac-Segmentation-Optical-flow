import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import RandomSampler, DataLoader
import cv2 as cv
from glob import glob
import numpy as np
import os
from sklearn.model_selection import train_test_split
from random import shuffle
from torchvision.transforms.functional import adjust_gamma
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random
import nibabel as nib
import torchvision.transforms.functional as TF
from scipy.ndimage import distance_transform_edt as eucl_distance
from boundary_utils import one_hot
from typing import cast
from torch import Tensor
from torch.utils.data import Sampler
import torch.nn.functional as F
import math
from skimage.measure import regionprops
import sys
import cv2
from scipy import stats
from pathlib import Path
from tqdm import tqdm
import shutil
from torchvision.transforms.functional import affine
from torchvision.transforms import InterpolationMode
import warnings

from math import cos, sin

class Q_value():
    def __init__(self, gamma, epsilon, R, number_of_outputs, rotation_step, translation_step):
        self.Tg = None
        self.gamma = gamma
        self.epsilon = epsilon
        self.R = R
        self.number_of_outputs = number_of_outputs
        self.rotation_step = rotation_step
        self.translation_step = translation_step

    def get_matrix_from_tensor(self, Tt):
        if self.number_of_outputs == 12:
            A = torch.tensor([[1, 0, 0, Tt[0]],
                            [0, cos(Tt[3]), -sin(Tt[3]), Tt[1]],
                            [0, sin(Tt[3]), cos(Tt[3]), Tt[2]],
                            [0, 0, 0, 1]])

            B = torch.tensor([[cos(Tt[4]), 0, sin(Tt[4]), 0],
                            [0, 0, 0, 0],
                            [-sin(Tt[4]), 0, cos(Tt[4]), 0],
                            [0, 0, 0, 1]])

            C = torch.tensor([[cos(Tt[5]), -sin(Tt[5]), 0, 0],
                            [sin(Tt[5]), cos(Tt[5]), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

            D = torch.matmul(A, B)
            return torch.matmul(D, C)
        
        elif self.number_of_outputs == 6:

            A = torch.tensor([[cos(Tt[0]), sin(Tt[0]), Tt[1]],
                            [-sin(Tt[0]), cos(Tt[0]), Tt[2]],
                            [0, 0, 1]])
            
            return A
            

    def get_optimal_action(self, Tt):
        at = torch.cat((torch.eye(self.number_of_outputs // 2), torch.eye(self.number_of_outputs // 2) * -1))
        at_o_Tt = at + Tt.view((1, 3))
        norms = [self.get_distance(at_o_Tt[i, :]) for i in range(self.number_of_outputs)]
        return np.argmin(np.asarray(norms))
    
    def get_distance(self, x):
        if self.number_of_outputs == 4 or self.number_of_outputs == 6:
            inversed = x * -1
            composed = self.Tg + inversed
        else:
            matrice = self.get_matrix_from_tensor(x)
            inversed = torch.inverse(matrice)
            Tg_matrix = self.get_matrix_from_tensor(self.Tg)
            composed = torch.matmul(Tg_matrix, inversed)
        composed = composed.float()
        norm = torch.norm(composed).item()
        return norm
    
    def set_Tg(self, Tg):
        self.Tg = Tg

    def take_action(self, action, Tt):
        current_Tt = Tt.clone()
        if action == 0:
            current_Tt[action] += self.rotation_step
        elif action == 3:
            current_Tt[0] -= self.rotation_step
        elif action < self.number_of_outputs // 2:
          current_Tt[action] += self.translation_step
        else:
          current_Tt[action - (self.number_of_outputs // 2)] -= self.translation_step
        return current_Tt
    
    def get_reward(self, Tt, Tt_next):
        return self.get_distance(Tt) - self.get_distance(Tt_next)

    def get_q_value(self, Tt, action):
        Tt_next = self.take_action(action, Tt)
        reward = self.get_reward(Tt, Tt_next)
        if self.get_distance(Tt_next) < self.epsilon:
            return reward + self.R
        optimal_action_next = self.get_optimal_action(Tt_next)
        return reward + self.gamma * self.get_q_value(Tt_next, optimal_action_next)
    
    def get_q(self, parameters, Tt):
        self.set_Tg(parameters)
        row = []
        for t in range(6):
            row.append(self.get_q_value(Tt, t))
        return torch.tensor(row).reshape((1, -1))

def get_heatmap(row, col, label):
    row = stats.norm.rvs(loc=row, scale=5, size=2000)
    col = stats.norm.rvs(loc=col, scale=5, size=2000)
    xy = np.vstack((row, col))
    kernel = stats.gaussian_kde(xy)
    X, Y = np.mgrid[0:label.shape[0], 0:label.shape[1]]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, (label.shape[0], label.shape[1]))
    #plt.imshow(Z)
    #plt.show()
    return Z

def set_angle(label, angle):
    (h, w) = label.shape[-2:]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    print(angle)
    rotated = cv2.warpAffine(label.astype(np.uint8), M, (w, h), flags=cv2.INTER_LINEAR)
    plt.imshow(rotated, cmap='gray')
    plt.show()

    rv_centroid, lv_centroid = get_rv_centroid(rotated)
    if rv_centroid[1] > lv_centroid[1]:
        print('flipped !')
        angle -= 180
    return angle

def get_rv_centroid(label):
    regions = regionprops(label.astype(int))
    #data = [(4 * np.pi * regions[i].area / (regions[i].perimeter ** 2), regions[i].centroid) for i in range(len(regions))]
    data = [(regions[i].eccentricity, regions[i].centroid) for i in range(len(regions))]
    data = sorted(data, key=lambda x: x[0])
    rv_centroid = data[-1][1]
    lv_centroid = data[0][1]

    #fig, ax = plt.subplots(1, 1)
    #ax.imshow(label, cmap='gray')
    #ax.scatter(x=rv_centroid[1], y=rv_centroid[0], c='r', s=40)
    #ax.scatter(x=lv_centroid[1], y=lv_centroid[0], c='g', s=40)
    #plt.show()

    return rv_centroid, lv_centroid

def get_rotation_batched_matrices(angle):
        matrices = []
        for i in range(angle.size(0)):
            m = torch.tensor([[math.cos(angle[i]), -1.0*math.sin(angle[i]), 0.], 
                            [math.sin(angle[i]), math.cos(angle[i]), 0.], 
                            [0, 0, 1]], device=angle.device).float()
            matrices.append(m)
        return torch.stack(matrices, dim=0)
    
def get_translation_batched_matrices(tx, ty):
        matrices = []
        for i in range(tx.size(0)):
            m = torch.tensor([[1, 0, tx[i]], 
                            [0, 1, ty[i]], 
                            [0, 0, 1]], device=tx.device).float()
            matrices.append(m)
        return torch.stack(matrices, dim=0)

def rotate_image(x, parameters):
    x = x[None, None].float()
    r = torch.tensor([[math.cos(parameters[0]), -1.0*math.sin(parameters[0]), 0.], [math.sin(parameters[0]), math.cos(parameters[0]), 0.], [0, 0, 1]]).float()
    t = torch.tensor([[1, 0, parameters[1]], [0, 1, parameters[2]], [0, 0, 1]]).float()
    t_2 = torch.tensor([[1, 0, -parameters[1]], [0, 1, -parameters[2]], [0, 0, 1]]).float()
    theta = torch.mm(t, torch.mm(r, t_2))[:-1].unsqueeze(0)

    grid = F.affine_grid(theta, x.size())
    rotated = F.grid_sample(x, grid, mode='nearest').squeeze()

    #fig, ax = plt.subplots(2, self.batch_size)
    #for i in range(self.batch_size):
    #    print(math.degrees(angle[i]))
    #    ax[0, i].imshow(x.cpu().numpy()[i, 0], cmap='gray')
    #    ax[1, i].imshow(rotated.cpu().numpy()[i, 0], cmap='gray')
    #plt.show()

    return rotated

def get_center(label):
    label_1 = np.copy(label)
    label_1[label_1 > 0] = 1

    center = list(regionprops(label_1.astype(int))[0].centroid)
    tx = -2 * ((label.shape[-1] / 2) - center[1]) / (label.shape[-1])
    ty = -2 * ((label.shape[-2] / 2) - center[0]) / (label.shape[-2])

    return center
    

def get_angle(label, center):

    rv_centroid, lv_centroid = get_rv_centroid(label)

    m = ((label.shape[-1] - rv_centroid[0]) - (label.shape[-1] - center[0])) / (rv_centroid[1] - center[1])

    angle = math.degrees(math.atan(m))
    if angle > 0:
        angle = math.degrees(np.pi - math.atan(m))
    else:
        angle = math.degrees(abs(math.atan(m)))
    angle = math.radians(angle)
    return angle 

def create_rotation_data(global_path, new_folder_name, big):
    patient_path_list = glob(global_path)
    sample_list = get_cases(patient_path_list)
    angle_list = []
    tx_list = []
    ty_list = []
    for j, sample in enumerate(tqdm(sample_list, position=0)):
        areas = []
        for i, file_path in enumerate(sample):
            label = np.load(file_path)['arr_1']
            label_1 = np.copy(label)
            label_1[label_1 > 0] = 1
            regions_label_1 = regionprops(label_1.astype(int))
            if len(np.unique(label)) > 3:
                areas.append((regions_label_1[0].area, i))
        index = max(areas, key=lambda x: x[0])[1]
        path = sample[index]
        label = np.load(path)['arr_1']
        if len(np.unique(label)) < 4:
            print('BUUUUUUUUUUUUUGGGGGGGGGGGGGGG')
            fig, ax = plt.subplots(1, 1)
            ax.imshow(label, cmap='gray')
            plt.show()
            sys.exit(0)
        else:
            parameters = get_parameters(label)

            angle_list.append(math.degrees(parameters[0]))
            tx_list.append(parameters[1])
            ty_list.append(parameters[2])

            #rotated = rotate_image(torch.from_numpy(label), parameters)

            #print(path)
            #print(index)
            #plt.imshow(label, cmap='gray')
            #plt.show()

            #print(parameters)
            #print(parameters)
            #q_value_calculator = Q_value(gamma=0.9, epsilon=0.5, R=10, number_of_outputs=6, rotation_step=1, translation_step=1)
            #Tt = torch.zeros(3)
            #q_values = q_value_calculator.get_q(torch.from_numpy(parameters), Tt)

            #fig, ax = plt.subplots(1, 1)
            #plt.imshow(label, cmap='gray')
            #plt.scatter(x=center[1], y=center[0], c='r', s=40)
            #plt.show()

            #label = torch.tensor(label).float()
            #rotated = affine(label[None, None], angle=0, translate=[tx, ty], scale=1, shear=0, interpolation=InterpolationMode.NEAREST)
            #rotated = affine(rotated, angle=angle, translate=[0, 0], scale=1, shear=0, interpolation=InterpolationMode.NEAREST)

            #r = torch.tensor([[np.cos(angle), -1.0*np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0, 0, 1]]).float()
            #t = torch.tensor([[1, 0, tx], [0, 1, ty], [0, 0, 1]]).float()
            #t_2 = torch.tensor([[1, 0, -tx], [0, 1, -ty], [0, 0, 1]]).float()
            #s = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()
            #theta = torch.mm(torch.mm(t, r), s)[:-1]
            #theta = torch.mm(t, torch.mm(r, t_2))[:-1]
            #theta = r[:-1]
            #theta = theta.unsqueeze(0)

            #grid = F.affine_grid(theta, (1, 1,) + label.size())
            #rotated = F.grid_sample(label[None, None], grid, mode='nearest')

            #rotated_1 = np.copy(rotated)
            #rotated_1[rotated_1 > 0] = 1
            #center2 = list(regionprops(rotated_1[0, 0].astype(int))[0].centroid)

            #fig, ax = plt.subplots(1, 1)
            #plt.imshow(rotated[0, 0], cmap='gray')
            #plt.scatter(x=center2[1], y=center2[0], c='b', s=40)
            #plt.scatter(x=center[1], y=center[0], c='r', s=40)
            #plt.show()

            #rotated_2 = torch.clone(rotated)
            #rotated_2[rotated_2 > 0] = 1
            #center = regionprops(rotated_2[0, 0].numpy().astype(int))[0].centroid
            #center = np.array([x for x in center])
            #print(center)


            #(h, w) = label.shape[-2:]
            ##(cX, cY) = (w // 2, h // 2)
            #M = cv2.getRotationMatrix2D(center[::-1], angle, 1.0)
            #rotated = cv2.warpAffine(label.astype(np.uint8), M, (w, h), flags=cv2.INTER_LINEAR)
            #fig, ax = plt.subplots(1, 1)
            #plt.imshow(rotated[0, 0], cmap='gray')
            #plt.scatter(x=center[1], y=center[0], c='r', s=40)
            #plt.show()

            for file_path in sample:
                metadata = '\\'.join(file_path.split('\\')[1:])
                folder_paths = '\\'.join(metadata.split('\\')[:-1])
                data = np.load(file_path)

                #Path(os.path.join(new_folder_name, folder_paths)).mkdir(parents=True, exist_ok=True)
                #np.savez_compressed(os.path.join(new_folder_name, metadata), arr_0=data['arr_0'], arr_1=data['arr_1'], max_area_index=index)
                
                #file_label = data['arr_1']
                #image = data['arr_0']
                #label_unique = np.unique(file_label)
                #if not big:
                #    if len(label_unique) < 3 or 1 not in label_unique:
                #        Path(os.path.join(new_folder_name, folder_paths)).mkdir(parents=True, exist_ok=True)
                #        np.savez_compressed(os.path.join(new_folder_name, metadata), arr_0=data['arr_0'], arr_1=data['arr_1'], parameters=parameters)
                #        continue
                #    else:
                #        parameters, angle, center = get_parameters(file_label)

                #        angle_list.append(math.degrees(angle))
                #        tx_list.append((file_label.shape[-1] / 2) - (center[1]))
                #        ty_list.append((file_label.shape[-2] / 2) - (center[0]))

                #        Path(os.path.join(new_folder_name, folder_paths)).mkdir(parents=True, exist_ok=True)
                #        np.savez_compressed(os.path.join(new_folder_name, metadata), arr_0=data['arr_0'], arr_1=data['arr_1'], parameters=parameters)
                #        continue
                #else:
                #    file_label = torch.from_numpy(file_label)
                #    image = torch.from_numpy(image)

                #    if len(label_unique) > 3 and 1 in label_unique:
                #        parameters, angle, center = get_parameters(file_label)

                #    for i in range(10):
                #        while True:
                #            random_angle = random.randint(-180, 180)
                #            out_angle = math.degrees(angle) + random_angle
                #            if out_angle > 0 and out_angle < 180:
                #                out_angle = math.radians(out_angle)
                #                break
                #        while True:
                #            random_tx = random.randint(-112, 112)
                #            out_tx = -2 * (((file_label.shape[-1] / 2) - (center[1] + random_tx)) / (file_label.shape[-1]))
                #            p_tx = (out_tx / (-2)) * file_label.shape[-1]
                #            if p_tx < 39 and p_tx > -39:
                #                break
                #        while True:
                #            random_ty = random.randint(-112, 112)
                #            out_ty = -2 * (((file_label.shape[-2] / 2) - (center[0] + random_ty)) / (file_label.shape[-2]))
                #            p_ty = (out_ty / (-2)) * file_label.shape[-2]
                #            if p_ty < 53 and p_ty > -9:
                #                break

                #        label_affine = affine(file_label[None, None], angle=random_angle, translate=[random_tx, random_ty], scale=1, shear=0, interpolation=InterpolationMode.NEAREST, center=center[::-1]).squeeze()
                #        image_affine = affine(image[None, None], angle=random_angle, translate=[random_tx, random_ty], scale=1, shear=0, interpolation=InterpolationMode.BILINEAR, center=center[::-1]).squeeze()
                #        #new_center = (np.array(center) + np.array([random_ty, random_tx])).tolist()

                #        new_parameters = [out_angle, out_tx, out_ty]

                #        #rotated = rotate_image(label_affine, new_parameters)

                #        #fig, ax = plt.subplots(1, 3)
                #        #ax[0].imshow(file_label, cmap='gray')
                #        #ax[1].imshow(label_affine, cmap='gray')
                #        #ax[2].imshow(rotated, cmap='gray')
                #        #plt.show()

                #        #fig, ax = plt.subplots(1, 2)
                #        #print(new_center)
                #        #print(new_parameters)
                #        #ax[0].imshow(label_affine, cmap='gray')
                #        #ax[1].imshow(image_affine, cmap='gray')
                #        #plt.show()

                #        #label_test = affine(label_affine[None, None], angle=out_angle, translate=[out_tx, out_ty], scale=1, shear=0, interpolation=InterpolationMode.NEAREST, center=new_center[::-1]).squeeze()
        
                #        #new_center = (np.array(new_center) + np.array([out_ty, out_tx])).tolist()
                        #print(new_center)
                        #fig, ax = plt.subplots(1, 1)
                        #plt.imshow(label_test, cmap='gray')
                        #plt.show()

                        #label = data['arr_1']
                        #label[label > 0] = 1
                        #regions = regionprops(label_1.astype(int))
                        #if len(regions) > 0:
                        #    center = regions[0].centroid
                        #    heatmap = get_heatmap(center[0], center[1], label)
                        #else:
                        #    heatmap = np.zeros_like(label)
                        #
                        #assert heatmap.shape == label.shape

                        #Path(os.path.join(new_folder_name, folder_paths)).mkdir(parents=True, exist_ok=True)
                        #np.savez_compressed(os.path.join(new_folder_name, metadata), arr_0=data['arr_0'], arr_1=data['arr_1'], parameters=parameters)
                        #np.savez_compressed(os.path.join(new_folder_name, metadata + f'_image{i}'), arr_0=image_affine, arr_1=label_affine, parameters=new_parameters)
    
    angle_list = torch.tensor(angle_list)
    tx_list = torch.tensor(tx_list)
    ty_list = torch.tensor(ty_list)
    
    print(f'min angle: {angle_list.min()}')
    print(f'max angle: {angle_list.max()}')
    print(f'mean angle: {angle_list.mean()}')
    print(f'std angle: {angle_list.std()}')
    print('**************************')
    print(f'min tx: {tx_list.min()}')
    print(f'max tx: {tx_list.max()}')
    print(f'mean tx: {tx_list.mean()}')
    print(f'std tx: {tx_list.std()}')
    print('**************************')
    print(f'min ty: {ty_list.min()}')
    print(f'max ty: {ty_list.max()}')
    print(f'mean ty: {ty_list.mean()}')
    print(f'std ty: {ty_list.std()}')
    print('**************************')


#new_folder_name = 'ACDC_resampled_index'
#
#dirpath = Path(new_folder_name)
#if dirpath.exists() and dirpath.is_dir():
#    shutil.rmtree(dirpath)
#
#create_rotation_data('ACDC_resampled/*', new_folder_name, big=False)



