# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 09:44:22 2022

@author: icv-leite
"""
#%% Imports
import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob

from scipy.ndimage import rotate,binary_fill_holes,zoom
from scipy.interpolate import splprep, splev
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table,find_contours

import torchvision
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

#%% Functions

def clahe_histogram_egalisation(img):
    img = np.uint8(cv2.normalize(img,None,0,255,cv2.NORM_MINMAX))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    avg, std = cv2.meanStdDev(img)
    img = (img-avg)/std
    img = np.expand_dims(img,-1).astype(np.float32)
    return img

def get_box(data,labels, info, tag_info , custome_output = False):
    contours = find_contours(data,labels)
    if np.asarray(contours).shape[0] == 1:
        info[tag_info+"_x_min"] = int(np.round(np.min(np.asarray(contours)[0,:,0])))
        info[tag_info+"_x_max"] = int(np.round(np.max(np.asarray(contours)[0,:,0])))
        info[tag_info+"_y_min"] = int(np.round(np.min(np.asarray(contours)[0,:,1])))
        info[tag_info+"_y_max"] = int(np.round(np.max(np.asarray(contours)[0,:,1])))
    else: 
        info[tag_info+"_x_min"] = int(np.round(np.min(np.asarray(contours[0])[:,0])))
        info[tag_info+"_x_max"] = int(np.round(np.max(np.asarray(contours[0])[:,0])))
        info[tag_info+"_y_min"] = int(np.round(np.min(np.asarray(contours[0])[:,1])))
        info[tag_info+"_y_max"] = int(np.round(np.max(np.asarray(contours[0])[:,1])))
    if custome_output: 
        center = [info[tag_info+"_x_min"] + (info[tag_info+"_x_max"]-info[tag_info+"_x_min"])/2,info[tag_info+"_y_min"] + (info[tag_info+"_y_max"]-info[tag_info+"_y_min"])/2]
        bx = (info[tag_info+"_x_min"], info[tag_info+"_x_max"], info[tag_info+"_x_max"], info[tag_info+"_x_min"], info[tag_info+"_x_min"])
        by = (info[tag_info+"_y_min"], info[tag_info+"_y_min"], info[tag_info+"_y_max"], info[tag_info+"_y_max"], info[tag_info+"_y_min"])
        return center, bx,by
    
def crop_box(data,info,tag_info,extend):
    if len(data.shape) == 2:
        data_return = data[info[tag_info+"_x_min"]-extend:info[tag_info+"_x_max"]+extend,
                           info[tag_info+"_y_min"]-extend:info[tag_info+"_y_max"]+extend]
    else:
        data_return = data[:,info[tag_info+"_x_min"]-extend:info[tag_info+"_x_max"]+extend,
                           info[tag_info+"_y_min"]-extend:info[tag_info+"_y_max"]+extend]
    return data_return

def crop_center(data,info,tag_info,extend): 
    x_s,y_s,x_ss,y_ss = 0,0,extend*2,extend*2
    x_min = info[tag_info+"centroid"][1]-extend
    x_max = info[tag_info+"centroid"][1]+extend
    y_min = info[tag_info+"centroid"][0]-extend
    y_max = info[tag_info+"centroid"][0]+extend
    #print("x_s=",x_s,"x_ss=",x_ss,"y_s=",y_s,"y_ss=",y_ss,"x_min=",x_min,"x_max=",x_max,"y_min=",y_min,"y_max=",y_max)
        
    if x_min<0:
        x_s = np.invert(x_min)
        x_min = 0
    if y_min < 0:
        y_s = np.invert(y_min)
        y_min = 0
    
    if len(data.shape) == 2:
        data_return = np.zeros((extend*2,extend*2))
        if x_max>data.shape[0]:
            x_max = data.shape[0]
        if y_max>data.shape[1]:
            y_max = data.shape[1]
        x_ss = x_s+(x_max-x_min)
        y_ss = y_s+(y_max-y_min)
        #print("x_s=",x_s,"x_ss=",x_ss,"y_s=",y_s,"y_ss=",y_ss,"x_min=",x_min,"x_max=",x_max,"y_min=",y_min,"y_max=",y_max)
        
        data_return[x_s:x_ss,y_s:y_ss] = data[x_min:x_max,
                           y_min:y_max]
    else:
        
        if x_max>data.shape[1]:
            x_max = data.shape[1]
        if y_max>data.shape[2]:
            y_max = data.shape[2]
        x_ss = x_s+(x_max-x_min)
        y_ss = y_s+(y_max-y_min)
        #print("x_s=",x_s,"x_ss=",x_ss,"y_s=",y_s,"y_ss=",y_ss,"x_min=",x_min,"x_max=",x_max,"y_min=",y_min,"y_max=",y_max)
        if len(data.shape) == 3:
            data_return = np.zeros((data.shape[0],extend*2,extend*2))
            data_return[:,x_s:x_ss,y_s:y_ss] = data[:,x_min:x_max,
                           y_min:y_max]
        if len(data.shape) == 4:
            data_return = np.zeros((data.shape[0],extend*2,extend*2,data.shape[3]))
            data_return[:,x_s:x_ss,y_s:y_ss,:] = data[:,x_min:x_max,
                           y_min:y_max,:]
    return data_return

def moyenne_mobile (liste, n_point):
    liste_bis = []
    for i in range (len(liste)):
        if(i==0):
            liste_bis.append(liste[i])
        else:
            moyenne = 0
            if (i >= n_point//2) and (i <= (len(liste) - n_point//2 - 1)):
                for j in range (i, i + n_point):
                    moyenne = moyenne + liste[j-(n_point//2)]
                moyenne = moyenne/n_point
                liste_bis.append(moyenne)
            else :
                    liste_bis.append(liste[i])
    return liste_bis

def snakeinterp(x=None, y=None):
    N = len(x)-1
    d = []
    remove = []
    for i in range(N):
        if(i ==N-1):
            d.append(math.hypot(x[0] - x[i], y[0] - y[i]))

        else:
            d.append(math.hypot(x[i+1] - x[i], y[i+1] - y[i]))
    return np.array(x),np.array(y)

def rotate_around_point(lists, radians, origin=(0, 0)):
    nv_liste = np.copy(lists)
    for i,xy in enumerate(lists):
        x, y = xy
        offset_x, offset_y = origin
        adjusted_x = (x - offset_x)
        adjusted_y = (y - offset_y)
        cos_rad = math.cos(radians)
        sin_rad = math.sin(radians)
        qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
        nv_liste[i] = [qx, qy]
    return nv_liste

def bspline(c ):
    x ,y = c[:,0],c[:,1]
    length = len(x)
    # f: X --> Y might not be a 1:1 correspondence
    x,y = snakeinterp(x, y)
    # get the cumulative distance along the contour
    dist = np.sqrt((x[:-1] - x[1:])**2 + (y[:-1] - y[1:])**2)
    dist_along = np.concatenate(([0], dist.cumsum()))

    # build a spline representation of the contour
    spline, u = splprep([x, y], u=dist_along, s=0)

    # resample it at smaller distance intervals
    interp_d = np.linspace(dist_along[0], dist_along[-1], length*3)
    interp_x, interp_y = splev(interp_d, spline)
    #xplot(interp_x, interp_y, '-+')



    # print(len(interp_x))
    x = moyenne_mobile(interp_x.flatten().tolist(),5)
    y = moyenne_mobile(interp_y.flatten().tolist(),5)
    
    contour = np.zeros((len(x),2))
    i = 0
    for a,b in zip(x,y):
        contour[i][0] = a
        contour[i][1] = b
        i+=1
    return contour

def check_flip(rot_mask_binary):
    mask_right = np.zeros(rot_mask_binary.shape)
    mask_left = np.zeros(rot_mask_binary.shape)
    mask_left[:,:int(rot_mask_binary.shape[0]/2)]=rot_mask_binary[:,:int(rot_mask_binary.shape[0]/2)]
    mask_right[:,int(rot_mask_binary.shape[0]/2):] = rot_mask_binary[:,int(rot_mask_binary.shape[0]/2):]
    def get_circle(pred_zero):
        image = np.uint8(cv2.normalize(pred_zero,None,0,255,cv2.NORM_MINMAX))
        cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        c = cnts[0]
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        area = cv2.contourArea(c)

        ((x, y), r) = cv2.minEnclosingCircle(c)
        contours = []
        point = [x,y-r]
        origin = [x,y]
        all_angle = np.linspace(0.0,2*np.pi,3600)
        for radians in all_angle:
            x, y = point
            offset_x, offset_y = origin
            adjusted_x = (x - offset_x)
            adjusted_y = (y - offset_y)
            cos_rad = math.cos(radians)
            sin_rad = math.sin(radians)
            qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
            qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
            contours.append([qx, qy])
        return np.asarray(contours)

    def apply_circle(sub_mask,mask):
        mask_c = np.zeros(mask.shape)
        contour = get_circle(sub_mask)
        mask_c[np.round(contour[:,1]+0.5,1).astype('int'), np.round(contour[:,0]+0.5,1).astype('int')] = 1
        # Fill in the hole created by the contour boundary
        mask_c = 2*binary_fill_holes(mask_c)

        tot = np.sum(mask_c == 2)
        mask_c[mask == 1] = 1
        count = np.sum(mask_c == 2)
        return mask_c , tot, count , contour

    mask_c_r , tot_r, count_r , contour_r = apply_circle(mask_right,rot_mask_binary)
    percentage_r = (count_r/tot_r)
    mask_c_l , tot_l, count_l , contour_l = apply_circle(mask_left,rot_mask_binary)
    percentage_l = (count_l/tot_l)


    if percentage_r < percentage_l:
        return False
    else: 
        return True
    
def zoom_volume(volume, factor):
    new_volume = np.zeros((volume.shape[0],int(volume.shape[1]*factor)+1,int(volume.shape[2]*factor)+1,volume.shape[3]))
    for j,image in enumerate(volume): 
        for i in range(volume.shape[3]):
            new_volume[j,:,:,i] = np.round(zoom(image[:,:,i],factor))
    print(new_volume.shape)
    print(np.unique(new_volume[0,:,:,0]))
    fig = plt.figure(figsize=(20,10))
    plt.figure(figsize=(20,15))
    ax0 = fig.add_subplot(151)
    ax1 = fig.add_subplot(152)
    ax2 = fig.add_subplot(153)
    ax3 = fig.add_subplot(154)
    ax4 = fig.add_subplot(155)
    ax0.imshow(new_volume[0,:,:,0])
    ax1.imshow(new_volume[0,:,:,1])
    ax2.imshow(new_volume[0,:,:,2])
    ax3.imshow(new_volume[0,:,:,3])
    ax4.imshow(new_volume[0,:,:,4])
    plt.show()
    return new_volume

#%% types of rotations

#%% Pre-processing

def process_data(global_binary_mask, all_mask, all_image, crop_size):
    
    tensors_rotations = False
    contour_rotations = True
    Zoom = False
    coef = 1.3
    flip_enable = True
    info = {} #
    #all_mask = []
    #all_image = []
    #for file_path in glob(os.path.join(patient_path, 'labeled/*')):
    #    d1 =np.load(file_path)
    #    all_mask.append(d1['arr_1'])
    #    all_image.append(d1['arr_0'])
    #all_mask=np.asarray(all_mask)
    #all_image=np.asarray(all_image)
    info['y_test'] = np.copy(all_mask)
    info['x_test'] = np.copy(all_image)
    
    #global_binary_mask=np.where(sum(all_mask)==0,sum(all_mask),1)
    binary_all_mask = np.full((all_mask.shape[0],all_mask.shape[1],all_mask.shape[2], len(np.unique(all_mask))), fill_value = 0)
    j = 0
    for i in np.unique(all_mask):
        if i == 0:
            binary_all_mask[all_mask != i, j] = 1
        else:
            binary_all_mask[all_mask == i, j] = 1
        j+=1
    if len(all_image.shape) == 3:
        all_image = np.expand_dims(all_image,3)
        
    #%% intermediare plt
    #print('not rotated !')
    #idx_image = 2
    #fig = plt.figure(figsize=(20,10))
    #ax0 = fig.add_subplot(231)
    #ax1 = fig.add_subplot(232)
    #ax2 = fig.add_subplot(233)
    #ax3 = fig.add_subplot(234)
    #ax4 = fig.add_subplot(235)
    #ax5 = fig.add_subplot(236)
    #ax0.imshow(global_binary_mask)
    #ax1.imshow(np.invert(binary_all_mask[idx_image,:,:,0]))
    #ax2.imshow(binary_all_mask[idx_image,:,:,1])
    #ax3.imshow(binary_all_mask[idx_image,:,:,2])
    #ax4.imshow(binary_all_mask[idx_image,:,:,3])
    #ax5.imshow(all_image[idx_image,:,:,0])
    #plt.show()

    #%% Get information from globale_binary mask
    label_mask = label(global_binary_mask)
    props = regionprops(label_mask)[0]
    info["global_centroid"] = [ int(np.round(props.centroid[1])),int(np.round(props.centroid[0]))]
    info["global_orientation"] = props.orientation
    get_box(global_binary_mask,0,info,"global_box")
    #%% first center crop box 
    box_size = crop_size
    crop_all_mask = crop_center(binary_all_mask,info,"global_",box_size)
    crop_all_image = crop_center(all_image,info,"global_",box_size)
    crop_binary_mask = crop_center(global_binary_mask,info,"global_",box_size)
    mask1 = crop_binary_mask
    #check if four_cavities
    if Zoom : 
        crop_binary_mask = zoom(crop_binary_mask,coef)
        crop_all_mask = zoom_volume(crop_all_mask,coef)
        crop_all_image = zoom(crop_all_image,coef)

    #%% Calculate angle

    angle = info["global_orientation"]
    if (np.degrees(angle) < 70 or np.degrees(angle) > 110) and np.degrees(angle) >0:
        #rot_path.append(path)
        
        rot = False
        angled = 90-np.degrees(angle)
        angle = np.radians(-90)+angle
        print("positive : ",angled)            

    elif(np.degrees(angle) > -70 or np.degrees(angle) < -110) and np.degrees(angle) <0:
        #rot_path.append(path)
        rot = False
        angled = -90-np.degrees(angle)
        angle = np.radians(+90)+angle
        print("negative : ",angled)


    if tensors_rotations:
        tensor_binary_mask = np.zeros((1,1,crop_binary_mask.shape[0],crop_binary_mask.shape[1]))
        tensor_all_mask = np.zeros((crop_all_mask.shape[0],crop_all_mask.shape[3],crop_all_mask.shape[1],crop_all_mask.shape[2]))
        tensor_all_image = np.zeros((crop_all_image.shape[0],1,crop_all_image.shape[1],crop_all_image.shape[2]))

        tensor_binary_mask[0,0,:,:]=crop_binary_mask
        for i in range(crop_all_mask.shape[3]):

            tensor_all_mask[:,i,:,:]=crop_all_mask[:,:,:,i]
        tensor_all_image[:,0,:,:]=crop_all_image[:,:,:,0]

        tensor_binary_mask = TF.rotate(torch.from_numpy(tensor_binary_mask), angled,interpolation = InterpolationMode.BILINEAR ).numpy()
        tensor_all_mask = TF.rotate(torch.from_numpy(tensor_all_mask), angled,interpolation = InterpolationMode.BILINEAR ).numpy()
        tensor_all_image = TF.rotate(torch.from_numpy(tensor_all_image), angled,interpolation = InterpolationMode.BILINEAR ).numpy()
        
        rot_all_crop_mask = np.copy(crop_all_mask)
        rot_all_crop_image = np.copy(crop_all_image)
        rot_mask_binary = np.round(tensor_binary_mask[0,0,:,:])
        for i in range(crop_all_mask.shape[3]):
            rot_all_crop_mask[:,:,:,i] = np.where(tensor_all_mask[:,i,:,:]<=0,tensor_all_mask[:,i,:,:],1)
        rot_all_crop_image[:,:,:,0] = tensor_all_image[:,0,:,:]
        rot_all_crop_mask[rot_all_crop_mask[:,:,:,0]==0,0] =2
        rot_all_crop_mask[rot_all_crop_mask[:,:,:,0]==1,0] =0
        rot_all_crop_mask[rot_all_crop_mask[:,:,:,0]==0,0] =1
        info["rot_all_crop_image"] = rot_all_crop_image
        info["rot_all_crop_mask"] = rot_all_crop_mask
        
        if flip_enable:
            if check_flip(rot_mask_binary):
                rot_mask_binary = np.flip(np.copy(rot_mask_binary),axis=1)
                for i in range(len(rot_all_crop_mask)):
                    rot_all_crop_mask[i] = np.flip(rot_all_crop_mask[i],axis=1)
                    rot_all_crop_image[i] = np.flip(rot_all_crop_image[i],axis=1)
                info["flip_r"] = True
            else: 
                rot_mask_binary = np.copy(rot_mask_binary)
        #np.savez_compressed(path_s,**info)
        #print('rotated !')
        #fig = plt.figure(figsize=(20,10))
        #ax0 = fig.add_subplot(231)
        #ax1 = fig.add_subplot(232)
        #ax2 = fig.add_subplot(233)
        #ax3 = fig.add_subplot(234)
        #ax4 = fig.add_subplot(235)
        #ax5 = fig.add_subplot(236)
        #ax0.imshow(rot_mask_binary)
        #ax1.imshow(rot_all_crop_mask[idx_image,:,:,0])
        #ax2.imshow(rot_all_crop_mask[idx_image,:,:,1])
        #ax3.imshow(rot_all_crop_mask[idx_image,:,:,2])
        #ax4.imshow(rot_all_crop_mask[idx_image,:,:,3])
        #ax5.imshow(rot_all_crop_image[idx_image,:,:,0])
        #plt.show()
        
        rot_all_crop_mask = np.transpose(rot_all_crop_mask, (0, 3, 1, 2))
        rot_all_crop_mask = torch.from_numpy(rot_all_crop_mask)
        rot_all_crop_image = np.transpose(rot_all_crop_image, (0, 3, 1, 2))
        rot_all_crop_image = torch.from_numpy(rot_all_crop_image)
        rot_all_crop_image = rot_all_crop_image.squeeze(1)
        rot_all_crop_mask = torch.argmax(rot_all_crop_mask, dim=1)
        return rot_all_crop_image, rot_all_crop_mask
        
    if contour_rotations:
        rot_all_crop_mask_v2 = np.copy(crop_all_mask)
        rot_all_crop_image_v2 = np.copy(crop_all_image)
        
        center = [int(mask1.shape[0]/2),int(mask1.shape[1]/2)]   
        contour_binary = find_contours(crop_binary_mask)[0]
        rot_mask_binary_v2  = np.zeros(mask1.shape)
        # Create a contour image by using the contour coordinates rounded to their nearest integer value
        contour_binary=rotate_around_point(bspline( contour_binary) ,angle , center)
        rot_mask_binary_v2[np.round(contour_binary[:,0]+0.5,1).astype('int'), np.round(contour_binary[:,1]+0.5,1).astype('int')] = 1
        # Fill in the hole created by the contour boundary
        rot_mask_binary_v2 =binary_fill_holes(rot_mask_binary_v2)
        def max_contour(contour):
            max_len= 0
            max_i= 0
            if len(contour) >1:
                
                for i,c in enumerate(contour):
                    if len(c) > max_len: 
                        max_i = i
                        max_len = len(c)
            return contour[max_i]
        for i ,element in enumerate(zip(crop_all_image,crop_all_mask)):
            #print(i)
            image1, mask = element
            # get contours
            nv_contour = []
            for z in range(mask.shape[-1]):
                if len(np.unique(mask[:,:,z])) != 1:
                    nv_contour.append(max_contour(find_contours(mask[:,:,z])))
                else:
                    nv_contour.append(np.zeros((800,2)))
            nv_mask =[]

            for contour in nv_contour:
                r_mask = np.zeros(mask1.shape)
                # Create a contour image by using the contour coordinates rounded to their nearest integer value
                if len(np.unique(contour))!= 1:
                    contour=rotate_around_point(bspline( contour) , angle, center)
                    r_mask[np.round(contour[:,0],1).astype('int'), np.round(contour[:,1],1).astype('int')] = 1
                    # Fill in the hole created by the contour boundary
                    r_mask =binary_fill_holes(r_mask)
                    nv_mask.append(r_mask)
                else : 
                    nv_mask.append(np.zeros((mask.shape[0],mask.shape[1])))
            image1 =rotate(image1, -np.degrees(angle),reshape = False)
            #nv_mask_finale  = np.zeros(mask.shape)

            rot_all_crop_mask_v2[i,:,:,2] = nv_mask[2]*1 - nv_mask[3]*1
            rot_all_crop_mask_v2[i,:,:,3] = nv_mask[3]*1
            rot_all_crop_mask_v2[i,:,:,1] = nv_mask[1]*1
            if len(np.unique(nv_mask[0]))!= 1:
                rot_all_crop_mask_v2[i,:,:,0] = np.invert(nv_mask[0])*1
            else : 
                rot_all_crop_mask_v2[i,:,:,0] = np.ones(nv_mask[0].shape)

            #rot_all_crop_mask_v2[i,rot_all_crop_mask[i,:,:,2] == 1,0] = 0

            rot_all_crop_image_v2[i,:,:] = image1
            
        if flip_enable:
            if check_flip(rot_mask_binary_v2):
                print('***************************')
                rot_mask_binary_v2 = np.flip(np.copy(rot_mask_binary_v2),axis=1)
                for i in range(len(rot_all_crop_mask_v2)):
                    rot_all_crop_mask_v2[i] = np.flip(rot_all_crop_mask_v2[i],axis=1)
                    rot_all_crop_image_v2[i] = np.flip(rot_all_crop_image_v2[i],axis=1)
                info["flip_r"] = True
            else: 
                rot_mask_binary_v2 = np.copy(rot_mask_binary_v2)
        #fig = plt.figure(figsize=(20,10))
        
        #print('rotated !')
        #
        #ax0 = fig.add_subplot(231)
        #ax1 = fig.add_subplot(232)
        #ax2 = fig.add_subplot(233)
        #ax3 = fig.add_subplot(234)
        #ax4 = fig.add_subplot(235)
        #ax5 = fig.add_subplot(236)
        #ax0.imshow(rot_mask_binary_v2)
        #ax1.imshow(rot_all_crop_mask_v2[idx_image,:,:,0])
        #ax2.imshow(rot_all_crop_mask_v2[idx_image,:,:,1])
        #ax3.imshow(rot_all_crop_mask_v2[idx_image,:,:,2])
        #ax4.imshow(rot_all_crop_mask_v2[idx_image,:,:,3])
        #ax5.imshow(rot_all_crop_image_v2[idx_image,:,:,0])
        #contour_1 = find_contours(rot_all_crop_mask_v2[idx_image,:,:,1])[0]
        #contour_2 = find_contours(rot_all_crop_mask_v2[idx_image,:,:,2])[0]
        #contour_3 = find_contours(rot_all_crop_mask_v2[idx_image,:,:,3])[0]
        #contour_0 = find_contours(rot_all_crop_mask_v2[idx_image,:,:,0])[0]
        #ax5.plot(contour_0[:,1],contour_0[:,0])
        #ax5.plot(contour_1[:,1],contour_1[:,0])
        #ax5.plot(contour_2[:,1],contour_2[:,0])
        #ax5.plot(contour_3[:,1],contour_3[:,0])
        #plt.show()

        rot_all_crop_mask_v2 = np.transpose(rot_all_crop_mask_v2, (0, 3, 1, 2))
        rot_all_crop_mask_v2 = torch.from_numpy(rot_all_crop_mask_v2)
        rot_all_crop_image_v2 = np.transpose(rot_all_crop_image_v2, (0, 3, 1, 2))
        rot_all_crop_image_v2 = torch.from_numpy(rot_all_crop_image_v2)
        rot_all_crop_image_v2 = rot_all_crop_image_v2.squeeze(1)
        rot_all_crop_mask_v2 = torch.argmax(rot_all_crop_mask_v2, dim=1)
        return rot_all_crop_image_v2, rot_all_crop_mask_v2


#
#
#for patient_path in glob('ACDC_data/*'):
#    print(patient_path)
#    process_data(patient_path)