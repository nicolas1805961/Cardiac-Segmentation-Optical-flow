# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:53:00 2023

@author: icv-leite
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as ax
import math
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import binary_fill_holes, center_of_mass
from skimage.measure import label, regionprops, find_contours
from scipy.spatial import distance_matrix
import mat73
from dataclasses import dataclass
import nibabel as nib
from glob import glob
import matplotlib
import os
from interpolation import interpolate_points

def smooth_line(cnt, window_size):
    """
    Smooth a line using a moving average filter.

    Parameters:
    - x: The x-coordinates of the line.
    - y: The y-coordinates of the line.
    - window_size: The size of the moving average window.

    Returns:
    - smoothed_y: The smoothed y-coordinates of the line.
    """
    cnt_size = len(cnt)
    x = np.concatenate((cnt[cnt_size-window_size:,0],cnt[:,0],cnt[:window_size,0]))
    y = np.concatenate((cnt[cnt_size-window_size:,1],cnt[:,1],cnt[:window_size,1]))
    # Create a window of ones with size equal to window_size
    window = np.ones(window_size) / window_size

    # Convolve the line with the window to apply the moving average filter
    x = np.convolve(x, window, mode='same')
    y = np.convolve(y, window, mode='same')

    cnt[:,0]=x[window_size:len(x)-window_size]
    cnt[:,1]=y[window_size:len(y)-window_size]
    return cnt

def extract_class_from_mask(pred,classe):
    mask = np.zeros(pred.shape)

    if  isinstance(classe,list):
        for cl in classe:
            mask = mask + np.where(pred == cl, 1, 0)
    else :
        mask = np.where(pred == classe, 1, 0)

    return mask

def check_contour(cnt):
    """
    Given a list of contours, returns the contour with the largest number of points.
    If there is only one contour in the list, that contour is returned.

    Parameters:
        cnt: A list of numpy arrays representing the contours.

    Returns:
        numpy array: The contour with the largest number of points.
    """
    if len(cnt) > 1:
        idx_out = np.argmax([len(elem) for elem in cnt])
        return np.asarray(cnt[idx_out])
    else:
        return np.asarray(cnt[0])

def get_contour( pred , classe, window_size = 5):
    """
    Get the contour of a multi class mask.

    Parameters
    ----------


    Returns
    -------
    cnt : ndarray
        The contour coordinates.
    img : ndarray
        The filled binary image.
    """

    mask = extract_class_from_mask(pred,classe)


    # Fill any holes in the image
    mask = binary_fill_holes(mask).astype("int8")

    # Find the contours in the image
    cnt = find_contours(mask)

    # If there are multiple contours, keep only the largest connected component
    if len(cnt) > 1:
        labeled = label(mask)
        props = regionprops(labeled)
        max_area = 0
        for prop in props:
            if prop.area > max_area:
                max_area = prop.area
                mask = labeled == prop.label
        cnt = find_contours(mask)
        cnt = check_contour(cnt)
    else:
        cnt = cnt[0]

    cnt = smooth_line(cnt, window_size)



    return mask, cnt

def extract_adj_contour2(contours, mask, good_classes, bad_classes=None, alls=False, precision=2):
    """
    Extracts adjacent contours based on the classes in the mask.

    Parameters:
    - contours: The input contours.
    - mask: The input mask.
    - good_classes: List of classes considered "good".
    - bad_classes: List of classes considered "bad".
    - alls: Boolean flag indicating whether all classes should be present in the box (default: False).
    - precision: The size of the box around the contour point (default: 2).

    Returns:
    The adjacent contour.
    """
    adjacent_contour = []
    for point in contours:
        x, y = np.round(point).astype(int)
        box = mask[x - precision: x + precision, y - precision: y + precision]
        if alls:
            if all(class_val in box for class_val in good_classes):
                if bad_classes is not None:
                    if all(class_val not in box for class_val in bad_classes):
                        adjacent_contour.append(point)
                else:
                    adjacent_contour.append(point)
        else:
            if all(class_val in box for class_val in good_classes):
                if bad_classes is not None:
                    if all(class_val not in box for class_val in bad_classes):
                        adjacent_contour.append(point)
                else:
                    adjacent_contour.append(point)
    return np.asarray(adjacent_contour)

def pol2cart(theta,rho):

    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def line_segment_intersection(p1, p2, p3, p4):
    xdiff_segment = p2[0] - p1[0]
    ydiff_segment = p2[1] - p1[1]
    xdiff_contour = p4[0] - p3[0]
    ydiff_contour = p4[1] - p3[1]

    div = xdiff_segment * ydiff_contour - ydiff_segment * xdiff_contour
    if div == 0:
        return None  # Lines are parallel or coincident
    else:
        dx_contour = p3[0] - p1[0]
        dy_contour = p3[1] - p1[1]
        t = (dx_contour * ydiff_contour - dy_contour * xdiff_contour) / div
        u = (dx_contour * ydiff_segment - dy_contour * xdiff_segment) / div
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_point = np.array([p1[0] + t * xdiff_segment, p1[1] + t * ydiff_segment])
            return intersection_point
        else:
            return None  # Intersection point is not within the line segment


def find_closest_point_on_segment(segment_point1, segment_point2, contour):
    closest_distance = float('inf')
    closest_point = None

    for i in range(len(contour)-1):
        contour_point1 = contour[i]
        contour_point2 = contour[i+1]

        intersection_point = line_segment_intersection(segment_point1, segment_point2, contour_point1, contour_point2)
        if intersection_point is not None:
            distance = np.linalg.norm(intersection_point - segment_point1)
            if distance < closest_distance:
                closest_distance = distance
                closest_point = intersection_point

    return closest_point

def draw_LV_SA(refPts,ref_contour, nbPoints= 5):

    ref1 = refPts[0]
    ref2 = refPts[1]
    TH, _ = np.arctan2(ref1[1] - ref2[1], ref1[0] - ref2[0]), np.sqrt((ref1[0] - ref2[0]) ** 2 + (ref1[1] - ref2[1]) ** 2)
    d = np.sqrt((ref1[0] - ref2[0])**2 + (ref1[1] - ref2[1])**2)
    contour = []
    for k in range(6 * nbPoints, 0, -1):
        x, y = pol2cart(TH + (2 * np.pi) / (nbPoints * 6) * k, d * 1.5)

        # plt.plot([x + ref2[0], ref2[0]], [y + ref2[1], ref2[1]])

        closest_point = find_closest_point_on_segment(np.asarray([ref2[0],ref2[1]]), np.asarray([x + ref2[0], y + ref2[1]]), ref_contour)
        if closest_point is not None:
            contour.append(closest_point)


    contour = np.asarray(contour)
    # plt.plot(contour[:,0],contour[:,1],'go')
    return contour


def draw_RV_SA(refPts,ref_contour, nbPoints=5):
    # plt.plot(ref_contour[:,0],ref_contour[:,1],'go')
    ref1 = refPts[0]
    ref2 = refPts[1]
    ref3 = refPts[2]
    ref4 = refPts[3]

    TH, _ = np.arctan2(ref1[1] - ref2[1], ref1[0] - ref2[0]), np.sqrt((ref1[0] - ref2[0]) ** 2 + (ref1[1] - ref2[1]) ** 2)
    if TH < 0 : TH += 2 * np.pi
    THi, _ = np.arctan2(ref1[1] - ref4[1], ref1[0] - ref4[0]), np.sqrt((ref1[0] - ref4[0]) ** 2 + (ref1[1] - ref4[1]) ** 2)
    if THi < 0 : THi += 2 * np.pi
    THj, _ = np.arctan2(ref3[1] - ref4[1], ref3[0] - ref4[0]), np.sqrt((ref3[0] - ref4[0]) ** 2 + (ref3[1] - ref4[1]) ** 2)
    if THj < 0 : THj += 2 * np.pi
    THk, _ = np.arctan2(ref3[1] - ref2[1], ref3[0] - ref2[0]), np.sqrt((ref3[0] - ref2[0]) ** 2 + (ref3[1] - ref2[1]) ** 2)
    if THk < 0 : THk += 2 * np.pi

    bigA = THi - THj
    if bigA < 0 : bigA += 2 * np.pi
    bigB = TH - THk
    if bigB < 0 : bigB += 2 * np.pi



    d = np.sqrt((ref2[0] - ref4[0]) ** 2 + (ref2[1] - ref4[1]) ** 2)
    contour = []
    
    for k in range(1, 2 * nbPoints +1 ,  1):
        x, y = pol2cart(TH - bigB / (nbPoints + 0.5) / 2 * k, d)

        #plt.plot([x + ref2[0], ref2[0]], [y + ref2[1], ref2[1]])
        closest_point = find_closest_point_on_segment(np.asarray([ref2[0],ref2[1]]), np.asarray([x + ref2[0], y + ref2[1]]), ref_contour)
        if closest_point is not None:
            contour.append(closest_point)

    for k in range(2 * (2 * nbPoints), 0, -1):
        x, y = pol2cart(THi - bigA / (2 * nbPoints + 0.5) / 2 * k, d * 2)
        #plt.plot([x + ref4[0], ref4[0]], [y + ref4[1], ref4[1]])
        closest_point = find_closest_point_on_segment(np.asarray([ref4[0],ref4[1]]), np.asarray([x + ref4[0], y + ref4[1]]), ref_contour)
        if closest_point is not None:
            contour.append(closest_point)


    contour = np.asarray(contour)
    #plt.plot(contour[:,0],contour[:,1],'go')
    return contour




def get_valve_extremitie(valve_coord, landmark=None):
    """
    Get the extremities of a valve contour.

    Parameters:
    - valve_coord: The coordinates of valve points.
    - landmark (optional): The coordinates of the landmark point.

    Returns:
    - closest_point: The coordinate of the closest point to the landmark (if provided), or the two furthest points.
    - furthest_point: The coordinate of the furthest point from the landmark (if provided), or the two furthest points.
    """
    if landmark is not None:
        dist = np.zeros((len(valve_coord)))
        for idx, p1 in enumerate(valve_coord):
            dist[idx] = distance(p1, landmark)
        return valve_coord[dist.argmin()], valve_coord[dist.argmax()]
    else:
        # if no landmark is provided, return the two furthest points
        dist = distance_matrix(valve_coord, valve_coord)
        return valve_coord[dist.argmax()//len(valve_coord)], valve_coord[dist.argmax()%len(valve_coord)]

def distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
        p1: The coordinates of the first point.
        p2: The coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def invert_x_y(contour):
    ch = np.copy(contour[:,0])
    contour[:,0],contour[:,1] = contour[:,1],ch
    return contour

def dist_calc(contour):
    """
    Calculate the total distance between consecutive points in a contour.

    Parameters:
        contour: The coordinates of the contour.

    Returns:
        float: The sum of distances between consecutive points in the contour.
    """
    all_dist = [distance(contour[i+1], contour[i]) for i in range(len(contour)-1)]
    return np.sum(all_dist)

def get_length_contours(Contour):
    """
    Calculate the lengths of different contours.

    Parameters:
    - Contour: The Contour object.

    Returns:
    - perimetre : A list of lengths of the LV contour.

    """

    return [dist_calc(Contour[i]) for i in range(len(Contour)) ]

def get_length_contours_unique(Contour):
    """
    Calculate the lengths of different contours.

    Parameters:
    - Contour: The Contour object.

    Returns:
    - perimetre : A list of lengths of the LV contour.

    """

    return dist_calc(Contour)

from scipy.interpolate import make_interp_spline


def re_interpolate(contour,n=30):
    """
    Re-interpolate a contour to have a specified number of points.

    Parameters:
    - contour: The input contour.
    - n (optional): The number of points in the re-interpolated contour. Default is 100.

    Returns:
    - contour: The re-interpolated contour.
    """
    x = np.array([i for i in range(len(contour))])
    contour_x_ln_ = np.linspace(x.min(), x.max(), n)
    try:
        contour = make_interp_spline(x, contour)
    except:
        print("interpolation contour error")
    contour = contour(contour_x_ln_)
    return contour


def to_mat_struct(liste,idc):
    return [np.array([x1[idc] for x1 in x ]) for x in liste]

def my_interpolation(contour_list, resX):
    out_list = []
    X = to_mat_struct(contour_list,0)
    Y = to_mat_struct(contour_list,1)
    nb_point = len(interpolate_points(X[0], Y[0], 1.5))
    for i in range(len(X)):
        current_x = np.asarray(X[i])
        current_y = np.asarray(Y[i])
        contour_interpoler = interpolate_points(current_x, current_y, 1.5, nb_point)
        #plt.figure()
        #plt.plot(contour_interpoler[:,0],contour_interpoler[:,1],'-+')
        #plt.show()
        out_list.append(get_length_contours_unique(contour_interpoler))
    
    return out_list

#data_mask =  mat73.loadmat("C:/Users/icv-leite/Documents/Data/Mat/Resnet_34/Mask/ID_._20150930_default_user.mat")
#imgs =data_mask["data_mask"]["orientations"]['short_axis']["slices"][3]['label_img']
if __name__ == "__main__":
    depth = 0
    #path_list = glob(r'2023-09-01_16H48\Registered\temp_allClasses\patient010*.nii.gz')
    #path_list = glob(r'2023-09-05_18H18\Registered\temp_allClasses\patient010*.nii.gz')
    path_list = glob(os.path.join(r'custom_lib_t_3\patient001', '*.gz'))
    path_list = [x for x in path_list if '_gt' in x]
    LV_ENDO = []
    LV_EPI = []

    matplotlib.use('QtAgg')
    fig, ax = plt.subplots(1, 1)

    contour_list_endo = []
    contour_list_epi = []

    for path in path_list:
        data = nib.load(path)
        pred = data.get_fdata()
        pred = pred[:, :, depth]
        
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(pred, cmap='gray')
        #plt.show()

        mask_lv_endo,cnt_lv_endo = get_contour(pred, 3)
        mask_lv_epi, cnt_lv_epi = get_contour(pred, [2, 3])
        mask_rv, cnt_rv = get_contour(pred, 1)

        cnt_septum = extract_adj_contour2(cnt_lv_epi, pred,[1,2],precision=4)

        # plt.plot(cnt_septum[:,1],cnt_septum[:,0],"r+")
        # plt.plot(cnt_lv_endo[:,0],cnt_lv_endo[:,1],"b+")
        l1,l3 = get_valve_extremitie(cnt_septum)

        l2 = center_of_mass(mask_lv_endo)
        l4 = center_of_mass(cnt_rv)

        # plt.plot([l1[1],l2[1],l3[1],l4[1]],[l1[0],l2[0],l3[0],l4[0]],"g^")

        refpoints = np.asarray([l1,l2,l3,l4])#[[l1[1],l1[0]],[l2[1],l2[0]],[l3[1],l3[0]],[l4[1],l4[0]]])
        #cnt_lv_endo = re_interpolate(cnt_lv_endo,n=200)
        #cnt_lv_epi = re_interpolate(cnt_lv_epi,n=200)
        point_joli = draw_LV_SA(refpoints,cnt_lv_endo)
        contour_list_endo.append(point_joli)
        #point_joli = np.append(point_joli,[point_joli[0]],axis=0)
        #plt.plot(point_joli[:,1],point_joli[:,0],'r-+')

        point_jolie = draw_LV_SA(refpoints,cnt_lv_epi)
        contour_list_epi.append(point_jolie)
        #point_jolie = np.append(point_jolie,[point_jolie[0]],axis=0)
        #plt.plot(point_jolie[:,1],point_jolie[:,0],'b-+')

        # point_joli = draw_RV_SA(refpoints,cnt_rv)
        LV_ENDO.append(get_length_contours_unique(point_joli))
        LV_EPI.append(get_length_contours_unique(point_jolie))

        #plt.show()
    zoom = data.header.get_zooms()

    LV_ENDO = my_interpolation(contour_list_endo, zoom[0])
    LV_EPI = my_interpolation(contour_list_epi, zoom[0])

    LV_ENDO = np.array(LV_ENDO)
    LV_EPI = np.array(LV_EPI)

    LV_EPI = np.array([(LV_EPI[i] - LV_EPI[0]) / (LV_EPI[0] + 1e-8) for i in range(len(LV_EPI))])
    LV_ENDO = np.array([(LV_ENDO[i] - LV_ENDO[0]) / (LV_ENDO[0] + 1e-8) for i in range(len(LV_ENDO))])

    lv_strain = (LV_EPI + LV_ENDO) / 2


    X = np.arange(len(path_list))
    ax.plot(X, lv_strain, label='LV')
    ax.set_xlim(left=0)
    ax.legend()
    plt.show()









