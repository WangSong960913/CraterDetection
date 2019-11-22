#!/usr/bin/env python
"""Unique Crater Distribution Functions

Functions for extracting craters from model target predictions and filtering
out duplicates.
"""
from __future__ import absolute_import, division, print_function
from PIL import Image
import matplotlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import utils.template_match_target as tmt
import utils.processing as proc
import utils.transform as trf
from keras.models import load_model
import os
import pandas as pd
from input_data_gen import ringmaker,circlemaker,get_merge_indices
#########################
def get_model_preds(CP):
    """Reads in or generates model predictions.

    Parameters
    ----------
    CP : dict
        Containins directory locations for loading data and storing
        predictions.

    Returns
    -------
    craters : h5py
        Model predictions.
    """
    n_imgs, dtype = CP['n_imgs'], CP['datatype']

    data = h5py.File(CP['dir_data'], 'r')

    Data = {
        dtype: [data['input_images'][:n_imgs].astype('float32'),
                data['target_masks'][:n_imgs].astype('float32')]
    }
    data.close()
    proc.preprocess(Data)

    model = load_model(CP['dir_model'])
    preds=[]
    for i in range(0,n_imgs,2):
        pred = model.predict(Data[dtype][0][i:i+2])
        for j in range(len(pred)):
            preds.append(pred[j])
    # save
    h5f = h5py.File(CP['dir_preds'], 'w')
    h5f.create_dataset(dtype, data=preds)
    #h5f.close()
    print("Successfully generated and saved model predictions.")
    return preds
def get_data(CP):
    """Reads in or generates model predictions.

    Parameters
    ----------
    CP : dict
        Containins directory locations for loading data and storing
        predictions.

    Returns
    -------
    craters : h5py
        Model predictions.
    """
    n_imgs, dtype = CP['n_imgs'], CP['datatype']

    data = h5py.File(CP['dir_data'], 'r')

    Data = {
        dtype: [data['input_images'][:n_imgs].astype('float32'),
                data['target_masks'][:n_imgs].astype('float32')]
    }
    data.close()
    craters = pd.HDFStore(CP['crater_data'], 'r')
    csvs=[]
    minrad, maxrad, cutrad, n_csvs ,dim= 3, 50, 0.8, len(craters),256
    diam = 'Diameter (pix)'
    for i in range(n_csvs):
        csv = craters[proc.get_id(i,2)]
        # remove small/large/half craters
        csv = csv[(csv[diam] < 2 * maxrad) & (csv[diam] > 2 * minrad)]
        csv = csv[(csv['x'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['y'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['x'] - cutrad * csv[diam] / 2 > 0)]
        csv = csv[(csv['y'] - cutrad * csv[diam] / 2 > 0)]

        if len(csv) < 1:  # Exclude csvs with few craters
            csvs.append([-1])
        else:
            csv_coords = np.asarray((csv['x'], csv['y'], csv[diam] / 2)).T
            csvs.append(csv_coords)
    return Data,csvs
def get_coords_classification(detect_coords,note_coords,longlat_thresh2 = 1.8,rad_thresh = 1.0):
    true_carter=[]
    detect_carter=[]
    Undetected_carter=[]
    detect_list=[]
    note_list=[]
    for i in range(len(detect_coords)):
        lo,la,r=detect_coords[i]
        for j in range(len(note_coords)):

            #print(note_coords)
            Long,Lat,Rad=note_coords[j]
            minr = np.minimum(r, Rad)
            dL = ((Long - lo)**2 + (Lat - la)**2) / minr**2
            dR = abs(Rad - r) / minr
            if (dR < rad_thresh) & (dL < longlat_thresh2):
                detect_list.append(i)
                note_list.append(j)
                true_carter.append(detect_coords[i])
                break
    for k in range(len(note_coords)):
        if k not in note_list:
            Undetected_carter.append(note_coords[k])
    for k in range(len(detect_coords)):
        if k not in detect_list:
            detect_carter.append(detect_coords[k])
    return true_carter,detect_carter,Undetected_carter
def draw_pic(img,detect_coords,note_coords,save_path):
    true_carter,detect_carter,Undetected_carter=get_coords_classification(detect_coords,note_coords)
    true_carter_color = (255,0,0)#blue
    detect_carter_color = (0,255,0)#green
    Undetected_carter_color=(0,0,255)#red

    ring_width=2
    for x,y,r in true_carter:
        cv2.circle(img, (int(x), int(y)), int(r), true_carter_color, ring_width)
    for x,y,r in detect_carter:
        cv2.circle(img, (int(x), int(y)), int(r), detect_carter_color, ring_width)
    for x,y,r in Undetected_carter:
         cv2.circle(img, (int(x), int(y)), int(r), Undetected_carter_color, ring_width)
    cv2.imwrite(save_path,img)

#########################
def add_unique_craters(craters, craters_unique, thresh_longlat2, thresh_rad):
    """Generates unique crater distribution by filtering out duplicates.

    Parameters
    ----------
    craters : array
        Crater tuples from a single image in the form (long, lat, radius).
    craters_unique : array
        Master array of unique crater tuples in the form (long, lat, radius)
    thresh_longlat2 : float.
        Hyperparameter that controls the minimum squared longitude/latitude
        difference between craters to be considered unique entries.
    thresh_rad : float
        Hyperparaeter that controls the minimum squared radius difference
        between craters to be considered unique entries.

    Returns
    -------
    craters_unique : array
        Modified master array of unique crater tuples with new crater entries.
    """
    k2d = 180. / (np.pi * 1737.4)       # km to deg
    Long, Lat, Rad = craters_unique.T
    for j in range(len(craters)):
        lo, la, r = craters[j].T
        la_m = (la + Lat) / 2.
        minr = np.minimum(r, Rad)       # be liberal when filtering dupes

        # duplicate filtering criteria
        dL = (((Long - lo) / (minr * k2d / np.cos(np.pi * la_m / 180.)))**2
              + ((Lat - la) / (minr * k2d))**2)
        dR = np.abs(Rad - r) / minr
        index = (dR < thresh_rad) & (dL < thresh_longlat2)

        if len(np.where(index == True)[0]) == 0:
            craters_unique = np.vstack((craters_unique, craters[j]))
    return craters_unique

#########################
def estimate_longlatdiamkm(dim, llbd, distcoeff, coords):
    """First-order estimation of long/lat, and radius (km) from
    (Orthographic) x/y position and radius (pix).

    For images transformed from ~6000 pixel crops of the 30,000 pixel
    LROC-Kaguya DEM, this results in < ~0.4 degree latitude, <~0.2
    longitude offsets (~2% and ~1% of the image, respectively) and ~2% error in
    radius. Larger images thus may require an exact inverse transform,
    depending on the accuracy demanded by the user.

    Parameters
    ----------
    dim : tuple or list
        (width, height) of input images.
    llbd : tuple or list
        Long/lat limits (long_min, long_max, lat_min, lat_max) of image.
    distcoeff : float
        Ratio between the central heights of the transformed image and original
        image.
    coords : numpy.ndarray
        Array of crater x coordinates, y coordinates, and pixel radii.

    Returns
    -------
    craters_longlatdiamkm : numpy.ndarray
        Array of crater longitude, latitude and radii in km.
    """
    # Expand coords.
    long_pix, lat_pix, radii_pix = coords.T

    # Determine radius (km).
    km_per_pix = 1. / trf.km2pix(dim[1], llbd[3] - llbd[2], dc=distcoeff)
    radii_km = radii_pix * km_per_pix

    # Determine long/lat.
    deg_per_pix = km_per_pix * 180. / (np.pi * 1737.4)
    long_central = 0.5 * (llbd[0] + llbd[1])
    lat_central = 0.5 * (llbd[2] + llbd[3])

    # Iterative method for determining latitude.
    lat_deg_firstest = lat_central - deg_per_pix * (lat_pix - dim[1] / 2.)
    latdiff = abs(lat_central - lat_deg_firstest)
    # Protect against latdiff = 0 situation.
    latdiff[latdiff < 1e-7] = 1e-7
    lat_deg = lat_central - (deg_per_pix * (lat_pix - dim[1] / 2.) *
                             (np.pi * latdiff / 180.) /
                             np.sin(np.pi * latdiff / 180.))
    # Determine longitude using determined latitude.
    long_deg = long_central + (deg_per_pix * (long_pix - dim[0] / 2.) /
                               np.cos(np.pi * lat_deg / 180.))

    # Return combined long/lat/radius array.
    return np.column_stack((long_deg, lat_deg, radii_km))

def extract_unique_craters(CP, craters_unique):
    """Top level function that extracts craters from model predictions,
    converts craters from pixel to real (degree, km) coordinates, and filters
    out duplicate detections across images.

    Parameters
    ----------
    CP : dict
        Crater Parameters needed to run the code.
    craters_unique : array
        Empty master array of unique crater tuples in the form
        (long, lat, radius).

    Returns
    -------
    craters_unique : array
        Filled master array of unique crater tuples.
    """ 

    # Load/generate model preds
    try:
        preds = h5py.File(CP['dir_preds'], 'r')[CP['datatype']]

        print("Loaded model predictions successfully")
    except:
        print("Couldnt load model predictions, generating")
        preds = get_model_preds(CP)
    Data,Carters=get_data(CP)
    # need for long/lat bounds
    P = h5py.File(CP['dir_data'], 'r')
    llbd, pbd, distcoeff = ('longlat_bounds', 'pix_bounds',
                            'pix_distortion_coefficient')
    #r_moon = 1737.4
    dim = (float(CP['dim']), float(CP['dim']))

    N_matches_tot = 0
    if not os.path.exists(CP['result_img']):
        os.mkdir(CP['result_img'])
    lenstr=""
    lenstr1="true_carter"
    lenstr2="detect_carter"
    lenstr3="undetected_carter"
    num=0
    num1=0
    num2=0
    num3=0
    for i in range(CP['n_imgs']):
        id = proc.get_id(i,2)
        print("Drawing picture:%d" %i)
        input_images=Data[CP['datatype']][0][i]
        imgs = Image.fromarray(input_images.astype('uint8')).convert('RGB')
        img = cv2.cvtColor(np.asarray(imgs),cv2.COLOR_RGB2BGR)

        coords = tmt.template_match_t(preds[i])
        num=num+len(coords)
        lenstr=lenstr+" "+str(len(coords))
        matplotlib.image.imsave(CP['result_img']+"/"+str(i)+'_mask.jpg', preds[i])
        true_carter,detect_carter,Undetected_carter=get_coords_classification(coords,Carters[i])
        lenstr1=lenstr1+" "+str(len(true_carter))
        num1=num1+len(true_carter)
        lenstr2=lenstr2+" "+str(len(detect_carter))
        num2=num2+len(detect_carter)
        lenstr3=lenstr3+" "+str(len(Undetected_carter))
        num3=num3+len(Undetected_carter)
        draw_pic(img,coords,Carters[i],CP['result_img']+"/"+str(i)+'.jpg')
        if len(coords) > 0:
            # for i in range(len(coords)):
            new_craters_unique = estimate_longlatdiamkm(
                dim, P[llbd][id], P[distcoeff][id][0], coords)
            N_matches_tot += len(coords)
            #print(id,new_craters_unique)
            # Only add unique (non-duplicate) craters
            if len(craters_unique) > 0:
                craters_unique = add_unique_craters(new_craters_unique,
                                                    craters_unique,
                                                    CP['llt2'], CP['rt2'])
            else:
                craters_unique = np.concatenate((craters_unique,
                                                 new_craters_unique))
    print(lenstr)
    print("total num:%d" %num)
    print(lenstr1)
    print(num1)
    print(lenstr2)
    print(num2)
    print(lenstr3)
    print(num3)
    np.save(CP['dir_result'], craters_unique)
    return craters_unique
