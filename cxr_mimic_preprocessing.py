# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
import time
import random 
import copy
import PIL
from PIL import Image
from tqdm import tqdm
import torchvision
from torchvision import datasets, models, transforms
# import zipfile

# data paths
# root dir: root path where necessary files are stored
# data dir: where image files are stored (change the path if it is stored somewhere else)
# res_dir: where resulting annotations and image matrix should be saved
root_dir = '/home/csi22304/diff-bias-proxies/' # to be filled
data_dir = '/home/csi22304/physionet/physionet.org/files/mimic-cxr-jpg/2.0.0/' # to be filled
res_dir = '/home/csi22304/diff-bias-proxies/res_dir/' # to be filled

# patient info - get patients.csv file from MIMIC-IV
patients = pd.read_csv(root_dir + 'patients.csv')
patients = patients[['subject_id','gender']]

# admission info for gender - get admissions.csv file from MIMIC-IV
admissions = pd.read_csv(root_dir + 'admissions.csv')
admissions = admissions[['subject_id','insurance','ethnicity','marital_status']]
admissions = admissions.drop_duplicates(subset=['subject_id'], keep='last')
admissions = admissions.sort_values(by=['subject_id'])

# chexpert labels
labels = pd.read_csv(root_dir + 'mimic-cxr-2.0.0-chexpert.csv')

# meta data
df = pd.read_csv(root_dir + 'mimic-cxr-2.0.0-metadata.csv')
df = df[['dicom_id','subject_id','study_id','ViewPosition']]
df.drop(df[(df['ViewPosition'] != 'PA') & (df['ViewPosition'] != 'AP') ].index, inplace=True)
# note that studies s52250236, s53104432, s55170157, s55802622 and s59316002 are also dropped

# merge with tables for patientand label info
df = pd.merge(df, patients)
df = pd.merge(df,admissions)
df = pd.merge(df, labels)

# add a path column
df['path'] = 'files/p' + df['subject_id'].astype(str).str[:2] + '/p' + df['subject_id'].astype(str) + '/s' + df['study_id'].astype(str) + '/' + df['dicom_id'].astype(str) + '.jpg'

# resize parameter
transResize = 128


# create img_mat
img_mat = np.zeros((len(df),transResize,transResize))
df2 = df.copy()
    
# initialize cnt
cnt = 0
# with zipfile.ZipFile(data_dir, 'r') as z:
        
# iterate through files
for filename in tqdm(df['path']):
        
    # read image
    try: 
        img = PIL.Image.open(data_dir+filename).convert('RGB') 
    except:
        # drop the row
        print(data_dir+filename)
        df2.drop(df2[df2['path'] == filename].index, inplace=True)
        continue
        
    # cut depending on the size
    width, height = img.size
    r_min = max(0,(height-width)/2)
    r_max = min(height,(height+width)/2)
    c_min = max(0,(width-height)/2)
    c_max = min(width,(height+width)/2)
    img = img.crop((c_min,r_min,c_max,r_max))
        
    # hist equalize and reshape
    img = img.resize((transResize,transResize))
    img = PIL.ImageOps.equalize(img)
    img = img.convert('L')
            
    # assign
    img_mat[cnt,:,:] = np.array(img)   
        
    # increment
    cnt = cnt + 1


# save
img_mat = img_mat[0:len(df2),:,:]
np.save(res_dir + 'files_' + str(transResize) + '.npy', img_mat)

# save dataframe as csv
df2.to_csv(res_dir + 'meta_data.csv',index=False)