# Import libraries
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
import zipfile

# def concatenate_npy_files(res_dir, transResize, num_chunks):
#     output_file_path = os.path.join(res_dir, f'files_{transResize}_concatenated.npy')

#     for i in range(1, num_chunks + 1):
#         print("Processing numpy chunk:", i)

#         chunk_file_path = os.path.join(res_dir, f'files_{transResize}_chunk{i}.npy')
#         try:
#             img_mat_chunk = np.load(chunk_file_path)

#             if i == 1:
#                 # For the first chunk, save it directly
#                 np.save(output_file_path, img_mat_chunk)
#             else:
#                 # For subsequent chunks, open the output file, extract data, and append new chunk
#                 with np.load(output_file_path) as existing_data:
#                     existing_array = existing_data['arr_0']
#                     combined_array = np.concatenate([existing_array, img_mat_chunk], axis=0)

#                 # Save the combined array back to the file
#                 np.save(output_file_path, combined_array)

#         except Exception as e:
#             print(f"Error processing numpy chunk {i}: {e}")

#     print("Numpy chunks concatenated and saved.")

def load_data_paths(root_dir, data_dir, res_dir):
    # Load patient info
    patients = pd.read_csv(root_dir + 'patients.csv')
    patients = patients[['subject_id', 'gender']]

    # Load admission info for gender
    admissions = pd.read_csv(root_dir + 'admissions.csv')
    admissions = admissions[['subject_id', 'insurance', 'ethnicity', 'marital_status']]
    admissions = admissions.drop_duplicates(subset=['subject_id'], keep='last')
    admissions = admissions.sort_values(by=['subject_id'])

    # Load chexpert labels
    labels = pd.read_csv(root_dir + 'mimic-cxr-2.0.0-chexpert.csv')

    # Load meta data
    df = pd.read_csv(root_dir + 'mimic-cxr-2.0.0-metadata.csv')
    df = df[['dicom_id', 'subject_id', 'study_id', 'ViewPosition']]
    df.drop(df[(df['ViewPosition'] != 'PA') & (df['ViewPosition'] != 'AP')].index, inplace=True)

    # Merge with tables for patient and label info
    df = pd.merge(df, patients)
    df = pd.merge(df, admissions)
    df = pd.merge(df, labels)

    # Add a path column
    df['path'] = 'files/p' + df['subject_id'].astype(str).str[:2] + '/p' + df['subject_id'].astype(str) + '/s' + df['study_id'].astype(str) + '/' + df['dicom_id'].astype(str) + '.jpg'

    return df

def process_images_chunk(df_chunk, data_dir, transResize):
    img_mat = np.zeros((len(df_chunk), transResize, transResize))
    df2 = df_chunk.copy()
    cnt = 0

    for filename in tqdm(df_chunk['path']):
        try:
            img = PIL.Image.open(data_dir + filename).convert('RGB')
        except:
            print(data_dir + filename)
            df2.drop(df2[df2['path'] == filename].index, inplace=True)
            continue

        width, height = img.size
        r_min = max(0, (height - width) / 2)
        r_max = min(height, (height + width) / 2)
        c_min = max(0, (width - height) / 2)
        c_max = min(width, (height + width) / 2)
        img = img.crop((c_min, r_min, c_max, r_max))

        img = img.resize((transResize, transResize))
        img = PIL.ImageOps.equalize(img)
        img = img.convert('L')

        img_mat[cnt, :, :] = np.array(img)
        cnt = cnt + 1

    return img_mat, df2

def save_results_chunk(img_mat, df_chunk, res_dir, transResize, chunk_num):
    img_mat = img_mat[0:len(df_chunk), :, :]
    np.save(res_dir + f'files_{transResize}_chunksmall{chunk_num}.npy', img_mat)
    df_chunk.to_csv(res_dir + f'meta_data_chunksmall{chunk_num}.csv', index=False)

# def concatenate_chunks(res_dir, transResize, num_chunks):
#     img_mats = []
#     dfs = []

#     for i in range(num_chunks):
#         img_mat_chunk = np.load(res_dir + f'files_{transResize}_chunk{i+1}.npy')
#         df_chunk = pd.read_csv(res_dir + f'meta_data_chunk{i+1}.csv')

#         img_mats.append(img_mat_chunk)
#         dfs.append(df_chunk)

#     img_mat_concat = np.concatenate(img_mats, axis=0)
#     df_concat = pd.concat(dfs, ignore_index=True)

#     # Save the concatenated results
#     np.save(res_dir + f'files_{transResize}_concatenated.npy', img_mat_concat)
#     df_concat.to_csv(res_dir + f'meta_data_concatenated.csv', index=False)



# Set paths
root_dir = '/home/csi22304/diff-bias-proxies/'
data_dir = '/home/csi22304/physionet/physionet.org/files/mimic-cxr-jpg/2.0.0/'
res_dir = '/home/csi22304/diff-bias-proxies/res_dir/'

# Resize parameter
transResize = 128

# Load data paths
df = load_data_paths(root_dir, data_dir, res_dir)

# Split data into chunks (for example, split into 3 chunks)
chunk_size = len(df) // 6
remainder = len(df) % 6

# Create chunks
chunks = [df[i:i + chunk_size] for i in range(0, len(df) - remainder, chunk_size)]

# Add the remaining images to the last chunk
chunks[-1] = pd.concat([chunks[-1], df.tail(remainder)], ignore_index=True)

# Process and save each chunk
for i, df_chunk in enumerate(chunks):
    if i + 1 < 4: 
        print("skipping step:", i+1)
        continue
    img_mat_chunk, df2_chunk = process_images_chunk(df_chunk, data_dir, transResize)
    save_results_chunk(img_mat_chunk, df2_chunk, res_dir, transResize, i + 1)
    print(f"Chunk {i + 1} processed and saved.")

    break

# # Concatenate all chunks
# num_chunks = len(chunks)
# concatenate_chunks(res_dir, transResize, num_chunks)
# print("Chunks concatenated and saved.")


import numpy as np
import pandas as pd
import os

# def concatenate_npy_files(res_dir, transResize, num_chunks):
    # output_file_path = os.path.join(res_dir, f'files_{transResize}.npy')
# 
    # for i in range(1, num_chunks + 1):
        # print("Processing numpy chunk:", i)
# 
        # chunk_file_path = os.path.join(res_dir, f'files_{transResize}_chunk{i}.npy')
        # try:
            # img_mat_chunk = np.load(chunk_file_path)
# 
            # if i == 1:
                # For the first chunk, create a memory-mapped array
                # np.save(output_file_path, img_mat_chunk)
            # else:
                # For subsequent chunks, open the memory-mapped array and append data
                # existing_array = np.lib.format.open_memmap(output_file_path, mode='r+')
                # combined_array = np.concatenate([existing_array, img_mat_chunk], axis=0)
                # existing_array.resize(combined_array.shape)  # Resize the memory-mapped array
                # existing_array[:] = combined_array  # Update the array on disk
# 
        # except Exception as e:
            # print(f"Error processing numpy chunk {i}: {e}")
# 
    # print("Numpy chunks concatenated and saved.")

# def concatenate_npy_files(res_dir, transResize, num_chunks):
#     output_file_path = os.path.join(res_dir, f'files_{transResize}.npz')

#     for i in range(1, num_chunks + 1):
#         print("Processing numpy chunk:", i)

#         chunk_file_path = os.path.join(res_dir, f'files_{transResize}_chunk{i}.npy')
#         try:
#             img_mat_chunk = np.load(chunk_file_path)

#             if i == 1:
#                 # For the first chunk, save it directly
#                 np.savez(output_file_path, img_mat_chunk=img_mat_chunk)
#             else:
#                 # For subsequent chunks, append to the zip archive
#                 with np.load(output_file_path) as existing_data:
#                     existing_array = existing_data['img_mat_chunk']
#                     combined_array = np.concatenate([existing_array, img_mat_chunk], axis=0)

#                 # Save the combined array back to the zip archive
#                 np.savez(output_file_path, img_mat_chunk=combined_array)
#         except Exception as e:
#             print(f"Error processing numpy chunk {i}: {e}")

#     print("Numpy chunks concatenated and saved.")
    
# def concatenate_npy_files(res_dir, transResize, num_chunks):
#     output_file_path = os.path.join(res_dir, f'files_{transResize}_concatenated.npy')

#     with open(output_file_path, 'wb') as output_file:
#         for i in range(1, num_chunks+1):
#             print("Processing numpy chunk:", i)

#             chunk_file_path = os.path.join(res_dir, f'files_{transResize}_chunk{i}.npy')
#             try:
#                 # Load numpy chunk data
#                 img_mat_chunk = np.load(chunk_file_path)

#                 # Save the chunk to the concatenated file
#                 if i == 1:
#                     # For the first chunk, simply save it
#                     np.save(output_file, img_mat_chunk)
#                 else:
#                     # For subsequent chunks, append the data
#                     np.save(output_file, img_mat_chunk, allow_pickle=False)
#             except Exception as e:
#                 print(f"Error processing numpy chunk {i}: {e}")

#     print("Numpy chunks concatenated and saved.")

# def concatenate_npy_files(res_dir, transResize, num_chunks):
#     output_file_path = os.path.join(res_dir, f'files_{transResize}_concatenated.npy')

#     with open(output_file_path, 'wb') as output_file:
#         for i in range(1, num_chunks+1):
#             print("Processing numpy chunk:", i)

#             chunk_file_path = os.path.join(res_dir, f'files_{transResize}_chunk{i}.npy')
#             try:
#                 # Load numpy chunk data
#                 img_mat_chunk = np.load(chunk_file_path)

#                 # Save the chunk to the concatenated file
#                 np.save(output_file, img_mat_chunk)
#             except Exception as e:
#                 print(f"Error processing numpy chunk {i}: {e}")

#     print("Numpy chunks concatenated and saved.")

def concatenate_csv_files(res_dir, transResize, num_chunks):
    output_file_path = os.path.join(res_dir, f'meta_data_concatenated.csv')

    for i in range(1, num_chunks+1):
        print("Processing CSV chunk:", i)

        chunk_file_path = os.path.join(res_dir, f'meta_data_chunk{i}.csv')
        try:
            # Load CSV chunk data
            df_chunk = pd.read_csv(chunk_file_path)

            # Save the chunk to the concatenated file
            df_chunk.to_csv(output_file_path, mode='a', index=False, header=(i == 1))
        except Exception as e:
            print(f"Error processing CSV chunk {i}: {e}")

    print("CSV chunks concatenated and saved.")

# Set paths
root_dir = '/home/csi22304/diff-bias-proxies/'
data_dir = '/home/csi22304/physionet/physionet.org/files/mimic-cxr-jpg/2.0.0/'
res_dir = '/home/csi22304/diff-bias-proxies/res_dir/'

# Resize parameter
transResize = 128

num_chunks = 4

# Concatenate numpy files
# concatenate_npy_files(res_dir, transResize, num_chunks)

# Concatenate CSV files
# concatenate_csv_files(res_dir, transResize, num_chunks)

# print("Chunks concatenated and saved.")
