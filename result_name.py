import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from HDF5DatasetWriter import HDF5DatasetWriter
from HDF5DatasetGenerator import HDF5DatasetGenerator
from utils import *
from tqdm import tqdm
import cv2
import shutil
# 可以在part1之前设定好（即循环外）
seed=1
# data_gen_args = dict(rotation_range=3,
#                     width_shift_range=0.01,
#                     height_shift_range=0.01,
#                     shear_range=0.01,
#                     zoom_range=0.01,
#                     fill_mode='nearest')

# image_datagen = ImageDataGenerator()
# mask_datagen = ImageDataGenerator()
# print('build ImageDataGenerator finished.')

# 可以在part1之前设定好（即循环外）
# 这儿的数量需要提前写好，感觉很不方便，但不知道怎么改，我是先跑了之前的程序，计算了一共有多少
# 张图片后再写的，但这样明显不是好的解决方案
# outputPath = "./data_train/test_liver.h5"

def get_livers(data_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(data_path)
    dicom_names = [s[:-4]+'_mask.png' for s in dicom_names ]
    #livers = np.array([plt.imread('mask.png') for s in range(len(dicom_names))])
    return dicom_names


#full_images2 = []
full_livers2 = []
for i in range(109,111):#后3个人作为测试样本
    if i == 16:
        continue
    data_path = './train/train/' + str(i+1000) + '/arterial phase'
    image_slices = [pydicom.dcmread(data_path + '/' + s) for s in os.listdir(data_path) if '.dcm' in s ]
    image_slices.sort(key = lambda x: int(x.InstanceNumber), reverse = True)#实验证明！逆序的！
    intercept = image_slices[0].RescaleIntercept
    slope = image_slices[0].RescaleSlope
    del image_slices
    livers = get_livers(data_path)
#================================================
    """ 省略进行的预处理操作，具体见part2"""
    #part2
    # 接part1
    # images = get_pixels_hu_by_simpleitk(data_path, intercept, slope)
    # images = transform_ctdata(images,3970,860)
    # #start,end = getRangImageDepth(livers)
    # #images = clahe_equalized(images,start,end)
    # images = clahe_equalized_for_test(images)
    # images /= 255.
    # # 仅提取腹部所有切片中包含了肝脏的那些切片，其余的不要
    # #total = (end ) - (start) +1
    # #print("%d person, total slices %d"%(i,total))
    # # 首和尾目标区域都太小，舍弃
    # #images = images[start:end]
    # print("%d person, images.shape:(%d,)"%(i,images.shape[0]))
    #livers[livers>0] = 1
    #livers = livers[start:end]

   
#=================================================
    #full_images2.append(images)
    full_livers2.extend(livers)
    print("%d Time finished"%(i))


# full_images2 = np.vstack(full_images2)
# full_images2 = np.expand_dims(full_images2,axis=-1)
#full_livers2 = np.vstack(full_livers2)
#full_livers2 = np.expand_dims(full_livers2,axis=-1)

np.save('name.npy', full_livers2)

# if os.path.exists(outputPath):
#   os.remove(outputPath)



# dataset = HDF5DatasetWriter(image_dims=(full_images2.shape[0], full_images2.shape[1], full_images2.shape[2], 1),
#                             mask_dims=(full_images2.shape[0], full_images2.shape[1], full_images2.shape[2], 1),
#                             outputPath=outputPath)

# print('build HDF5DatasetWriter finished')
# dataset.add(full_images2, full_livers2)

# print("total images in val ",dataset.close())

num_pic = len(full_livers2)

for i in range(num_pic):
    #0_pred.png
    src = './preds/'+str(i)+'_pred.png'
    dst = full_livers2[i]
    shutil.copyfile(src,dst)
