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
from skimage import io
from Unet import *

# partB 接partA
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1
TOTAL = 2782 # 总共的训练数据
TOTAL_VAL = 152 # 总共的validation数据
# part1部分储存的数据文件
outputPath = './data_train/train_liver.h5' # 训练文件
val_outputPath = './data_train/val_liver.h5'
test_outputPath = './data_train/test_liver.h5'
#checkpoint_path = 'model.ckpt'
BATCH_SIZE = 2 # 根据服务器的GPU显存进行调整

#os.environ["CUDA_VISIBLE_DEVICES"] = "6"

print('-'*30)
print('Loading and preprocessing test data...')
test_reader = HDF5DatasetGenerator(dbPath=test_outputPath,batchSize=BATCH_SIZE)
test_iter = test_reader.generator(shuffle=False, passes=1)
print('-'*30)


print('-'*30)
model = get_unet()
print('Loading saved weights...')
print('-'*30)
model.load_weights('./models/dont_change_lr/weights_unet-18--0.76.h5')
    




# print('-' * 30)
# print('Saving predicted masks to files...')
# np.save('imgs_mask_test.npy', imgs_mask_test)
# print('-' * 30)

pred_dir = 'preds'
if not os.path.exists(pred_dir):
	os.mkdir(pred_dir)

i = 0
for fixed_test_images, fixed_test_masks in test_iter:
	print('-'*30)
	print('Predicting masks on test data...')
	imgs_mask_test = model.predict(fixed_test_images, verbose=1)
	print('-'*30)
	print('imgs_mask_test',imgs_mask_test.shape)
	j = 0
	for image in imgs_mask_test:
		print('image',image.shape)
		image = (image[:, :, 0] * 255.).astype(np.uint8)
		gt = (fixed_test_masks[j,:,:,0] * 255.).astype(np.uint8)
		ini = (fixed_test_images[j,:,:,0] *255.).astype(np.uint8)
		io.imsave(os.path.join(pred_dir, str(i*BATCH_SIZE+j)+'_ini.png'), ini)
		io.imsave(os.path.join(pred_dir, str(i*BATCH_SIZE+j)+'_pred.png'), image)
		#io.imsave(os.path.join(pred_dir, str(i*BATCH_SIZE+j)+'_gt.png'), gt)
		j += 1
	i += 1

print("total images in test ",str(i))