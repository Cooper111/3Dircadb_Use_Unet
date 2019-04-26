import numpy as np
import cv2
import re
import sys

def get_pixels_hu(scans):
    #type(scans[0].pixel_array)
    #Out[15]: numpy.ndarray
    #scans[0].pixel_array.shape
    #Out[16]: (512, 512)
    # image.shape: (129,512,512)
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def transform_ctdata(image, windowWidth, windowCenter, normal=False):
        """
        注意，这个函数的self.image一定得是float类型的，否则就无效！
        return: trucated image according to window center and window width
        """
        minWindow = float(windowCenter) - 0.5*float(windowWidth)
        newimg = (image - minWindow) / float(windowWidth)
        newimg[newimg < 0] = 0
        newimg[newimg > 1] = 1
        if not normal:
            newimg = (newimg * 255).astype('uint8')
        return newimg

def getRangImageDepth(image):
    """
    args:
    image ndarray of shape (depth, height, weight)
    """
    # 得到轴向上出现过目标（label>=1)的切片
    z = np.any(image, axis=(1,2)) # z.shape:(depth,)
    startposition,endposition = np.where(z)[0][[0,-1]]
    return startposition, endposition

def getRangImageDepth(image):
    """
    args:
    image ndarray of shape (depth, height, weight)
    """
    firstflag = True
    startposition = 0
    endposition = 0
    for z in range(image.shape[0]):
        notzeroflag = np.max(image[z])
        if notzeroflag and firstflag:
            startposition = z
            firstflag = False
        if notzeroflag:
            endposition = z
    return startposition, endposition

def clahe_equalized(imgs,start,end):
   assert (len(imgs.shape)==3)  #3D arrays
   #create a CLAHE object (Arguments are optional).
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   imgs_equalized = np.empty(imgs.shape)
   for i in range(start, end+1):
       imgs_equalized[i,:,:] = clahe.apply(np.array(imgs[i,:,:], dtype = np.uint8))
   return imgs_equalized

def get_highest_acc():
    pattern = 'model.(?P<epoch>\d+)-(?P<val_acc>[0-9]*\.?[0-9]*).hdf5'
    p = re.compile(pattern)
    acces = [float(p.match(f).groups()[1]) for f in os.listdir('models/') if p.match(f)]
    if len(acces) == 0:
        return sys.float_info.min
    else:
        return np.max(acces)