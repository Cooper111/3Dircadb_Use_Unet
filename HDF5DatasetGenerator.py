# -*- coding: utf-8 -*-
import h5py
import os
import numpy as np

class HDF5DatasetGenerator:
    
    def __init__(self, dbPath, batchSize, preprocessors=None,
                 aug=None, binarize=True, classes=2):
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        
        self.db = h5py.File(dbPath)
        self.numImages = self.db["images"].shape[0]
#        self.numImages = total
        print("total images:",self.numImages)
        self.num_batches_per_epoch = int((self.numImages-1)/batchSize) + 1
        
    
    def generator(self, shuffle=True, passes=np.inf):
        epochs = 0
        
        while epochs < passes:
            shuffle_indices = np.arange(self.numImages) 
            shuffle_indices = np.random.permutation(shuffle_indices)
            for batch_num in range(self.num_batches_per_epoch):
                
                start_index = batch_num * self.batchSize
                end_index = min((batch_num + 1) * self.batchSize, self.numImages)
                
                # h5py get item by index,参数为list，而且必须是增序
                batch_indices = sorted(list(shuffle_indices[start_index:end_index]))
                
                images = self.db["images"][batch_indices,:,:,:]
                labels = self.db["masks"][batch_indices,:,:,:]
                
#                if self.binarize:
#                    labels = np_utils.to_categorical(labels, self.classes)
                
                if self.preprocessors is not None:
                    procImages = []
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        procImages.append(image)
                    
                    images = np.array(procImages)
                
                if self.aug is not None:
                    # 不知道意义何在？本身images就有batchsize个了
                    (images, labels) = next(self.aug.flow(images, labels,
                                                        batch_size=self.batchSize))
                yield (images, labels)
            
            epochs += 1
            
    def close(self):
        self.db.close()
