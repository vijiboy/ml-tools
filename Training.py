import cv2
import numpy as np
import logging as log
import os

import ImageSplitting as Imaging

def getTrainingDataFromFolder(FolderPath):
    trainingData = None
    grassFiles =  [f for f in os.listdir(FolderPath)
                   if os.path.isfile(os.path.join(FolderPath,f)) ] 
    log.debug(grassFiles)
    blockWidth, blockHeight = 30,30
    splittingBlock = Imaging.SplittingBlock(blockWidth, blockHeight,
                                            OverlapHorizontal=0, OverlapVertical=0)
    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                        svm_type = cv2.SVM_C_SVC)
    for imgName in grassFiles:
        imgFileName = os.path.join(FolderPath, imgName)
        if not os.path.isfile(imgFileName):
            log.warning("Training: File not found: {}".format(imgFileName))
            continue

        img = cv2.imread(imgFileName,0) # data = grayscale pixel values
        if img is None:
            log.warning("Training: Unable to load Image {}".format(imgFileName))
            continue

        log.debug('loaded image: {}; shape: {}'.format(imgFileName, img.shape))
        imgBlockStructure = Imaging.SplitImageinBlocksByShifting(img,splittingBlock)
        log.debug('imgBlockStructure for image: {}'.format(
            imgblock.shape for imgblock
            in Imaging.iterateImageBlocks(imgBlockStructure)))
        imgBlockStack = np.vstack(
            [np.array(imgBlock.flatten()) for imgBlock
             in Imaging.iterateImageBlocks(imgBlockStructure)
             if imgBlock.shape == (blockWidth,blockHeight)]
            )
        log.debug('imgBlockStack for image: {}'.format(len(imgBlockStack)))
        if trainingData is None: trainingData = imgBlockStack
        else: trainingData = np.vstack([trainingData,imgBlockStack])

    log.debug('trainData.shape {}'.format(trainingData.shape))
    return trainingData

