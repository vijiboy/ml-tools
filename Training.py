import cv2
import numpy as np
import logging as log
import os
import random

import ImageSplitting as Imaging


def prepareTrainingDataFromImageStrucuture(FolderPath):
    ''' Returns tuple containing trainingData matrix, label matrix, labels dict '''
    log.debug('Call: prepareTrainingDataFromImageStrucuture()')
    trainingData = None
    labels = dict()
    labelData = []
    imageStructure = Imaging.ImageStructureCreateFromFolder(FolderPath)
    log.debug('Image Structure: {}'.format(imageStructure))
    flatStructure = Imaging.getFlattenedStructure(imageStructure).viewitems()
    log.debug('flattened image structure: {}'.format(flatStructure))
    imageFilenames = [k for k,v in flatStructure]
    random.shuffle(imageFilenames)
    log.debug('image list shuffled: {}'.format(imageFilenames))
    blockWidth, blockHeight = 70,70
    splittingBlock = Imaging.SplittingBlock(blockWidth, blockHeight,
                                            OverlapHorizontal=20, OverlapVertical=20)
    log.debug('image splitting: {}'.format(splittingBlock))
    for imgFileName in imageFilenames:
        img = cv2.imread(imgFileName,0) # data = grayscale pixel values
        # changing to float type early for training requirement
        if img is None:
            log.warning("Training: Unable to load Image {}".format(imgFileName))
            continue
        log.debug('loaded image: {}; shape: {}'.format(imgFileName, img.shape))
        # split image into smaller blocks 
        imgBlockStructure = Imaging.SplitImageinBlocksByShifting(img,splittingBlock)
        log.debug('image:{}, BlockStructure: {}'.
                  format(imgFileName, [imgblock.shape for imgblock in Imaging.iterateImageBlocks(imgBlockStructure)]))
        # 1. Select blocks of same size,
        # 2. Flatten them to 1D matrices
        # 3. Stack them in one array
        # TODO replace deprecated vstack with stack
        imgBlockStack = np.vstack(
            [np.array(imgBlock.flatten(), dtype=np.float) for imgBlock
             in Imaging.iterateImageBlocks(imgBlockStructure)
             if imgBlock.shape == (blockWidth,blockHeight)]
            )
        if imgBlockStack is None or len(imgBlockStack) == 0:
            continue 
        log.debug('image:{}, Blocks: stacked: {}, shape:{} '.format(imgFileName, len(imgBlockStack), imgBlockStack[0].shape))

        # append new image blocks to trainingData
        if trainingData is None: trainingData = imgBlockStack
        else: trainingData = np.vstack([trainingData,imgBlockStack])

        # create Label for current image using its parent folder name
        parentFolderName = os.path.split(os.path.dirname(imgFileName))[1]
        # expecting folder name as label name in form labelNo_labelText
        # e.g. for label names e.g. 0_NotGrass, 1_Grass, etc
        labelNo, labelText = parentFolderName.split("_") 
        if not labels.has_key(labelNo):
            labels[labelNo] = labelText
        labelData.extend([labelNo for index in range(len(imgBlockStack))])

    labelData = np.array(labelData, dtype=np.float)
    log.debug('trainData.shape {}'.format(trainingData.shape))
    return (trainingData, labelData, labels)

def TrainNSaveSVM(TrainingData, LabelData, SvmDataFileName='svm_data.dat'):
    ''' Trains and saves SVM into (.dat) file. Pass None in Filename to avoid saving.'''
    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                        svm_type = cv2.SVM_C_SVC)
    svm = cv2.SVM()
    svm.train(TrainingData,LabelData,params=svm_params)
    if SvmDataFileName: # if None, do not save
        svm.save(SvmDataFileName) 

def LoadNDetectSVM(SvmDataFileName, FolderPath):
    ''' Prepare data from Folderpath, use SVM from SvmDataFileName for detection
Returns actual and detected values for trained classes'''
    samples, labelValues, labels = prepareTrainingDataFromImageStrucuture(
        FolderPath)
    svm = cv2.SVM()
    log.debug("loading SVM .DAT file '{}'...".format(SvmDataFileName))
    svm.load(SvmDataFileName) 
    log.debug('loaded SVM .DAT file successfully.')
    detectedValues = svm.predict_all(np.float32(samples)) 
    # ensure 1-D row matrix
    detectedValues.shape = 1, detectedValues.size # ensure row matrix
    labelValues.shape = 1, labelValues.size # ensure row matrix
    log.debug('labelValues{}: {}\n detectedValues{}: {}'
              .format(labelValues.shape, labelValues,
                      detectedValues.shape, detectedValues))
    return labelValues, detectedValues
