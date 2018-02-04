# To test creation of data used for subsequent training 

import unittest
import logging as log
import cv2
import numpy as np

import os

log.basicConfig() # by default only log messages for error and critical conditions
log.getLogger().setLevel(log.INFO) # uncomment for more verbose log messages
#log.getLogger().setLevel(log.DEBUG) # uncomment for even more verbose log messages

import ImageSplitting as Imaging
import Training


class Test_Training(unittest.TestCase):
    def test_CreateImageStructureFromFolders_AndApplyFunctionsOnThem(self):
        inputPath = os.path.join(os.path.abspath(os.curdir), 'input', 'train', 'grass')
        # load file names only without loading actual images
        imageStructure = Imaging.ImageStructureCreateFromFolder(inputPath)
        self.assertTrue(v is None for k,v in
                        Imaging.getFlattenedStructure(imageStructure))
        # load images by applying load image function to all image file names 
        imageStructure = Imaging.ImageStructureApplyFunc(imageStructure, Imaging.loadImageFromFile, UseKey = True)
        # use flat structure to iterate all images when required
        self.assertTrue(all(Imaging.getFlattenedStructure(imageStructure)))

    def test_PrepareTrainingDataUsingImageStructure(self):
        inputPath = os.path.join(os.path.abspath(os.curdir), 'input', 'train', 'grass')
        trainingData, labelData, labels = Training.prepareTrainingDataFromImageStrucuture(inputPath)
        self.assertEqual((2779,900),trainingData.shape)
        log.info('labelData {}: {}'.format(labelData.shape,labelData))
        log.info('labels: {}'.format(labels))
        
    def test_TrainTheSVM(self):
        inputPath = os.path.join(os.path.abspath(os.curdir), 'input', 'train', 'grass')
        trainingData, labelData, labels = Training.prepareTrainingDataFromImageStrucuture(inputPath)
        log.info('trainingData {1} :\n {0}, datatype: {2}'.format(trainingData, trainingData.shape, trainingData.dtype))
        Training.TrainNSaveSVM(trainingData, labelData)
        pass
        
