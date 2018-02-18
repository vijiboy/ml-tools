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
        self.assertIsNotNone(trainingData)
        self.assertIsNotNone(labelData)
        self.assertIsNotNone(labels)
        
    def test_TrainingSavesSVMFile(self):
        trainFolder = os.path.join(os.path.abspath(os.curdir), 'input', 'train', 'grass')
        trainingData, labelData, labels = Training.prepareTrainingDataFromImageStrucuture(trainFolder)
        svmDatFile = 'svm_data.dat'
        if os.path.isfile(svmDatFile): os.remove(svmDatFile)
        self.assertFalse(os.path.isfile(svmDatFile))
        Training.TrainNSaveSVM(np.float32(trainingData), np.float32(labelData))
        self.assertTrue(os.path.isfile(svmDatFile))
        
    def test_TrainDetectUsingSVM(self):
        trainFolder = os.path.join(os.path.abspath(os.curdir), 'input', 'train', 'grass')
        trainingData, labelData, labels = Training.prepareTrainingDataFromImageStrucuture(trainFolder)
        Training.TrainNSaveSVM(np.float32(trainingData), np.float32(labelData), 'svm_test.dat')
        expected, actual = Training.LoadNDetectSVM('svm_test.dat',trainFolder)
        log.info('expected labels{}:\n{}, \nactual labels{}:\n{}'.format(expected.shape, expected, actual.shape, actual))
        mask = expected == actual
        correct = np.count_nonzero(mask)
        log.info('Correct Results {} of Total {}.'.format(correct, actual.size))
        accuracy = round(correct * 100.0 / actual.size)
        log.info('detection accuracy: {}%'.format(accuracy))
        self.assertGreaterEqual(accuracy, 95.0) # achieve accuracy
        # increase accuracy include color i.e. R,G,B.
        # 10101010 training grass.Nograss.grass.Nograss.grass

