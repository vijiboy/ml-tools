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
    def test_CreateImageStructureFromFolders(self):
        inputPath = os.path.join(os.curdir, 'input', 'train', 'grass')
        imageStructure = Imaging.ImageStructureCreateFromFolder(inputPath)
        imageStructure = Imaging.ImageStructureApplyFunc(imageStructure, Imaging.loadImageFromFile, UseKey = True)
        imageStructure = Imaging.ImageStructureApplyFunc(imageStructure, np.shape)
        self.assertGreaterEqual(len(Imaging.getFlattenedStructure(imageStructure)), 1)
        self.assertTrue(all(Imaging.getFlattenedStructure(imageStructure)))

    def test_LoadImageArrayIntoImageStructure(self):
        inputPath = os.path.join(os.path.abspath(os.curdir), 'input', 'train', 'grass')
        grassPath = os.path.join(inputPath, '1_Grass')
        trainingData = Training.getTrainingDataFromFolder(grassPath)
        self.assertEqual((1221,900),trainingData.shape)
        

            
                


            
        
        
