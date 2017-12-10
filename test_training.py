# To test creation of data used for subsequent training 

import unittest
import logging as log
import cv2
import numpy as np

log.basicConfig() # by default only log messages for error and critical conditions
#log.getLogger().setLevel(log.INFO) # uncomment for more verbose log messages
#log.getLogger().setLevel(log.DEBUG) # uncomment for even more verbose log messages

import TrainingData as data

class TestTrainingData_Creation(unittest.TestCase):

    def test_SplitsImageInBlocks_PerfectSplittingImage(self):
        # split image (filepath) into blocks
        original_image = data.loadImageFromFile('test/EquallySplitting_Image.png')
        imageBlocks = data.SplitImageinBlocksByShifting(original_image)
        totalBlocks = sum(1 for block in data.ImageBlocksIterator(imageBlocks))
        self.assertEqual(totalBlocks, 4)
        # assert split operation by joining image blocks and comparing to original image
        blankOriginalImage = np.zeros_like(original_image)
        recreatedImage = data.CreateImageFromBlocks(imageBlocks, blankOriginalImage)
        self.assertTrue((recreatedImage == original_image).all())

    def test_SplitsImageInBlocks_UnequallySplittingImage(self):
        # split image (filepath) into blocks
        original_image = data.loadImageFromFile('test/UnequallySplitting_Image.png')
        imageBlocks = data.SplitImageinBlocksByShifting(original_image)
        totalBlocks = sum(1 for block in data.ImageBlocksIterator(imageBlocks))
        self.assertEqual(totalBlocks, 9)
        # assert split operation by joining image blocks and comparing to original image
        blankOriginalImage = np.zeros_like(original_image)
        recreatedImage = data.CreateImageFromBlocks(imageBlocks, blankOriginalImage)
        self.assertTrue((recreatedImage == original_image).all())

    def test_SplitsImageInBlocks_WithOverlappingBlocks(self):
        original_image = data.loadImageFromFile('test/EquallySplitting_Image.png')
        imageBlocks = data.SplitImageinBlocksByShifting(original_image,
                                      BlockWidth=10, BlockHeight=10,
                                      OverlapHorizontal=1, OverlapVertical=1)
        totalBlocks = sum(1 for block in data.ImageBlocksIterator(imageBlocks))
        self.assertEqual (totalBlocks, 9)
        # assert split operation by joining image blocks and comparing to original image
        blankOriginalImage = np.zeros_like(original_image)
        recreatedImage = data.CreateImageFromBlocks(imageBlocks, blankOriginalImage, OverlapHorizontal=1, OverlapVertical=1)
        self.assertTrue((recreatedImage == original_image).all())

    @unittest.skip("TODO: Design: group imageBlocks and Block(Width/Height), (Horizontal/Vertical) Shift")
    def test_ImageBlocksStoreWidthHeightAndOverlap(self):
        self.assertTrue(False)

    

class TestTrainingData_Labelling(unittest.TestCase):

    @unittest.skip("TODO: Next")
    def test_LabelsAnImageUsingMasking(self):
        input_image = data.loadImageFromFile('test/EquallySplitting_Image.png')
        imageBlocks = data.SplitImageinBlocksByShifting(original_image)
        
