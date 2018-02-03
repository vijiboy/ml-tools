import unittest
import logging as log
import cv2
import numpy as np

import os

log.basicConfig() # by default only log messages for error and critical conditions
log.getLogger().setLevel(log.INFO) # uncomment for more verbose log messages
#log.getLogger().setLevel(log.DEBUG) # uncomment for even more verbose log messages

import ImageSplitting as Imaging
import Labelling 

class Test_Labelling(unittest.TestCase):
    def test_MarkingImagePixelsUsesNumpyBinaryMask_AdvancedIndexing(self):
        # Create an 'RGB' image of size 3X3
        imgArray = np.arange(0,48,1)
        imgArray.shape = (4,4,3) 
        # binary mask to mark individual pixels
        binaryMask = np.ndarray(imgArray.shape[:2], dtype = np.bool) 
        binaryMask[:] = False
        self.assertTrue((binaryMask==False).all())
        binaryMask[2] = True # mark few pixels using binary index
        # use binary advanced indexing
        self.assertTrue((imgArray[binaryMask].flat==imgArray[2].flat).all()) 

    def test_BinaryMaskDecidesIfImageBlockIsSelectedOrNot(self):
        image = Imaging.loadImageFromFile('test/EquallySplitting_Image.png')
        image = Imaging.loadImageFromFile('test/MaskImageAllGreen.png')
        binaryMask = np.ndarray(image.shape[:2], dtype = np.bool) 
        # Enable Complete Mask (False). masking all pixels, marking none.
        binaryMask[:] = False 
        self.assertFalse(Labelling.IsImageBlockSatisfyingSelectionPercentage
                         (image, SelectionMask=binaryMask))
        # Disable mask for less than 10% i.e. 9% of the image pixels
        binaryMask[:9,:4] = True  # 36 = 9% of 400
        self.assertEqual(np.count_nonzero(binaryMask.flat),36) 
        self.assertFalse(Labelling.IsImageBlockSatisfyingSelectionPercentage
                         (image, SelectionMask=binaryMask))
        # Disable Complete Mask (True). unmasking all pixels, marking all.
        binaryMask[:] = True 
        self.assertTrue(Labelling.IsImageBlockSatisfyingSelectionPercentage
                        (image, SelectionMask=binaryMask))
        # Enable 10% of the mask
        binaryMask[:] = False 
        binaryMask[:10,:4] = True # 40 = 10% of 400 
        self.assertTrue(Labelling.IsImageBlockSatisfyingSelectionPercentage
                        (image, SelectionMask=binaryMask))

    def test_UtilityToGetBinaryMaskFromImage(self):
        maskImageGreen = Imaging.loadImageFromFile('test/MaskImageAllGreen.png')

        hexColorGreen = Labelling.rgb_to_hex((0,255,0))
        greenLabel = Labelling.getBinaryMaskFromColorCodedImage(maskImageGreen, hexColorGreen)
        self.assertEqual(greenLabel.shape, maskImageGreen.shape[:2])
        self.assertTrue((greenLabel==True).all())

        maskImageGreen10 = Imaging.loadImageFromFile('test/MaskImageGreen10.png')
        greenLabel10= Labelling.getBinaryMaskFromColorCodedImage(maskImageGreen10, hexColorGreen)
        self.assertTrue(Labelling.IsImageBlockSatisfyingSelectionPercentage(maskImageGreen10, greenLabel10, False, SelectionPercentage = 10))

        blueLabel = Labelling.getBinaryMaskFromColorCodedImage(maskImageGreen10, '#0000ff')
        self.assertTrue((blueLabel==False).all())

    def test_ImageSplitting_Using_BinaryMaskAsLabel(self):
        inputImage = Imaging.loadImageFromFile('test/ObjectInTopLeft.png')
        maskImageTopLeft = Imaging.loadImageFromFile('test/MaskImageGreen10.png')
        greenLabel = Labelling.getBinaryMaskFromColorCodedImage(maskImageTopLeft)
        self.assertEqual(inputImage.shape[:2], greenLabel.shape) # ensure both image and mask are same width and height

        # get blocks: split both image and block 
        splittingBlock = Imaging.SplittingBlock(blockWidth=10, blockHeight=10,
                                             OverlapHorizontal=0, OverlapVertical=0)
        imageBlocksStructure = Imaging.SplitImageinBlocksByShifting(inputImage, splittingBlock)
        maskBlocksStructure = Imaging.SplitImageinBlocksByShifting(greenLabel, splittingBlock)
        imageBlocks = Imaging.iterateImageBlocks(imageBlocksStructure)
        maskBlocks = Imaging.iterateImageBlocks(maskBlocksStructure)

        selectedBlocks = []
        for iBlock in Labelling.iterateImageBlocksBasedOnMask(imageBlocks,
                                                         maskBlocks):
            selectedBlocks.append(iBlock)

        # there should be only one selected block here
        self.assertEqual(len(selectedBlocks), 1) 

    def test_ImageSplittingWithOverlaps_WithBinaryMaskLabelling(self):
        inputImage = Imaging.loadImageFromFile('test/ObjectInTopLeft.png')
        maskImageTopLeft = Imaging.loadImageFromFile('test/MaskImageGreen10.png')
        greenLabel = Labelling.getBinaryMaskFromColorCodedImage(maskImageTopLeft)

        # get blocks: split both image and block with overlaps
        splittingBlock = Imaging.SplittingBlock(blockWidth=10, blockHeight=10,
                                             OverlapHorizontal=2, OverlapVertical=2)
        imageBlocksStructure = Imaging.SplitImageinBlocksByShifting(inputImage, splittingBlock)
        maskBlocksStructure = Imaging.SplitImageinBlocksByShifting(greenLabel, splittingBlock)
        imageBlocks = Imaging.iterateImageBlocks(imageBlocksStructure)
        maskBlocks = Imaging.iterateImageBlocks(maskBlocksStructure)

        # check image reconstruction from overlapped blocks
        blankOriginalImage = np.zeros_like(inputImage)
        recreatedImage = Imaging.CreateImageFromBlocks(imageBlocksStructure,
                                                    blankOriginalImage, splittingBlock)
        self.assertTrue(np.array_equal(inputImage, recreatedImage))

        # get selected blocks
        selectedBlocks = []
        for iBlock in Labelling.iterateImageBlocksBasedOnMask(imageBlocks,
                                                         maskBlocks):
            selectedBlocks.append(iBlock)
        self.assertEqual(len(selectedBlocks), 4)
