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
        splittingBlock = data.SplittingBlock(blockWidth=10, blockHeight=10,
                                             OverlapHorizontal=0, OverlapVertical=0)
        imageBlocks = data.SplitImageinBlocksByShifting(original_image, splittingBlock)
        totalBlocks = sum(1 for block in data.iterateImageBlocks(imageBlocks))
        self.assertEqual(totalBlocks, 4)
        # assert split operation by joining image blocks and comparing to original image
        blankOriginalImage = np.zeros_like(original_image)
        recreatedImage = data.CreateImageFromBlocks(imageBlocks, blankOriginalImage,
                                                    splittingBlock)
        self.assertTrue((recreatedImage == original_image).all())

    def test_SplitsImageInBlocks_UnequallySplittingImage(self):
        # split image (filepath) into blocks
        original_image = data.loadImageFromFile('test/UnequallySplitting_Image.png')
        splittingBlock = data.SplittingBlock(blockWidth=10, blockHeight=10,
                                             OverlapHorizontal=0, OverlapVertical=0)
        imageBlocks = data.SplitImageinBlocksByShifting(original_image, splittingBlock)
        totalBlocks = sum(1 for block in data.iterateImageBlocks(imageBlocks))
        self.assertEqual(totalBlocks, 9)
        # assert split operation by joining image blocks and comparing to original image
        blankOriginalImage = np.zeros_like(original_image)
        recreatedImage = data.CreateImageFromBlocks(imageBlocks, blankOriginalImage,
                                                    splittingBlock)
        self.assertTrue((recreatedImage == original_image).all())

    def test_SplitsImageInBlocks_WithOverlappingBlocks(self):
        original_image = data.loadImageFromFile('test/EquallySplitting_Image.png')
        splittingBlock = data.SplittingBlock(blockWidth=10, blockHeight=10,
                                             OverlapHorizontal=1, OverlapVertical=1)
        imageBlocks = data.SplitImageinBlocksByShifting(original_image, splittingBlock)
        totalBlocks = sum(1 for block in data.iterateImageBlocks(imageBlocks))
        self.assertEqual (totalBlocks, 9)
        # assert split operation by joining image blocks and comparing to original image
        blankOriginalImage = np.zeros_like(original_image)
        recreatedImage = data.CreateImageFromBlocks(imageBlocks, blankOriginalImage,
                                                    splittingBlock)
        self.assertTrue((recreatedImage == original_image).all())

    @unittest.skip("TODO: Design: group imageBlocks and Block(Width/Height), (Horizontal/Vertical) Shift")
    def test_ImageBlocksStoreWidthHeightAndOverlap(self):
        pass


class TestTrainingData_Labelling(unittest.TestCase):
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
        image = data.loadImageFromFile('test/EquallySplitting_Image.png')
        image = data.loadImageFromFile('test/MaskImageAllGreen.png')
        binaryMask = np.ndarray(image.shape[:2], dtype = np.bool) 
        # Enable Complete Mask (False). masking all pixels, marking none.
        binaryMask[:] = False 
        self.assertFalse(data.IsImageBlockSatisfyingSelectionPercentage
                         (image, SelectionMask=binaryMask))
        # Disable mask for less than 10% i.e. 9% of the image pixels
        binaryMask[:9,:4] = True  # 36 = 9% of 400
        self.assertEqual(np.count_nonzero(binaryMask.flat),36) 
        self.assertFalse(data.IsImageBlockSatisfyingSelectionPercentage
                         (image, SelectionMask=binaryMask))
        # Disable Complete Mask (True). unmasking all pixels, marking all.
        binaryMask[:] = True 
        self.assertTrue(data.IsImageBlockSatisfyingSelectionPercentage
                        (image, SelectionMask=binaryMask))
        # Enable 10% of the mask
        binaryMask[:] = False 
        binaryMask[:10,:4] = True # 40 = 10% of 400 
        self.assertTrue(data.IsImageBlockSatisfyingSelectionPercentage
                        (image, SelectionMask=binaryMask))

    def test_UtilityToGetBinaryMaskFromImage(self):
        maskImageGreen = data.loadImageFromFile('test/MaskImageAllGreen.png')

        hexColorGreen = data.rgb_to_hex((0,255,0))
        greenLabel = data.getBinaryMaskFromColorCodedImage(maskImageGreen, hexColorGreen)
        self.assertEqual(greenLabel.shape, maskImageGreen.shape[:2])
        self.assertTrue((greenLabel==True).all())

        maskImageGreen10 = data.loadImageFromFile('test/MaskImageGreen10.png')
        greenLabel10= data.getBinaryMaskFromColorCodedImage(maskImageGreen10, hexColorGreen)
        self.assertTrue(data.IsImageBlockSatisfyingSelectionPercentage(maskImageGreen10, greenLabel10, False, SelectionPercentage = 10))

        blueLabel = data.getBinaryMaskFromColorCodedImage(maskImageGreen10, '#0000ff')
        self.assertTrue((blueLabel==False).all())

    def test_ImageSplitting_Using_BinaryMaskAsLabel(self):
        inputImage = data.loadImageFromFile('test/ObjectInTopLeft.png')
        maskImageTopLeft = data.loadImageFromFile('test/MaskImageGreen10.png')
        greenLabel = data.getBinaryMaskFromColorCodedImage(maskImageTopLeft)
        self.assertEqual(inputImage.shape[:2], greenLabel.shape) # ensure both image and mask are same width and height

        # get blocks: split both image and block 
        splittingBlock = data.SplittingBlock(blockWidth=10, blockHeight=10,
                                             OverlapHorizontal=0, OverlapVertical=0)
        imageBlocksStructure = data.SplitImageinBlocksByShifting(inputImage, splittingBlock)
        maskBlocksStructure = data.SplitImageinBlocksByShifting(greenLabel, splittingBlock)
        imageBlocks = data.iterateImageBlocks(imageBlocksStructure)
        maskBlocks = data.iterateImageBlocks(maskBlocksStructure)

        selectedBlocks = []
        for iBlock in data.iterateImageBlocksBasedOnMask(imageBlocks,
                                                         maskBlocks):
            selectedBlocks.append(iBlock)

        # there should be only one selected block here
        self.assertEqual(len(selectedBlocks), 1) 

    def test_ImageSplittingWithOverlaps_WithBinaryMaskLabelling(self):
        inputImage = data.loadImageFromFile('test/ObjectInTopLeft.png')
        maskImageTopLeft = data.loadImageFromFile('test/MaskImageGreen10.png')
        greenLabel = data.getBinaryMaskFromColorCodedImage(maskImageTopLeft)

        # get blocks: split both image and block with overlaps
        splittingBlock = data.SplittingBlock(blockWidth=10, blockHeight=10,
                                             OverlapHorizontal=2, OverlapVertical=2)
        imageBlocksStructure = data.SplitImageinBlocksByShifting(inputImage, splittingBlock)
        maskBlocksStructure = data.SplitImageinBlocksByShifting(greenLabel, splittingBlock)
        imageBlocks = data.iterateImageBlocks(imageBlocksStructure)
        maskBlocks = data.iterateImageBlocks(maskBlocksStructure)

        # check image reconstruction from overlapped blocks
        blankOriginalImage = np.zeros_like(inputImage)
        recreatedImage = data.CreateImageFromBlocks(imageBlocksStructure,
                                                    blankOriginalImage, splittingBlock)
        self.assertTrue(np.array_equal(inputImage, recreatedImage))

        # get selected blocks
        selectedBlocks = []
        for iBlock in data.iterateImageBlocksBasedOnMask(imageBlocks,
                                                         maskBlocks):
            selectedBlocks.append(iBlock)
        self.assertEqual(len(selectedBlocks), 4)

