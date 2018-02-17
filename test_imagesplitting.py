import unittest
import logging as log
import cv2
import numpy as np

import os

log.basicConfig() # by default only log messages for error and critical conditions
log.getLogger().setLevel(log.INFO) # uncomment for more verbose log messages
#log.getLogger().setLevel(log.DEBUG) # uncomment for even more verbose log messages

import ImageSplitting as data

class Test_ImageSplitting(unittest.TestCase):
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

    def test_CreatesImageStructureFromAFolderHierarchy(self):
        testPath = os.path.join(os.curdir, 'myTest')
        # create folder 'myTest' with sub-folders and files for test purpose
        map(data.touch,
            [os.path.join(testPath,testfile) for testfile in
             ['f1/11.jpg', 'f1/12.jpg', 'f1/a/a1.jpg',
              'f2/21.jpg', 'f2/22.jpg', 'f2/b/b1.jpg']])
        self.assertTrue(os.path.exists(testPath))
        log.info('testPath ' + testPath)
        imgStrucuture_relativePaths = data.ImageStructureCreateFromFolder(testPath, relativepaths=True)
        dict_relativePaths = {'f1': {'11.jpg': None, '12.jpg': None, 'a': {'a1.jpg': None}},
         'f2': {'21.jpg': None, '22.jpg': None, 'b': {'b1.jpg': None}}}
        self.assertDictEqual(imgStrucuture_relativePaths, dict_relativePaths)

        imgStrucuture_absolutePaths = data.ImageStructureCreateFromFolder(testPath, relativepaths=False)
        dict_absolutePaths = {'./myTest/f1': {'./myTest/f1/11.jpg': None,'./myTest/f1/12.jpg': None,
                         './myTest/f1/a': {'./myTest/f1/a/a1.jpg': None}},
         './myTest/f2': {'./myTest/f2/21.jpg': None, './myTest/f2/22.jpg': None,
                         './myTest/f2/b': {'./myTest/f2/b/b1.jpg': None}}}
        self.assertDictEqual( imgStrucuture_absolutePaths, dict_absolutePaths)
        




