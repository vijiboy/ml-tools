import cv2
import numpy as np
import logging as log
import os

from collections import namedtuple
SplittingBlock = namedtuple('SplittingBlock','blockWidth, blockHeight, OverlapHorizontal, OverlapVertical')

def SplitImageinBlocksByShifting(Image, SplitBlock):
    ''' Splits an image into blocks of size BlockWidth X BlockHeight pixels, 
blocks are created by traversing first from left to right and then top to bottom '''
    if not SplitBlock  or type(SplitBlock) != SplittingBlock:
        raise ValueError("Error: Invalid Argument - SplitBlock")
    BlockWidth, BlockHeight, OverlapHorizontal, OverlapVertical = SplitBlock
    if OverlapHorizontal >= BlockWidth or OverlapVertical >= BlockHeight:
        raise ValueError("Invalid Argument: Vertical and/or Horizontal Overlap Values." 
                         " Ensure Overlaps are more than Block Height/Width ")
    imageBlocks = []
    imageHeight = Image.shape[0]
    imageWidth = Image.shape[1]
    row = 0
    while row < imageHeight:
        col = 0 
        imageBlocksInRow = []
        while col < imageWidth:
            imageBlock = Image[row:row+BlockHeight, col:col+BlockWidth]
            log.debug('Image Block: At Image Position(x,y):({}, {}); Shape:{}.'.
                      format(row,col, imageBlock.shape))
            imageBlocksInRow.append(imageBlock)
            col = col + BlockWidth - OverlapHorizontal
        imageBlocks.append(imageBlocksInRow)
        row = row + BlockHeight - OverlapVertical
    return imageBlocks

def CreateImageFromBlocks(ImageBlocks, BlankImage, SplitBlock):
    ''' Utility to recreate image inside BlankImage from ImageBlocks by joining them. 
 Note: BlankImage must be sufficiently large enough to hold ImageBlocks'''
    BlockWidth, BlockHeight, OverlapHorizontal, OverlapVertical = SplitBlock
    row = 0
    for imageBlocksInRow in ImageBlocks:
        col = 0
        for thisImageBlock in imageBlocksInRow:
            log.debug('copying imageBlock:{0} to BlankImage:{1} at row,col:{2},{3}'.
                     format(thisImageBlock.shape, BlankImage.shape, row,col))
            BlankImage[row:row+BlockHeight, col:col+BlockWidth] = thisImageBlock
            col = col + BlockWidth - OverlapHorizontal
        row = row + BlockHeight - OverlapVertical
    return BlankImage
    
def iterateImageBlocks(ImageBlocks):
    '''Iterator returning next image block from the sequence created by 
SplitImageinBlocksByShifting() in order from topleft block to bottomright'''
    if ImageBlocks is None: raise ValueError("argument empty: ImageBlocks")
    linearBlocks = []
    for row, imageBlocksInRow in enumerate(ImageBlocks):
        for col, imageBlock in enumerate(imageBlocksInRow):
            log.debug('Image Block: At Sequence Location:({}, {}); Shape:{}.'.
                      format(row,col,imageBlock.shape))
            yield imageBlock

def loadImageFromFile(Filename):
    image =  cv2.imread(Filename) # load as is
    if image is None:
        raise ValueError('Invalid imagePath: image not found "{}"'.format(Filename))
    return image

def ImageStructureCreateFromFolder(FolderPath, relativepaths=False):
    '''Creates ImageStructure, a dictionary matching the structure of FolderPath containing sub-folders as dictionaries and images as key/value pairs '''
    if not os.path.isdir(FolderPath):
        raise ValueError("Invalid Argument: FolderPath - '{}'."
                         " Path invalid or not a folder.".format(FolderPath))
    imageStructure = dict()
    folderDict = imageStructure
    for root, dirs, files in os.walk(FolderPath, topdown=True):
        folderDict[root] = dict()
        for fileName in files:
            keyName = fileName if relativepaths else os.path.join(root,fileName)
            folderDict[root][keyName] = None
        folderDict = folderDict[root]
    log.debug('ImageStructure: {}'.format(imageStructure))
    return imageStructure

def ImageStructureApplyFunc(ImageStructure, ImageProcessingFunc, UseKey=False):
    ''' Applies a function to each image in the ImageStructure inplace '''
    for key,value in ImageStructure.viewitems(): # could just use keys
        if type(ImageStructure[key]) is not dict:
            input = key if UseKey else value
            newValue = ImageProcessingFunc(input)
            ImageStructure[key] = newValue
            log.debug('Applied ImageProcessingFunc() to {} new value= {}.'.format(input, newValue))
        else:
            ImageStructureApplyFunc(ImageStructure[key], ImageProcessingFunc, UseKey)
    return ImageStructure

def getFlattenedStructure(ImageStructure, flatStructure=None):
    ''' creates or updates new flat structure of images as key/value pairs 
where key is image path and value is image array '''
    if flatStructure is None: flatStructure = dict()
    elif type(flatStructure) is not dict:
        raise ValueError('Invalid Argument: flatStructure. Should be dict')

    for key,value in ImageStructure.viewitems(): # could just use keys
        if type(ImageStructure[key]) is not dict:
            flatStructure[key] = ImageStructure[key]
        else:
            getFlattenedStructure(ImageStructure[key], flatStructure)

    log.debug(flatStructure)
    return flatStructure

    
