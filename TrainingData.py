import cv2
import numpy as np
import logging as log

def SplitImageinBlocksByShifting(Image,
                                 BlockWidth=10, BlockHeight=10, 
                                 OverlapHorizontal=0, OverlapVertical=0):
    ''' Splits an image into blocks of size BlockWidth X BlockHeight pixels, 
blocks are created by traversing first from left to right and then top to bottom,
shifting by HorizontalShift pixels and VerticalShift pixels respectively '''
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

def IsImageBlockSatisfyingSelectionPercentage(
        Image, SelectionMask, InvertMask=False, SelectionPercentage = 10):
    ''' Returns True if SelectionMask (a binary mask) is True for more than 10% 
    of total Image pixels, else returns False. 
    Setting InvertMask=True selects Image pixels where SelectionMask is False ''' 
    #image assumed RGB / 3-plane and Mask assumed Gray Scale i.e. single plane
    if SelectionMask.shape != Image.shape[:2]: 
        raise ValueError("Invalid Argument: SelectionMask shape/size {} should be same as Image {}"
                         .format(SelectionMask.shape, Image.shape[:2]))
    TotalPixels = reduce(lambda x,y: x*y, Image.shape[:2]) # Find Image Width*Height
    AllowedPixels = SelectedPixels = np.count_nonzero(SelectionMask)
    if InvertMask : AllowedPixels = TotalPixels - SelectedPixels
    AllowedPixelsPercent = int(round(AllowedPixels * 100.0 / TotalPixels))
    return AllowedPixelsPercent >= SelectionPercentage 

def CreateImageFromBlocks(ImageBlocks, BlankImage, BlockWidth=10, BlockHeight=10,
                          OverlapHorizontal=0, OverlapVertical=0):
    ''' Utility to recreate image inside BlankImage from ImageBlocks by joining them. 
 Note: BlankImage must be sufficiently large enough to hold ImageBlocks'''
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

def getBinaryMaskFromColorCodedImage(maskImage, hexColor='#00ff00'):
    ''' gets binary Mask from an rgb image using hexColor (default green) '''
    if len(maskImage.shape) != 3: raise ValueError('Expected rgb numpy array')
    imgRows,imgCols,imgChannels = maskImage.shape;
    rgbPixelsDiff = (maskImage == np.asarray(hex_to_rgb(hexColor)))
    rgbPixelsDiff.shape = (imgRows*imgCols, imgChannels) # flatten for easy iteration & query
    diffListTrueAll = [True if ((rgbDiff == True).all()) else False for rgbDiff in rgbPixelsDiff]
    maskTrueAll = np.asarray(diffListTrueAll)
    maskTrueAll.shape = imgRows, imgCols # binary mask is like a gray image with binary values
    return maskTrueAll

def hex_to_rgb(value):
    ''' convert color in hexadecimal triplet back to its rgb triplet '''
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_hex(rgb):
    ''' convert an rgb color (RGB triplet) to hexadecimal format (hex triplet) string '''
    return '#%02x%02x%02x' % rgb

def iterateImageBlocksBasedOnMask(imageBlocksIterator, maskBlocksIterator, selectionPercentage=10):
    ''' iterate image blocks selected by mask blocks satisfying selection percentage '''
    for iBlock, mBlock in zip(imageBlocksIterator, maskBlocksIterator):
        if IsImageBlockSatisfyingSelectionPercentage(iBlock, mBlock, False, selectionPercentage):
            IsSelected = True
        else: IsSelected = False
        
        log.debug('Image Block ({}); maskBlock ({}); Selected: {}'.
                    format(iBlock.shape, mBlock.shape, IsSelected))

        if IsSelected:
            yield iBlock


