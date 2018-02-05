
test_ImageBlocksStoreWidthHeightAndOverlap (test_imagesplitting.Test_ImageSplitting) ... skipped 'TODO: Design: group imageBlocks and Block(Width/Height), (Horizontal/Vertical) Shift'
test_SplitsImageInBlocks_PerfectSplittingImage (test_imagesplitting.Test_ImageSplitting) ... ok
test_SplitsImageInBlocks_UnequallySplittingImage (test_imagesplitting.Test_ImageSplitting) ... ok
test_SplitsImageInBlocks_WithOverlappingBlocks (test_imagesplitting.Test_ImageSplitting) ... ok
test_BinaryMaskDecidesIfImageBlockIsSelectedOrNot (test_labelling.Test_Labelling) ... ok
test_ImageSplittingWithOverlaps_WithBinaryMaskLabelling (test_labelling.Test_Labelling) ... ok
test_ImageSplitting_Using_BinaryMaskAsLabel (test_labelling.Test_Labelling) ... ok
test_MarkingImagePixelsUsesNumpyBinaryMask_AdvancedIndexing (test_labelling.Test_Labelling) ... ok
test_UtilityToGetBinaryMaskFromImage (test_labelling.Test_Labelling) ... ok
test_CreateImageStructureFromFolders_AndApplyFunctionsOnThem (test_training.Test_Training) ... ok
test_PrepareTrainingDataUsingImageStructure (test_training.Test_Training) ... INFO:root:labelData (2779,): [ 1.  1.  1. ...,  0.  0.  0.]
INFO:root:labels: {'1': 'Grass', '0': 'NotGrass'}
ok
test_TrainTheSVM (test_training.Test_Training) ... INFO:root:trainingData (2779, 900) :
 [[ 136.   79.   64. ...,   14.   70.   98.]
 [  33.   53.  137. ...,   37.   38.   74.]
 [  64.   55.   71. ...,  133.  129.  187.]
 ..., 
 [  92.   92.   93. ...,  128.  128.  128.]
 [ 103.  103.  104. ...,    0.    0.    0.]
 [ 113.  112.  109. ...,    0.    0.    0.]], datatype: float64
ok

----------------------------------------------------------------------
Ran 12 tests in 2.638s

OK (skipped=1)
