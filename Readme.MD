
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
OpenCV Error: Bad argument (train data must be floating-point matrix) in cvCheckTrainData, file /home/travis/miniconda/conda-bld/work/opencv-2.4.11/modules/ml/src/inner_functions.cpp, line 857
ERROR

======================================================================
ERROR: test_TrainTheSVM (test_training.Test_Training)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test_training.py", line 41, in test_TrainTheSVM
    Training.TrainNSaveSVM(trainingData, labelData)
  File "Training.py", line 60, in TrainNSaveSVM
    svm.train(TrainingData,LabelData,params=svm_params)
error: /home/travis/miniconda/conda-bld/work/opencv-2.4.11/modules/ml/src/inner_functions.cpp:857: error: (-5) train data must be floating-point matrix in function cvCheckTrainData


----------------------------------------------------------------------
Ran 12 tests in 1.029s

FAILED (errors=1, skipped=1)
