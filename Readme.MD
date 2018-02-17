test_CreatesImageStructureFromAFolderHierarchy (test_imagesplitting.Test_ImageSplitting) ... INFO:root:testPath ./myTest
ok
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
test_DetectUsingSVM (test_training.Test_Training) ... INFO:root:loading SVM .DAT file 'svm_test.dat'...
INFO:root:loaded SVM .DAT file successfully.
INFO:root:labelValues(1, 2393): [[ 0.  0.  0. ...,  1.  1.  1.]]
 detectedValues(1, 2393): [[ 0.  0.  0. ...,  1.  1.  1.]]
INFO:root:Correct Results 1383 of Total 2393.
INFO:root:detection accuracy: 57.7935645633
FAIL
test_PrepareTrainingDataUsingImageStructure (test_training.Test_Training) ... INFO:root:labelData (2393,): [ 0.  0.  0. ...,  1.  1.  1.]
INFO:root:labels: {'1': 'Grass', '0': 'NotGrass'}
ok
test_TrainTheSVM (test_training.Test_Training) ... ok

======================================================================
FAIL: test_DetectUsingSVM (test_training.Test_Training)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test_training.py", line 56, in test_DetectUsingSVM
    self.assertGreaterEqual(accuracy, 95.0) # achieve accuracy
AssertionError: 57.79356456330965 not greater than or equal to 95.0

----------------------------------------------------------------------
Ran 14 tests in 4.412s

FAILED (failures=1, skipped=1)
