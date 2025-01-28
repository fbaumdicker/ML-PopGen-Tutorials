#!/bin/bash

mkdir -p testSet
diploSHIC makeTrainingSets testFvecs/neut.fvec testFvecs/soft_ testFvecs/hard_ 5 0,1,2,3,4,6,7,8,9,10 testSet/
python testClassifierAndPlotConfusionMatrix.py clf testSet preds.txt covfefe.pdf
