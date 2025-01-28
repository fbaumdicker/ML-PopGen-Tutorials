#!/bin/bash

# first create a directory to stash our training set
mkdir -p trainingSet

# step 1: build our training set
diploSHIC makeTrainingSets trainingFvecs/neut.fvec trainingFvecs/soft_ trainingFvecs/hard_ 5 0,1,2,3,4,6,7,8,9,10 trainingSet/

# step 2: train our classifier
diploSHIC train trainingSet/ trainingSet/ clf
