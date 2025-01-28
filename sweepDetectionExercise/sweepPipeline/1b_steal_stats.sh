#!/bin/bash

wc trainingFvecs/* testFvecs/*

cp -r ../preCookedData/trainingFvecs .
cp -r ../preCookedData/testFvecs .

wc trainingFvecs/* testFvecs/*
