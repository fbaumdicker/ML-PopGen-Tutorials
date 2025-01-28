#!/bin/bash


mkdir -p trainingFvecs trainingFvecLogs
echo "calculating neutral stats for training sims"
diploSHIC fvecSim --numSubWins=11 haploid trainingSims/neut.msOut.gz trainingFvecs/neut.fvec &> trainingFvecLogs/neut.log
for (( i=0 ; i<11 ; i++ ))
do
    echo "calculating stats for hard_${i} for training sims"
    diploSHIC fvecSim --numSubWins=11 haploid trainingSims/hard_${i}.msOut.gz trainingFvecs/hard_${i}.fvec &> trainingFvecLogs/hard_${i}.log
    echo "calculating stats for soft_${i} for training sims"
    diploSHIC fvecSim --numSubWins=11 haploid trainingSims/soft_${i}.msOut.gz trainingFvecs/soft_${i}.fvec &> trainingFvecLogs/soft_${i}.log
done

mkdir -p testFvecs testFvecLogs
echo "calculating neutral stats for test sims"
diploSHIC fvecSim --numSubWins=11 haploid testSims/neut.msOut.gz testFvecs/neut.fvec &> testFvecLogs/neut.log
for (( i=0 ; i<11 ; i++ ))
do
    echo "calculating stats for hard_${i} for test sims"
    diploSHIC fvecSim --numSubWins=11 haploid testSims/hard_${i}.msOut.gz testFvecs/hard_${i}.fvec &> testFvecLogs/hard_${i}.log
    echo "calculating stats for soft_${i} for test sims"
    diploSHIC fvecSim --numSubWins=11 haploid testSims/soft_${i}.msOut.gz testFvecs/soft_${i}.fvec &> testFvecLogs/soft_${i}.log
done
