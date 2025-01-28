#!/bin/bash

sampleSize=20
numReps=100
recSites=100
theta=110
rho=110
alpha=1000
maxSweepAge=0.01
maxInitFreq=0.05
numSubWins=11

neutralCmd="discoal ${sampleSize} ${numReps} ${recSites} -t ${theta} -r ${rho}"
mkdir -p trainingSims
echo "simulating neutral evolution for training"
${neutralCmd} | gzip > trainingSims/neut.msOut.gz
i=0
for sweepLoc in 0.0454545 0.1363636 0.2272727 0.3181818 0.4090909 0.5000000 0.5909091 0.6818182 0.7727273 0.8636364 0.9545455;
do
    echo "simulating hard sweep at position $sweepLoc for training"
    ${neutralCmd} -ws 0 -a ${alpha} -Pu 0 ${maxSweepAge} -x ${sweepLoc} | gzip > trainingSims/hard_${i}.msOut.gz
    echo "simulating soft sweep at position $sweepLoc for training"
    ${neutralCmd} -ws 0 -a ${alpha} -Pu 0 ${maxSweepAge} -x ${sweepLoc} -Pf 0 ${maxInitFreq} | gzip > trainingSims/soft_${i}.msOut.gz
    i=$(( i + 1 ))
done

mkdir -p testSims
echo "simulating neutral evolution for testing"
${neutralCmd} | gzip > testSims/neut.msOut.gz
i=0
for sweepLoc in 0.0454545 0.1363636 0.2272727 0.3181818 0.4090909 0.5000000 0.5909091 0.6818182 0.7727273 0.8636364 0.9545455;
do
    echo "simulating hard sweep at position $sweepLoc for testing"
    ${neutralCmd} -ws 0 -a ${alpha} -Pu 0 ${maxSweepAge} -x ${sweepLoc} | gzip > testSims/hard_${i}.msOut.gz
    echo "simulating soft sweep at position $sweepLoc for testing"
    ${neutralCmd} -ws 0 -a ${alpha} -Pu 0 ${maxSweepAge} -x ${sweepLoc} -Pf 0 ${maxInitFreq} | gzip > testSims/soft_${i}.msOut.gz
    i=$(( i + 1 ))
done
