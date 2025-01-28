#!/bin/bash

vcfFile=../preCookedData/CEU50.chr2LCT.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz

# The command below calculates feature vectors on a vcf file
# To see how it works, type:
# diploSHIC fvecVcf -h
# It is a bit involved so for now we are going to skip this step:
# diploSHIC fvecVcf haploid $vcfFile 2 243199373 $fvecFile --ancFileName /data/daniel/preCookedData/chr2.anc.fa &> realfvec.log

# Instead, let's cheat:
fvecFile=../preCookedData/CEU50.chr2LCT.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.fvec.gz

diploSHIC predict clf.json clf.weights.h5 $fvecFile real_preds.txt
