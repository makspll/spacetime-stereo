#!/bin/bash

WEIGHTS_DIR=../weights/kitti15
METHODS=(LEAStereo STSEarlyFusionConcat STSEarlyFusionConcat2 STSEarlyFusionConcat2Big)

python gen_predictions.py splits.json --method STSEarlyFusion4D --resume_method LEAStereoOrig

for i in "${METHODS[@]}"
do 
python gen_predictions.py splits.json --resume $WEIGHTS_DIR/$i/best.pth --resume_method $i --method $i
done 
