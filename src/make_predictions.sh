#!/bin/bash

WEIGHTS_DIR=../weights/kitti15
METHODS=(LEAStereo STSEarlyFusionConcat STSEarlyFusionConcat2 STSEarlyFusionConcat2Big LEAStereo_712 LEAStereo_4918 LEAStereo_9205 LEAStereo_712_LEAStereoOrig LEAStereo_4918_LEAStereoOrig LEAStereo_9205_LEAStereoOrig)

for i in "${METHODS[@]}"
do 
METHOD_NAME="$(cut -d'_' -f1 <<<"${i}")"
RESUME_METHOD="$(cut -d'_' -f3 <<<"${i}")" 
if [[ ${#RESUME_METHOD} -lt 2 ]] ; then
RESUME_METHOD=$METHOD_NAME
fi
echo $METHOD_NAME
echo $RESUME_METHOD
python gen_predictions.py splits.json ../datasets --resume "${WEIGHTS_DIR}/${i}/best.pth" --resume_method "${RESUME_METHOD}" --method "${METHOD_NAME}" --save_name "${i}"
done 
