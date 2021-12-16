#!/bin/bash

# WEIGHTS_DIR=../weights/kitti15
# METHODS=(STSEarlyFusionConcat2Big_712 STSEarlyFusionConcat2Big_4918 STSEarlyFusionConcat2Big_9205 LEAStereo STSEarlyFusionConcat STSEarlyFusionConcat2 STSEarlyFusionConcat2Big STSEarlyFusionTimeMatch LEAStereo_712 LEAStereo_4918 LEAStereo_9205 LEAStereo_712_LEAStereoOrig LEAStereo_4918_LEAStereoOrig LEAStereo_9205_LEAStereoOrig)

# for i in "${METHODS[@]}"
# do 
# METHOD_NAME="$(cut -d'_' -f1 <<<"${i}")"
# RESUME_METHOD="$(cut -d'_' -f3 <<<"${i}")" 
# if [[ ${#RESUME_METHOD} -lt 2 ]] ; then
# RESUME_METHOD=$METHOD_NAME
# fi
# echo $METHOD_NAME
# echo $RESUME_METHOD
# python gen_predictions.py splits.json ../datasets --resume "${WEIGHTS_DIR}/${i}/best.pth" --resume_method "${RESUME_METHOD}" --method "${METHOD_NAME}" --save_name "${i}"
# done 

WEIGHTS_DIR=../weights/t1=t0/kitti15
METHODS=( STSEarlyFusionConcat STSEarlyFusionConcat2 STSEarlyFusionConcat2Big STSEarlyFusionTimeMatch )

for i in "${METHODS[@]}"
do
python gen_predictions.py splits.json ../datasets --resume "${WEIGHTS_DIR}/${i}/best.pth" --resume_method "${i}" --method "${i}" --save_name "${i}_t1=t0" -rk '{"l1":"l0","r1":"r0"}'
done 
