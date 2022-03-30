#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
echo "submit for test"
python3 -u eval.py \
    --dataset cityscapes \
    --arch network.bialign.BiAlignNetDFNetV2SimpGateDeepLoss \
    --inference_mode whole \
    --scales 1.0 \
    --split test \
    --cv_split 0 \
    --dump_images \
    --ckpt_path ${2} \
    --snapshot ${1}
