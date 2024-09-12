#!/bin/bash

if [[ ! -d "pytorch_object_detection" ]]; then
    mkdir -p "pytorch_object_detection"
fi
cd pytorch_object_detection
if [[ ! -f "mobilenet_v2.pth" ]]; then
    wget https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    mv mobilenet_v2-b0353104.pth mobilenet_v2.pth
fi

if [[ ! -f "resnet50.pth" ]]; then
    wget https://download.pytorch.org/models/resnet50-0676ba61.pth
    mv resnet50-0676ba61.pth resnet50.pth
fi

if [[ ! -f "fasterrcnn_resnet50_fpn_coco.pth" ]]; then
    wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
    mv fasterrcnn_resnet50_fpn_coco-258fb6c6.pth fasterrcnn_resnet50_fpn_coco.pth
fi

# 退出pytorch_object_detection目录
cd ..

# ====================================================================================

