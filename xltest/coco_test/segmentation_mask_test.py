
import os
import random

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

random.seed(0)

json_path = "/mnt/i/DataSets/coco/2014/annotations/instances_val2014.json"
img_path = "/mnt/i/DataSets/coco/2014/images/val2014"

# random pallette
pallette = [0, 0, 0] + [random.randint(0, 255) for _ in range(255*3)]

# load coco data
coco = COCO(annotation_file=json_path)

# get all image index info
ids = list(sorted(coco.imgs.keys()))
print("number of images: {}".format(len(ids)))

# get all coco class labels
coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])

# 遍历前三张图像
for img_id in ids[:3]:
    # 获取对应图像id的所有annotations idx信息
    ann_ids = coco.getAnnIds(imgIds=img_id)
    # 根据annotations idx信息获取所有标注信息
    targets = coco.loadAnns(ann_ids)

    # get image file name
    path = coco.loadImgs(img_id)[0]['file_name']
    # read image
    img = Image.open(os.path.join(img_path, path)).convert('RGB')
    img_w, img_h = img.size

    masks = []
    cats = []
    for target in targets:
        cats.append(target["category_id"])  # get object class id
        polygons = target["segmentation"]   # get object polygons
        rles = coco_mask.frPyObjects(polygons, img_h, img_w)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = mask.any(axis=2)
        masks.append(mask)

    cats = np.array(cats, dtype=np.int32)
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)

    # merge all instance masks into a single segmentation map
    # with its corresponding categories
    target = (masks * cats[:, None, None]).max(axis=0)
    # discard overlapping instances
    target[masks.sum(0) > 1] = 255
    target = Image.fromarray(target.astype(np.uint8))

    target.putpalette(pallette)
    plt.imshow(target)
    plt.show()
