import json
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

json_path = "/mnt/i/DataSets/coco/2014/annotations/instances_val2014.json"
img_path = "/mnt/i/DataSets/coco/2014/images/val2014"

coco = COCO(annotation_file=json_path)

ids = list(sorted(coco.imgs.keys()))
print("number of images: {}".format(len(ids)))

coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])
# print(coco_classes)

for img_id in ids[:3]:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    targets = coco.loadAnns(ann_ids)

    path = coco.loadImgs(img_id)[0]['file_name']
    # print(path)
    
    img = Image.open(os.path.join(img_path, path)).convert('RGB')
    draw = ImageDraw.Draw(img)
    for target in targets:
        x, y, w, h = target['bbox']
        x1, y1, x2, y2 = x, y, int(x+w), int(y+h)
        draw.rectangle((x1, y1, x2, y2))
        draw.text((x1, y1), coco_classes[target["category_id"]])
        
    plt.imshow(img)
    plt.show()