import json

json_path = "/mnt/i/DataSets/coco/2014/annotations/instances_val2014.json"
json_labels = json.load(open(json_path, 'r'))
print(json_labels["info"])