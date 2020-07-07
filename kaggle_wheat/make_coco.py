
import numpy as np
import pandas as pd
import json

data_dir='../global-wheat-detection/train'
train_csv='../global-wheat-detection/train.csv'


ratio=0.9

train_data=pd.read_csv(train_csv)

print(train_data)


image_ids=list(set(train_data['image_id']))

train_list=image_ids[:int(ratio*len(image_ids))]
val_list=image_ids[int(ratio*len(image_ids)):]

klasses=set(train_data['source'])



train_data_coco = {}
train_data_coco['licenses'] = []
train_data_coco['info'] = []
train_data_coco['categories'] = [{'id': 1, 'name': 'wheat', 'supercategory': 'wheat'}]
train_data_coco['images'] = []
train_data_coco['annotations'] = []

img_id=0
anno_id=0
for k,id in enumerate(train_list):


    file_name=data_dir+'/'+id+'.jpg'
    img_entry = {'file_name': file_name, 'id': img_id, 'height': 1024, 'width': 1024}
    train_data_coco['images'].append(img_entry)




    bboxes=train_data[train_data['image_id']==id]

    cur_label_message=data_dir+'/'+str(id)+'.jpg|'

    for box in bboxes['bbox']:
        curbox=box[1:-1].split(',')
        cur_box_info=[float(x) for x in curbox]
        xmin = int(cur_box_info[0])
        ymin = int(cur_box_info[1])
        xmax = int(cur_box_info[0]+cur_box_info[2])
        ymax = int(cur_box_info[1]+cur_box_info[3])

        anno_entry = {'image_id': img_id, 'category_id': 1, 'id': anno_id,\
                        'iscrowd': 0, 'area': int(xmax-xmin) * int(ymax-ymin),\
                        'bbox': [int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)]}
        train_data_coco['annotations'].append(anno_entry)

        anno_id+=1
    img_id+=1
with open('./Train_cocoStyle.json', 'w') as outfile:
    json.dump(train_data_coco, outfile,indent=2)


test_data_coco = {}
test_data_coco['licenses'] = []
test_data_coco['info'] = []
test_data_coco['categories'] = [{'id': 1, 'name': 'wheat', 'supercategory': 'wheat'}]
test_data_coco['images'] = []
test_data_coco['annotations'] = []

for k,id in enumerate(val_list):



    file_name = data_dir + '/' + id + '.jpg'
    img_entry = {'file_name': file_name, 'id': img_id, 'height': 1024, 'width': 1024}
    test_data_coco['images'].append(img_entry)


    bboxes=train_data[train_data['image_id']==id]

    for box in bboxes['bbox']:
        curbox = box[1:-1].split(',')
        cur_box_info = [float(x) for x in curbox]
        xmin = int(cur_box_info[0])
        ymin = int(cur_box_info[1])
        xmax = int(cur_box_info[0] + cur_box_info[2])
        ymax = int(cur_box_info[1] + cur_box_info[3])

        anno_entry = {'image_id': img_id, 'category_id': 1, 'id': anno_id, \
                      'iscrowd': 0, 'area': int(xmax - xmin) * int(ymax - ymin), \
                      'bbox': [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]}
        test_data_coco['annotations'].append(anno_entry)

        anno_id += 1
    img_id += 1


with open('./Val_cocoStyle.json', 'w') as outfile:
    json.dump(test_data_coco, outfile,indent=2)












